from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelSpec:
    id: str
    alias: str
    provider: str


def load_models_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_models(models_cfg: Dict[str, Any], use_key: str) -> List[ModelSpec]:
    if use_key not in models_cfg:
        raise ValueError(f"models.use '{use_key}' not found in models config.")
    specs: List[ModelSpec] = []
    for entry in models_cfg[use_key]:
        specs.append(ModelSpec(id=entry["id"], alias=entry["alias"], provider=entry["provider"]))
    return specs


class ModelRunner:
    def __init__(self, generation_cfg: Dict[str, Any]) -> None:
        self.generation_cfg = generation_cfg
        self._hf_cache: Dict[str, Dict[str, Any]] = {}
        self._openai_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None

    def get_tokenizer(self, model_spec: ModelSpec) -> Optional[Any]:
        if model_spec.provider != "hf":
            return None
        self._ensure_hf_loaded(model_spec.id)
        return self._hf_cache[model_spec.id].get("tokenizer")

    def generate(self, model_spec: ModelSpec, prompt: str) -> str:
        if model_spec.provider == "hf":
            return self._generate_hf(model_spec.id, prompt)
        if model_spec.provider == "openai":
            return self._generate_openai(model_spec.id, prompt)
        if model_spec.provider == "anthropic":
            return self._generate_anthropic(model_spec.id, prompt)
        raise ValueError(f"Unknown provider: {model_spec.provider}")

    def _ensure_hf_loaded(self, model_id: str) -> None:
        if model_id in self._hf_cache:
            return
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        lower_model_id = model_id.lower()
        is_phi3_mini = lower_model_id.startswith("microsoft/phi-3-mini")
        is_phi3_small = lower_model_id.startswith("microsoft/phi-3-small")
        is_ministral3 = lower_model_id.startswith("mistralai/ministral-3-")

        # Phi-3-small requires remote code; Phi-3-mini is handled without remote code.
        trust_remote_code = not is_phi3_mini
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            cfg = None

        if cfg is not None:
            cfg_name = cfg.__class__.__name__.lower()
            model_type = str(getattr(cfg, "model_type", "")).lower()
            is_ministral3 = is_ministral3 or cfg_name == "mistral3config" or model_type == "mistral3"

        if is_ministral3:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            if tokenizer is not None and tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._hf_cache[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "processor": processor,
                "kind": "mistral3",
            }
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, use_fast=False, trust_remote_code=trust_remote_code
            )

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": trust_remote_code,
        }

        # Only Phi-3-mini gets explicit config patch.
        if is_phi3_mini:
            if cfg is None:
                cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            if isinstance(getattr(cfg, "rope_scaling", None), dict):
                rs = dict(cfg.rope_scaling)
                if "type" not in rs:
                    rs["type"] = rs.get("rope_type", "longrope")
                cfg.rope_scaling = rs
            model_kwargs["config"] = cfg

        # Important: Phi-3-small must load native config from remote code (no explicit config passed).
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._hf_cache[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "processor": None,
            "kind": "causal_lm",
        }

    def _prepare_hf_prompt_inputs(self, model_id: str, prompt: str) -> Dict[str, Any]:
        import torch

        cache = self._hf_cache[model_id]
        model = cache["model"]
        tokenizer = cache.get("tokenizer")
        processor = cache.get("processor")

        if cache.get("kind") == "mistral3":
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, return_tensors="pt")
            return {k: v.to(model.device) for k, v in inputs.items()}

        if getattr(tokenizer, "chat_template", None):
            try:
                chat_inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # HF versions differ: this may return Tensor or BatchEncoding/dict.
                if isinstance(chat_inputs, torch.Tensor):
                    input_ids = chat_inputs
                    attention_mask = torch.ones_like(input_ids)
                else:
                    input_ids = chat_inputs["input_ids"]
                    attention_mask = chat_inputs.get("attention_mask")
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                return {
                    "input_ids": input_ids.to(model.device),
                    "attention_mask": attention_mask.to(model.device),
                }
            except Exception:
                # Some tokenizer chat templates require extra Jinja variables.
                # Fall back to direct tokenization of the already-formatted prompt.
                pass

        tokenized = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(model.device) for k, v in tokenized.items()}

    def _generate_hf(self, model_id: str, prompt: str) -> str:
        import torch

        self._ensure_hf_loaded(model_id)
        cache = self._hf_cache[model_id]
        model = cache["model"]
        tokenizer = cache.get("tokenizer")
        processor = cache.get("processor")
        inputs = self._prepare_hf_prompt_inputs(model_id, prompt)

        do_sample = bool(self.generation_cfg.get("do_sample", False))
        gen_cfg = {
            "max_new_tokens": int(self.generation_cfg.get("max_new_tokens", 512)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_cfg["temperature"] = float(self.generation_cfg.get("temperature", 0.0))
            gen_cfg["top_p"] = float(self.generation_cfg.get("top_p", 1.0))

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_cfg)
        gen_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        if cache.get("kind") == "mistral3":
            return processor.batch_decode(gen_tokens.unsqueeze(0), skip_special_tokens=True)[0]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def hf_continuation_logprob(self, model_spec: ModelSpec, prompt: str, continuation: str) -> float:
        import torch

        if model_spec.provider != "hf":
            raise ValueError("Continuation logprob is only supported for HF models.")

        self._ensure_hf_loaded(model_spec.id)
        cache = self._hf_cache[model_spec.id]
        model = cache["model"]
        tokenizer = cache.get("tokenizer")
        prompt_inputs = self._prepare_hf_prompt_inputs(model_spec.id, prompt)
        prompt_ids = prompt_inputs["input_ids"]
        continuation_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            model.device
        )
        if continuation_ids.shape[-1] == 0:
            return 0.0

        full_ids = torch.cat([prompt_ids, continuation_ids], dim=1)
        full_attention = torch.ones_like(full_ids, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=full_ids, attention_mask=full_attention).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[-1]
        total = 0.0
        for pos in range(prompt_len, full_ids.shape[-1]):
            token_id = int(full_ids[0, pos].item())
            total += float(log_probs[0, pos - 1, token_id].item())
        return total

    def hf_first_token_prob(self, model_spec: ModelSpec, prompt: str, token_text: str) -> float:
        import torch

        if model_spec.provider != "hf":
            raise ValueError("First-token probability is only supported for HF models.")

        self._ensure_hf_loaded(model_spec.id)
        cache = self._hf_cache[model_spec.id]
        model = cache["model"]
        tokenizer = cache.get("tokenizer")
        inputs = self._prepare_hf_prompt_inputs(model_spec.id, prompt)
        with torch.no_grad():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        next_logits = logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_logits, dim=-1)

        token_ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            return 0.0
        # Use first token only by design for judge calibration.
        token_id = int(token_ids[0])
        return float(probs[token_id].item())

    def hf_first_token_candidate_probs(
        self, model_spec: ModelSpec, prompt: str, candidates: List[str]
    ) -> Dict[str, Any]:
        import torch

        if model_spec.provider != "hf":
            raise ValueError("First-token probability is only supported for HF models.")

        self._ensure_hf_loaded(model_spec.id)
        cache = self._hf_cache[model_spec.id]
        model = cache["model"]
        tokenizer = cache.get("tokenizer")
        inputs = self._prepare_hf_prompt_inputs(model_spec.id, prompt)
        with torch.no_grad():
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        next_logits = logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_logits, dim=-1)

        out_probs: Dict[str, float] = {}
        out_token_ids: Dict[str, int | None] = {}
        for cand in candidates:
            token_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if not token_ids:
                out_probs[cand] = 0.0
                out_token_ids[cand] = None
                continue
            token_id = int(token_ids[0])
            out_probs[cand] = float(probs[token_id].item())
            out_token_ids[cand] = token_id

        return {
            "tokenizer_id": str(getattr(tokenizer, "name_or_path", model_spec.id)),
            "candidate_probs": out_probs,
            "candidate_first_token_ids": out_token_ids,
        }

    @staticmethod
    def llr_to_prob(llr_margin: float) -> float:
        if llr_margin >= 0:
            z = math.exp(-llr_margin)
            return 1.0 / (1.0 + z)
        z = math.exp(llr_margin)
        return z / (1.0 + z)

    def _generate_openai(self, model_id: str, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for provider=openai") from exc

        if self._openai_client is None:
            self._openai_client = OpenAI()
        resp = self._openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(self.generation_cfg.get("temperature", 0.0)),
            top_p=float(self.generation_cfg.get("top_p", 1.0)),
            max_tokens=int(self.generation_cfg.get("max_new_tokens", 512)),
        )
        return resp.choices[0].message.content

    def _generate_anthropic(self, model_id: str, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for provider=anthropic") from exc

        if self._anthropic_client is None:
            self._anthropic_client = Anthropic()
        resp = self._anthropic_client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(self.generation_cfg.get("temperature", 0.0)),
            max_tokens=int(self.generation_cfg.get("max_new_tokens", 512)),
        )
        return resp.content[0].text

    def unload_model(self, model_id: str) -> None:
        """Remove a model from the HF cache and release GPU memory.

        Call this between sequential judge phases (e.g. after finishing all
        WildGuard judgments and before loading ShieldGemma) so that only one
        large model occupies GPU VRAM at a time.
        """
        import gc

        if model_id not in self._hf_cache:
            return
        cached = self._hf_cache.pop(model_id)
        model = cached.get("model")
        if model is not None:
            del model
        tokenizer = cached.get("tokenizer")
        if tokenizer is not None:
            del tokenizer
        processor = cached.get("processor")
        if processor is not None:
            del processor
        del cached
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
