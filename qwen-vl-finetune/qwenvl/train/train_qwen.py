# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.data.hf_loader import (
    load_filtered_pyranet_samples,
    load_hf_verilog_completion_samples,
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class TextSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        samples: Sequence[Dict[str, str]],
        model_max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.samples = list(samples)
        self.model_max_length = model_max_length

    def __len__(self) -> int:
        return len(self.samples)

    def _format_chat(self, instruction: str, response: str) -> Tuple[str, str]:
        if getattr(self.tokenizer, "chat_template", None):
            prompt_messages = [{"role": "user", "content": instruction}]
            full_messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = self.tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                )
                full_text = self.tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                )
            return prompt_text, full_text

        prompt_text = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
        full_text = f"{prompt_text}{response}"
        return prompt_text, full_text

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        prompt_text, full_text = self._format_chat(
            sample["instruction"], sample["response"]
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.model_max_length,
        )["input_ids"]
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.model_max_length,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollatorForTextSFT:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(
        self, instances: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in instances]
        attention_mask = [item["attention_mask"] for item in instances]
        labels = [item["labels"] for item in instances]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def _is_vl_model_name(model_name_or_path: str) -> bool:
    return "vl" in (model_name_or_path or "").lower()


def _resolve_lora_targets(model: torch.nn.Module, candidates: Sequence[str]) -> List[str]:
    available = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        available.add(leaf)
    selected = [m for m in candidates if m in available]
    return selected or list(candidates)


def _parse_dataset_tokens(dataset_use: str) -> List[str]:
    tokens = [token.strip() for token in (dataset_use or "").split(",") if token.strip()]
    return tokens or ["hf_pyranet"]


def _is_pyranet_token(token: str) -> bool:
    return (token or "").strip().lower() in {
        "hf_pyranet",
        "hf_pyranet_verilog",
        "pyranet_hf",
    }


def _is_verilog_github_token(token: str) -> bool:
    return (token or "").strip().lower() in {"hf_verilog_github", "hf_verilog_gihub"}


def _load_text_sft_samples(data_args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_tokens = _parse_dataset_tokens(getattr(data_args, "dataset_use", ""))
    hf_max_samples = int(getattr(data_args, "hf_max_samples", -1))
    max_samples = hf_max_samples if hf_max_samples > 0 else None

    min_rank = int(getattr(data_args, "hf_pyranet_min_rank", 16))
    allowed_complexities = getattr(
        data_args, "hf_pyranet_allowed_complexities", "Basic,Expert"
    )
    require_compile_success = bool(
        getattr(data_args, "hf_pyranet_require_compile_success", True)
    )
    require_empty_compile_results = bool(
        getattr(data_args, "hf_pyranet_require_empty_compile_results", True)
    )

    # Backward compatibility for older argument names.
    if getattr(data_args, "hf_min_rank", None) is not None:
        min_rank = int(data_args.hf_min_rank)
    if getattr(data_args, "hf_allowed_complexities", None) is not None:
        allowed_complexities = data_args.hf_allowed_complexities
    if getattr(data_args, "hf_require_compile_success", None) is not None:
        require_compile_success = bool(data_args.hf_require_compile_success)
    if getattr(data_args, "hf_require_empty_compile_results", None) is not None:
        require_empty_compile_results = bool(
            data_args.hf_require_empty_compile_results
        )

    all_samples: List[Dict[str, Any]] = []
    load_logs: List[Dict[str, Any]] = []

    for token in dataset_tokens:
        token_lc = token.strip().lower()

        if _is_pyranet_token(token_lc):
            dataset_repo = getattr(
                data_args,
                "hf_pyranet_repo",
                getattr(data_args, "hf_dataset_repo", "bnadimi/PyraNet-Verilog"),
            )
            dataset_split = getattr(
                data_args,
                "hf_pyranet_split",
                getattr(data_args, "hf_dataset_split", "train"),
            )
            samples, stats = load_filtered_pyranet_samples(
                dataset_repo=dataset_repo,
                split=dataset_split,
                min_rank=min_rank,
                allowed_complexities=allowed_complexities,
                require_compile_success=require_compile_success,
                require_empty_compile_results=require_empty_compile_results,
                max_samples=max_samples,
            )
            all_samples.extend(samples)
            load_logs.append(
                {
                    "token": token,
                    "dataset_repo": stats["dataset_repo"],
                    "split": stats["split"],
                    "total_samples": stats["total_samples"],
                    "filtered_samples": stats["filtered_samples"],
                }
            )
            continue

        if _is_verilog_github_token(token_lc):
            dataset_repo = getattr(
                data_args,
                "hf_verilog_github_repo",
                getattr(data_args, "hf_dataset_repo", "shailja/Verilog_GitHub"),
            )
            dataset_split = getattr(
                data_args,
                "hf_verilog_github_split",
                getattr(data_args, "hf_dataset_split", "train"),
            )

            samples, stats = load_hf_verilog_completion_samples(
                dataset_repo=dataset_repo,
                split=dataset_split,
                max_samples=max_samples,
                code_field="text",
            )
            all_samples.extend(samples)
            load_logs.append(
                {
                    "token": token,
                    "dataset_repo": stats["dataset_repo"],
                    "split": stats["split"],
                    "total_samples": stats["total_samples"],
                    "filtered_samples": stats["filtered_samples"],
                }
            )
            continue

        raise ValueError(
            f"Unsupported dataset token for text training: {token}. "
            "Use hf_pyranet, hf_verilog_github or comma-separated combinations."
        )

    return all_samples, load_logs


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_name = model_args.model_name_or_path
    is_vl = _is_vl_model_name(model_name)
    model_dtype = None
    if training_args.bf16:
        model_dtype = torch.bfloat16
    elif training_args.fp16:
        model_dtype = torch.float16

    if is_vl:
        if "qwen3" in model_name.lower() and "a" in Path(model_name.rstrip("/")).name.lower():
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=model_dtype,
                trust_remote_code=getattr(model_args, "trust_remote_code", True),
            )
            data_args.model_type = "qwen3vl"
        elif "qwen3" in model_name.lower():
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=model_dtype,
                trust_remote_code=getattr(model_args, "trust_remote_code", True),
            )
            data_args.model_type = "qwen3vl"
        elif "qwen2.5" in model_name.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=model_dtype,
                trust_remote_code=getattr(model_args, "trust_remote_code", True),
            )
            data_args.model_type = "qwen2.5vl"
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=model_dtype,
                trust_remote_code=getattr(model_args, "trust_remote_code", True),
            )
            data_args.model_type = "qwen2vl"

        print(
            f"the initialized model is {model_name} the class is {model.__class__.__name__}"
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=getattr(model_args, "trust_remote_code", True),
        )

        if data_args.data_flatten or data_args.data_packing:
            replace_qwen2_vl_attention_class()
        model.config.use_cache = False

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=getattr(model_args, "trust_remote_code", True),
        )

        # Keep tokenizer settings consistent across Trainer and data collator paths.
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.model_max_length = training_args.model_max_length
            processor.tokenizer.padding_side = tokenizer.padding_side
            if processor.tokenizer.pad_token_id is None and tokenizer.pad_token_id is not None:
                processor.tokenizer.pad_token = tokenizer.pad_token

        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model, TaskType

            print("LoRA enabled")
            for p in model.parameters():
                p.requires_grad = False

            target_modules = _resolve_lora_targets(
                model,
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            lora_config = LoraConfig(
                r=training_args.lora_r or 64,
                lora_alpha=training_args.lora_alpha or 128,
                lora_dropout=training_args.lora_dropout or 0.05,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        else:
            set_model(model_args, model)

            if torch.distributed.get_rank() == 0:
                model.visual.print_trainable_parameters()
                model.model.print_trainable_parameters()

        data_module = make_supervised_data_module(processor, data_args=data_args)
        trainer = Trainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        model.config.use_cache = True
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

    else:
        print(
            f"the initialized model is {model_name} the class is AutoModelForCausalLM (text-only)"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=getattr(model_args, "trust_remote_code", True),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=model_dtype,
            trust_remote_code=getattr(model_args, "trust_remote_code", True),
        )
        model.config.use_cache = False

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        if training_args.lora_enable:
            from peft import LoraConfig, TaskType, get_peft_model

            for p in model.parameters():
                p.requires_grad = False

            target_modules = _resolve_lora_targets(
                model,
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            if local_rank in (-1, 0):
                try:
                    model.print_trainable_parameters()
                except Exception:
                    pass

        samples, load_logs = _load_text_sft_samples(data_args)
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            for entry in load_logs:
                logging.warning(
                    "Loaded %s[%s] (%s): total=%d filtered=%d",
                    entry["dataset_repo"],
                    entry["split"],
                    entry["token"],
                    entry["total_samples"],
                    entry["filtered_samples"],
                )

        train_dataset = TextSFTDataset(
            tokenizer=tokenizer,
            samples=samples,
            model_max_length=training_args.model_max_length,
        )
        data_collator = DataCollatorForTextSFT(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        model.config.use_cache = True
        if training_args.lora_enable:
            trainer.model.save_pretrained(training_args.output_dir)
        else:
            safe_save_model_for_hf_trainer(
                trainer=trainer, output_dir=training_args.output_dir
            )
        tokenizer.save_pretrained(training_args.output_dir)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Avoid NCCL warnings on shutdown in torchrun dryruns and short runs.
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    # Override via env var: ATTN_IMPL=flash_attention_2|sdpa|eager
    attn_impl = os.environ.get("ATTN_IMPL", "sdpa")
    train(attn_implementation=attn_impl)
