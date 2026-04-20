import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset


DEFAULT_DATASET_REPO = "bnadimi/PyraNet-Verilog"
DEFAULT_DATASET_SPLIT = "train"


def parse_allowed_complexities(
    allowed_complexities: Optional[Sequence[str] | str],
) -> Tuple[str, ...]:
    if allowed_complexities is None:
        return ("Basic", "Expert")
    if isinstance(allowed_complexities, str):
        parts = [part.strip() for part in allowed_complexities.split(",")]
        return tuple(part for part in parts if part)
    return tuple(str(part).strip() for part in allowed_complexities if str(part).strip())


def _parse_description(description_field: Any) -> Optional[Dict[str, Any]]:
    if isinstance(description_field, dict):
        return description_field
    if isinstance(description_field, str):
        try:
            parsed = json.loads(description_field)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_rank(rank_value: Any) -> Optional[int]:
    try:
        return int(rank_value)
    except (TypeError, ValueError):
        return None


def _is_selected(
    desc_json: Dict[str, Any],
    min_rank: int,
    allowed_complexities: Tuple[str, ...],
    require_compile_success: bool,
    require_empty_compile_results: bool,
) -> bool:
    compile_status = desc_json.get("compile_status")
    compile_results = desc_json.get("compile_results")
    rank = _normalize_rank(desc_json.get("rank"))
    complexity = str(desc_json.get("complexity", "")).strip()

    if rank is None:
        return False

    if require_compile_success and compile_status != "No error!":
        return False

    if require_empty_compile_results and compile_results != "":
        return False

    if rank < min_rank:
        return False

    if allowed_complexities and complexity not in allowed_complexities:
        return False

    return True


def build_verilog_instruction(spec_text: str) -> str:
    spec_text = (spec_text or "").strip()
    return (
        "Write synthesizable Verilog HDL code for the following specification.\n\n"
        f"Specification:\n{spec_text}\n\n"
        "Return only Verilog code."
    )


def build_verilog_completion_instruction(prefix_text: str) -> str:
    prefix_text = (prefix_text or "").rstrip()
    return (
        "Continue the following Verilog code exactly from where it ends.\n"
        "Return only the continuation text (no explanation, no markdown).\n\n"
        "Verilog prefix:\n"
        f"{prefix_text}"
    )


def _extract_code_text(row: Dict[str, Any], preferred_field: Optional[str] = None) -> str:
    if preferred_field:
        value = row.get(preferred_field)
        if isinstance(value, str):
            return value

    for field in (
        "code",
        "verilog",
        "sv_code",
        "text",
        "content",
        "module",
        "rtl",
    ):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value

    for value in row.values():
        if isinstance(value, str) and "module" in value.lower() and value.strip():
            return value

    return ""


def _split_completion_target(
    code_text: str,
    prefix_ratio: float = 0.6,
    min_prefix_chars: int = 32,
    min_suffix_chars: int = 32,
) -> Optional[Tuple[str, str]]:
    text = (code_text or "").strip()
    if not text:
        return None

    lines = text.splitlines()
    if len(lines) >= 4:
        split_idx = int(len(lines) * prefix_ratio)
        split_idx = max(1, min(len(lines) - 1, split_idx))
        prefix = "\n".join(lines[:split_idx]).rstrip()
        suffix = "\n".join(lines[split_idx:]).lstrip()
    else:
        split_idx = int(len(text) * prefix_ratio)
        split_idx = max(1, min(len(text) - 1, split_idx))
        prefix = text[:split_idx].rstrip()
        suffix = text[split_idx:].lstrip()

    if len(prefix) < min_prefix_chars or len(suffix) < min_suffix_chars:
        return None
    return prefix, suffix


def load_filtered_pyranet_samples(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    split: str = DEFAULT_DATASET_SPLIT,
    min_rank: int = 16,
    allowed_complexities: Optional[Sequence[str] | str] = ("Basic", "Expert"),
    require_compile_success: bool = True,
    require_empty_compile_results: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    allowed_complexities_tuple = parse_allowed_complexities(allowed_complexities)
    dataset = load_dataset(dataset_repo, split=split)

    samples: List[Dict[str, Any]] = []
    rank_counter = Counter()
    complexity_counter = Counter()

    total_samples = 0
    parsed_description_samples = 0

    for row in dataset:
        total_samples += 1
        desc_json = _parse_description(row.get("description"))
        if desc_json is None:
            continue

        parsed_description_samples += 1

        if not _is_selected(
            desc_json,
            min_rank=min_rank,
            allowed_complexities=allowed_complexities_tuple,
            require_compile_success=require_compile_success,
            require_empty_compile_results=require_empty_compile_results,
        ):
            continue

        rank = int(desc_json.get("rank"))
        complexity = str(desc_json.get("complexity", "")).strip()
        instruction = build_verilog_instruction(str(desc_json.get("description", "")))
        response = str(row.get("code", ""))

        if not instruction.strip() or not response.strip():
            continue

        rank_counter[rank] += 1
        complexity_counter[complexity] += 1

        samples.append(
            {
                "instruction": instruction,
                "response": response,
                "rank": rank,
                "complexity": complexity,
                "compile_status": desc_json.get("compile_status"),
                "compile_results": desc_json.get("compile_results"),
            }
        )

        if max_samples is not None and max_samples > 0 and len(samples) >= max_samples:
            break

    stats = {
        "dataset_repo": dataset_repo,
        "split": split,
        "total_samples": total_samples,
        "parsed_description_samples": parsed_description_samples,
        "filtered_samples": len(samples),
        "rank_counter": rank_counter,
        "complexity_counter": complexity_counter,
        "allowed_complexities": allowed_complexities_tuple,
        "min_rank": min_rank,
        "require_compile_success": require_compile_success,
        "require_empty_compile_results": require_empty_compile_results,
    }

    return samples, stats


def load_hf_verilog_completion_samples(
    dataset_repo: str,
    split: str = DEFAULT_DATASET_SPLIT,
    max_samples: Optional[int] = None,
    code_field: Optional[str] = None,
    prefix_ratio: float = 0.6,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dataset = load_dataset(dataset_repo, split=split)

    samples: List[Dict[str, Any]] = []
    total_samples = 0
    code_like_samples = 0

    for row in dataset:
        total_samples += 1
        code_text = _extract_code_text(row, preferred_field=code_field)
        if not code_text.strip():
            continue

        code_like_samples += 1
        split_pair = _split_completion_target(code_text, prefix_ratio=prefix_ratio)
        if split_pair is None:
            continue

        prefix, suffix = split_pair
        instruction = build_verilog_completion_instruction(prefix)
        response = suffix
        if not instruction.strip() or not response.strip():
            continue

        samples.append(
            {
                "instruction": instruction,
                "response": response,
            }
        )

        if max_samples is not None and max_samples > 0 and len(samples) >= max_samples:
            break

    stats = {
        "dataset_repo": dataset_repo,
        "split": split,
        "total_samples": total_samples,
        "code_like_samples": code_like_samples,
        "filtered_samples": len(samples),
        "code_field": code_field,
        "prefix_ratio": prefix_ratio,
    }
    return samples, stats


def to_qwenvl_conversation_samples(
    samples: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    converted = []
    for sample in samples:
        converted.append(
            {
                "conversations": [
                    {"from": "human", "value": sample["instruction"]},
                    {"from": "gpt", "value": sample["response"]},
                ]
            }
        )
    return converted