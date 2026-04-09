"""Qwen Vision-Language Model Wrapper."""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class QwenVLModel:
    """Qwen Vision-Language Model for multimodal question answering.

    Attributes:
        modality (str): The modality type, always 'multimodal'.
        model (Qwen3VLForConditionalGeneration): The loaded Qwen3-VL model.
        processor (AutoProcessor): The processor for input formatting.
        device (str): The device to run the model on (e.g., 'cuda:0', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        min_pixels (int): Minimum number of pixels for image processing.
        max_pixels (int): Maximum number of pixels for image processing.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """Initialize the Qwen3-VL model.

        Args:
            model_name: HuggingFace model identifier (e.g., 'Qwen/Qwen3-VL-8B-Instruct').
            device: Device to run the model on (default: 'cuda').
            torch_dtype: Data type for model weights (default: 'bfloat16').
            attn_implementation: Attention implementation (default: 'sdpa').
            max_new_tokens: Maximum number of tokens to generate (default: 512).
            min_pixels: Minimum number of pixels for image processing (default: 256*28*28).
            max_pixels: Maximum number of pixels for image processing (default: 1280*28*28).

        Raises:
            ValueError: If model_name is not provided.
        """
        if not model_name:
            raise ValueError("model_name must be provided")

        self.modality = "multimodal"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(
        self, question: str, imgs: Optional[list[Image.Image]] = None
    ) -> str:
        """Generate an answer to a question with optional images.

        Args:
            question: The text question to answer.
            imgs: Optional list of PIL Images to use as visual context.

        Returns:
            The generated text answer as a string.
        """
        with torch.no_grad():
            if imgs is None:
                imgs = []

            content = []
            for img in imgs:
                content.append({"type": "image", "image": img})

            content.append({"type": "text", "text": question})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

            inputs.pop("token_type_ids", None)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0]
