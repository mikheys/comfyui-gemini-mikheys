import logging
import os
import random
from io import BytesIO

import google.generativeai as genai
from google.api_core import exceptions, retry
from PIL import Image
from torch import Tensor

from .utils import images_to_pillow, temporary_env_var, pillow_to_tensor


class NanoBananaNode:

    @classmethod
    def INPUT_TYPES(cls):  # noqa
        seed = random.randint(1, 2**31)

        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "safety_settings": (["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE"],),
                "model": (["gemini-2.5-flash-image-preview"],),
            },
            "optional": {
                "api_key": ("STRING", {}),
                "proxy": ("STRING", {}),
                "image_to_edit": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "system_instruction": ("STRING", {}),
                "error_fallback_value": ("STRING", {"lazy": True}),
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 1, "step": 0.05}),
                "num_predict": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit_image"

    CATEGORY = "Gemini"

    def __init__(self):
        self.image_output: Tensor | None = None

    def edit_image(self, **kwargs):
        return (self.image_output,)

    def check_lazy_status(
        self,
        prompt: str,
        safety_settings: str,
        model: str,
        api_key: str | None = None,
        proxy: str | None = None,
        image_to_edit: Tensor | list[Tensor] | None = None,
        reference_image_1: Tensor | list[Tensor] | None = None,
        reference_image_2: Tensor | list[Tensor] | None = None,
        reference_image_3: Tensor | list[Tensor] | None = None,
        reference_image_4: Tensor | list[Tensor] | None = None,
        reference_image_5: Tensor | list[Tensor] | None = None,
        system_instruction: str | None = None,
        error_fallback_value: str | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        **kwargs,
    ):
        self.image_output = None
        self.text_output = ""
        if not system_instruction:
            system_instruction = None
        
        contents = [prompt]

        # Process reference images first
        if reference_image_1 is not None:
            contents.extend(images_to_pillow(reference_image_1))
        if reference_image_2 is not None:
            contents.extend(images_to_pillow(reference_image_2))
        if reference_image_3 is not None:
            contents.extend(images_to_pillow(reference_image_3))
        if reference_image_4 is not None:
            contents.extend(images_to_pillow(reference_image_4))
        if reference_image_5 is not None:
            contents.extend(images_to_pillow(reference_image_5))

        # Process the main image to be edited LAST
        if image_to_edit is not None:
            contents.extend(images_to_pillow(image_to_edit))

        if api_key:
            genai.configure(api_key=api_key, transport="rest")
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(transport="rest")

        model = genai.GenerativeModel(model, safety_settings=safety_settings, system_instruction=system_instruction)

        retry_config = retry.Retry(
            predicate=retry.if_exception_type(
                exceptions.InternalServerError,
                exceptions.ResourceExhausted,
                exceptions.ServiceUnavailable,
            )
        )

        generation_config = genai.GenerationConfig()
        if temperature is not None and temperature >= 0:
            generation_config.temperature = temperature
        if num_predict is not None and num_predict > 0:
            generation_config.max_output_tokens = num_predict

        try:
            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                # Wrap the function with our retry configuration
                retry_wrapped_generate_content = retry_config(model.generate_content)
                # Call the wrapped function
                response = retry_wrapped_generate_content(contents=contents, generation_config=generation_config)
            
            output_images = []
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        output_images.append(image)
                    except Exception as e:
                        logging.getLogger("ComfyUI-Gemini").error(f"Failed to identify image from response part: {e}")
                        logging.getLogger("ComfyUI-Gemini").error(f"Response part data: {part.inline_data.data}")
            
            if output_images:
                self.image_output = pillow_to_tensor(output_images)

        except Exception:
            if error_fallback_value is None:
                logging.getLogger("ComfyUI-Gemini").debug("ComfyUI-Gemini: exception occurred:", exc_info=True)
                return (None,)
            if error_fallback_value == "":
                raise
        return []


NODE_CLASS_MAPPINGS = {
    "Nano_Banana": NanoBananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nano_Banana": "Nano banana",
}
