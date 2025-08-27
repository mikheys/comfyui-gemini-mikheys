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

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "edit_image"

    CATEGORY = "Gemini"

    def __init__(self):
        self.image_output: Tensor | None = None
        self.status_output: str | None = None

    def edit_image(self, **kwargs):
        return (self.image_output, self.status_output,)

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
        self.status_output = "Error: Unknown"
        output_images = []

        try:
            if not system_instruction:
                system_instruction = None
            
            contents = [prompt]

            # Process reference images first
            ref_images = [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5]
            for img in ref_images:
                if img is not None:
                    contents.extend(images_to_pillow(img))

            # Process the main image to be edited LAST
            if image_to_edit is not None:
                contents.extend(images_to_pillow(image_to_edit))

            if api_key:
                genai.configure(api_key=api_key, transport="rest")
            elif "GOOGLE_API_KEY" in os.environ:
                genai.configure(transport="rest")

            model_instance = genai.GenerativeModel(model, safety_settings=safety_settings, system_instruction=system_instruction)

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

            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                retry_wrapped_generate_content = retry_config(model_instance.generate_content)
                response = retry_wrapped_generate_content(contents=contents, generation_config=generation_config)
            
            if not response.candidates:
                self.status_output = "Error: Prompt or image was blocked by safety filters."
                logging.warning(self.status_output)
            else:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None and part.inline_data.data:
                        try:
                            image = Image.open(BytesIO(part.inline_data.data))
                            output_images.append(image)
                        except Exception as e:
                            log_msg = f"Could not decode image from API response: {e}"
                            logging.warning(log_msg)
                            self.status_output = f"Error: {log_msg}"

                if output_images:
                    self.status_output = "Complete"
                else:
                    # This case happens if the response had candidates but no valid image data
                    if self.status_output.startswith("Error:"):
                        pass # Keep the more specific error
                    else:
                        self.status_output = "Error: No image data in API response."

        except Exception as e:
            self.status_output = f"Error: {e}"
            logging.error(f"An exception occurred in NanoBananaNode: {e}", exc_info=True)

        # Fallback to a black image if there was any issue
        if not output_images:
            width, height = 512, 512 # Default size
            if image_to_edit is not None:
                height, width = image_to_edit.shape[1], image_to_edit.shape[2]
            black_image = Image.new('RGB', (width, height), (0, 0, 0))
            output_images.append(black_image)
        
        self.image_output = pillow_to_tensor(output_images)
        return []


NODE_CLASS_MAPPINGS = {
    "Nano_Banana": NanoBananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nano_Banana": "Nano banana",
}
