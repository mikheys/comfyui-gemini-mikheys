import logging
import os
import random
from io import BytesIO

from google import genai
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
                "model": ([
                    "gemini-2.5-flash-image-preview",
                ],),
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
#                "system_instruction": ("STRING", {}), # <-- не влияет на результат
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 1, "step": 0.05}),
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
        model: str,
        safety_settings: str | None = None,
        api_key: str | None = None,
        proxy: str | None = None,
        image_to_edit: Tensor | list[Tensor] | None = None,
        reference_image_1: Tensor | list[Tensor] | None = None,
        reference_image_2: Tensor | list[Tensor] | None = None,
        reference_image_3: Tensor | list[Tensor] | None = None,
        reference_image_4: Tensor | list[Tensor] | None = None,
        reference_image_5: Tensor | list[Tensor] | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        self.image_output = None
        self.status_output = "Error: Unknown"
        output_images = []

        try:
            # Подготовка клиента
            if api_key:
                client = genai.Client(api_key=api_key)
            elif "GOOGLE_API_KEY" in os.environ:
                client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            else:
                raise RuntimeError("API key not provided")

            contents = [prompt]

            # Референсные изображения
            ref_images = [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5]
            for img in ref_images:
                if img is not None:
                    contents.extend(images_to_pillow(img))

            # Основное редактируемое изображение
            if image_to_edit is not None:
                contents.extend(images_to_pillow(image_to_edit))

            # Настройки генерации
            gen_config = {}
            if temperature is not None and temperature >= 0:
                gen_config["temperature"] = temperature
            if system_instruction:
                gen_config["system_instruction"] = system_instruction

            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                retry_config = retry.Retry(
                    predicate=retry.if_exception_type(
                        exceptions.InternalServerError,
                        exceptions.ResourceExhausted,
                        exceptions.ServiceUnavailable,
                    )
                )
                response = retry_config(client.models.generate_content)(
                    model=model,
                    contents=contents,
                    config=gen_config,
                )

            # Обработка ответа
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
                    if self.status_output.startswith("Error:"):
                        pass
                    else:
                        self.status_output = "Error: No image data in API response."

        except Exception as e:
            self.status_output = f"Error: {e}"
            logging.error(f"An exception occurred in NanoBananaNode: {e}", exc_info=True)

        # Фолбэк: чёрное изображение
        if not output_images:
            width, height = 512, 512
            if image_to_edit is not None:
                height, width = image_to_edit.shape[1], image_to_edit.shape[2]
            black_image = Image.new("RGB", (width, height), (0, 0, 0))
            output_images.append(black_image)

        self.image_output = pillow_to_tensor(output_images)
        return []


NODE_CLASS_MAPPINGS = {
    "Nano_Banana": NanoBananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nano_Banana": "Nano Banana",
}
