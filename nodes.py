import logging
import os
import random
from io import BytesIO

from google import genai
from google.api_core import exceptions, retry
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

# Убедитесь, что файл utils.py находится в том же каталоге
from .utils import images_to_pillow, temporary_env_var, pillow_to_tensor


class NanoBananaNode:
    # Список соотношений сторон из ar.py
    RATIO_OPTIONS = {
        "1:1 ◻": (1, 1), "5:4 ▭": (5, 4), "4:3 ▭": (4, 3), "3:2 ▭": (3, 2),
        "16:9 ▭": (16, 9), "2:1 ▭": (2, 1), "21:9 ▭": (21, 9), "32:9 ▭": (32, 9),
        "4:5 ▯": (4, 5), "3:4 ▯": (3, 4), "2:3 ▯": (2, 3), "9:16 ▯": (9, 16),
        "1:2 ▯": (1, 2), "9:21 ▯": (9, 21), "9:32 ▯": (9, 32),
    }

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
                "operation": (["None", "generate", "edit", "style_transfer", "object_insertion"], {"default": "None"}),
                "quality": (["None", "standard", "high"], {"default": "None"}),
                # ИЗМЕНЕНО: Новый список соотношений сторон
                "aspect_ratio": (["None"] + list(cls.RATIO_OPTIONS.keys()), {"default": "None"}),
                "character_consistency": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": -0.05, "min": -0.05, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "edit_image"
    CATEGORY = "Gemini"

    def __init__(self):
        self.image_output = None
        self.status_output = None

    def edit_image(self, **kwargs):
        return (self.image_output, self.status_output,)
    
    # НОВЫЙ МЕТОД: Логика расчета размеров из ar.py
    def _calculate_dimensions(self, ratio_str):
        aspect_width, aspect_height = self.RATIO_OPTIONS.get(ratio_str, (1, 1))
        
        Megapixel = 1.05  # Как вы просили
        Precision = 0.30  # Из скрипта ar.py
        
        total_pixels = int(Megapixel * 1_000_000)
        
        width = int((total_pixels * (aspect_width / aspect_height)) ** 0.5)
        height = int(width * (aspect_height / aspect_width))
        
        # "Привязка" к 64 пикселям
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        # Упрощенная корректировка точности, для надежности
        while abs((width / height) - (aspect_width / aspect_height)) > Precision:
            if (width / height) > (aspect_width / aspect_height):
                width -= 64
            else:
                height -= 64
            if width < 64 or height < 64:
                break
        
        return max(64, width), max(64, height)

    def _create_fallback_image(self, width=512, height=512, text="Generation error"):
        img = Image.new('RGB', (width, height), color=(40, 40, 40))
        try:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", size=max(12, height // 20))
            except Exception:
                font = ImageFont.load_default()
            draw.multiline_text((width/2, height/2), text, font=font, fill=(255,255,255), anchor="mm", align="center")
        except Exception:
            pass
        return pillow_to_tensor([img])

    def _build_final_prompt(self, prompt, operation, aspect_ratio, quality, character_consistency, has_edit_image, has_ref_images):
        if not prompt: prompt = ""
        
        parts = []
        base_prompt = prompt
        if operation == "generate":
            base_quality_instruction = "Generate a high-quality, photorealistic image"
            if has_ref_images:
                base_prompt = f"{base_quality_instruction}, inspired by the style and elements of the reference images: {prompt}."
            else:
                base_prompt = f"{base_quality_instruction} of: {prompt}."
        elif operation == "edit":
            if not has_edit_image: return "Error: 'edit' operation requires an 'image_to_edit' input."
            base_prompt = f"Edit the provided image according to the instruction: {prompt}. Preserve original composition and quality, only apply the requested changes."
        elif operation == "style_transfer":
            if not has_ref_images: return "Error: 'style_transfer' requires at least one reference image."
            if not has_edit_image: return "Error: 'style_transfer' requires a main 'image_to_edit'."
            base_prompt = f"Apply the style from the reference images to the main image: {prompt}. Blend stylistic elements naturally."
        elif operation == "object_insertion":
            if not has_edit_image: return "Error: 'object_insertion' requires an 'image_to_edit'."
            base_prompt = f"Insert or blend the following object/idea into the main image: {prompt}. Ensure natural lighting, shadows and perspective."
        
        parts.append(base_prompt)

        if aspect_ratio and aspect_ratio != "None" and operation != "edit":
            aspect_instructions = {
                "1:1 ◻": "Render this in a square format.", "16:9 ▭": "Render this in a widescreen landscape format.",
                "9:16 ▯": "Render this in a portrait format.", "4:3 ▭": "Render this in a standard landscape format.",
                "3:4 ▯": "Render this in a standard portrait format."
            }
            parts.append(aspect_instructions.get(aspect_ratio.split(" ")[0]))
        
        if quality and quality != "None":
            if quality == "high": parts.append("Use the highest available quality settings.")
            elif quality == "standard": parts.append("Use standard quality settings.")

        if character_consistency and (has_ref_images or has_edit_image):
            parts.append("Maintain character consistency and visual identity from the provided images.")

        return " ".join(filter(None, parts)).strip()

    def check_lazy_status(
        self, prompt: str, model: str, api_key: str | None = None, proxy: str | None = None,
        image_to_edit=None, reference_image_1=None, reference_image_2=None,
        reference_image_3=None, reference_image_4=None, reference_image_5=None,
        operation: str = "None", quality: str = "None", aspect_ratio: str = "None",
        character_consistency: bool = True, system_instruction: str | None = None,
        temperature: float | None = None, seed: int | None = None, **kwargs,
    ):
        self.image_output = None
        self.status_output = "Error: Unknown"
        output_images = []
        request_log = "Log not generated due to early error."

        has_edit_image = image_to_edit is not None
        ref_list = [reference_image_1, reference_image_2, reference_image_3, reference_image_4, reference_image_5]
        has_ref_images = any(img is not None for img in ref_list)
        
        # --- НОВАЯ ЛОГИКА АВТОМАТИЧЕСКОГО СОЗДАНИЯ ХОЛСТА ---
        if not has_edit_image and not has_ref_images and aspect_ratio != "None":
            print("INFO: No input image detected. Generating a blank canvas based on aspect ratio.")
            width, height = self._calculate_dimensions(aspect_ratio)
            
            blank_pil = Image.new('RGB', (width, height), color=(255, 255, 255))
            image_to_edit = pillow_to_tensor([blank_pil])
            
            operation = "edit"  # Принудительно ставим режим 'edit'
            has_edit_image = True # Обновляем флаг
        # ---------------------------------------------------------

        final_prompt = self._build_final_prompt(prompt, operation, aspect_ratio, quality, character_consistency, has_edit_image, has_ref_images)
        if isinstance(final_prompt, str) and final_prompt.startswith("Error:"):
            self.status_output = final_prompt
            self.image_output = self._create_fallback_image(512, 512, text=final_prompt.replace("Error: ", ""))
            return []

        contents = [final_prompt or ""]
        # Используем обновленный ref_list
        for img_tensor in ref_list:
            if img_tensor is not None:
                for pil_image in images_to_pillow(img_tensor):
                    contents.append(pil_image)
        # Используем обновленный image_to_edit
        if image_to_edit is not None:
            for pil_image in images_to_pillow(image_to_edit):
                contents.append(pil_image)

        gen_config = {}
        if temperature is not None and temperature >= 0: gen_config["temperature"] = temperature
        if seed is not None and isinstance(seed, int): gen_config["seed"] = seed
        if system_instruction: gen_config["system_instruction"] = system_instruction

        try:
            if api_key and api_key.strip(): client = genai.Client(api_key=api_key)
            elif "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"].strip(): client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            else: raise RuntimeError("API key not provided")

            request_log = (f"Model: {model}\n"
                           f"Operation: {operation if operation != 'None' else 'None'}\n"
                           f"Temperature: {gen_config.get('temperature')}\n"
                           f"Seed: {gen_config.get('seed')}\n"
                           f"Quality: {quality if quality != 'None' else 'None'}\n"
                           f"Aspect Ratio: {aspect_ratio if aspect_ratio != 'None' else 'None'}\n"
                           f"Character Consistency: {character_consistency}\n"
                           f"Num contents parts: {len(contents)}\n"
                           f"Final Prompt: {final_prompt}")

            with temporary_env_var("HTTP_PROXY", proxy), temporary_env_var("HTTPS_PROXY", proxy):
                retry_config = retry.Retry(predicate=retry.if_exception_type(
                    exceptions.InternalServerError, exceptions.ResourceExhausted, exceptions.ServiceUnavailable))
                
                response = retry_config(client.models.generate_content)(
                    model=model,
                    contents=contents,
                    config=gen_config
                )

            if not getattr(response, "candidates", None):
                self.status_output = "Error: Prompt or image was blocked by safety filters."
            else:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and getattr(candidate.content, "parts", None):
                        for part in candidate.content.parts:
                            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                                try:
                                    image = Image.open(BytesIO(part.inline_data.data))
                                    if image.mode != "RGB": image = image.convert("RGB")
                                    output_images.append(image)
                                except Exception as e:
                                    log_msg = f"Could not decode image: {e}"
                                    if not self.status_output.startswith("Error:"): self.status_output = f"Error: {log_msg}"
                if output_images: self.status_output = "Complete"
                elif not self.status_output.startswith("Error:"): self.status_output = "Error: No image data in API response."

        except Exception as e:
            self.status_output = f"Error: {e}"

        if not output_images:
            width, height = 512, 512
            if image_to_edit is not None and hasattr(image_to_edit, "shape") and len(image_to_edit.shape) == 4:
                height = int(image_to_edit.shape[1])
                width = int(image_to_edit.shape[2])
            fallback_text = (self.status_output or "Generation failed").replace("Error: ", "")
            self.image_output = self._create_fallback_image(width, height, text=fallback_text)
        else:
            self.image_output = pillow_to_tensor(output_images)

        self.status_output = f"{self.status_output}\n\n--- Request details ---\n{request_log}"
        return []

NODE_CLASS_MAPPINGS = {
    "Nano_Banana": NanoBananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Nano_Banana": "Nano Banana",
}