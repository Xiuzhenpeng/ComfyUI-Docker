import os
import torch
import random
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION, LoraLoader
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Base64ToImageNode
class Base64ToImageNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base64_str": ("STRING", {
                    "multiline": True,
                    "default": "请输入base64编码的图片"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_base64"
    CATEGORY = "IAT/Input"

    def convert_base64(self, base64_str):
        try:
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            return (img,)
        except Exception as e:
            print(f"Base64转换失败: {str(e)}")
            blank_img = torch.zeros((1, 512, 512, 3))
            return (blank_img,)

# FloatInputNode
class FloatInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "IAT/Input"

    def get_float(self, value):
        return (value,)

# ImageMatchSize
class ImageMatchSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "match_size"
    CATEGORY = "IAT"

    def match_size(self, reference_image, input_image):
        ref_img = 255. * reference_image[0].cpu().numpy()
        ref_img = Image.fromarray(np.clip(ref_img, 0, 255).astype(np.uint8))
        ref_width, ref_height = ref_img.size

        input_img = 255. * input_image[0].cpu().numpy()
        input_img = Image.fromarray(np.clip(input_img, 0, 255).astype(np.uint8))
        resized_img = input_img.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
        resized_img = np.array(resized_img).astype(np.float32) / 255.0
        resized_img = torch.from_numpy(resized_img)[None,]
        return (resized_img,)

# ImageResizeLongestSideNode
class ImageResizeLongestSideNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_side": ("INT", {"default": 1536, "min": 64, "max": MAX_RESOLUTION}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_longest_side"
    CATEGORY = "IAT"

    def resize_longest_side(self, image, longest_side):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        width, height = img.size
        if width > height:
            new_width = longest_side
            new_height = int(height * (longest_side / width))
        else:
            new_height = longest_side
            new_width = int(width * (longest_side / height))
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img = np.array(resized_img).astype(np.float32) / 255.0
        resized_img = torch.from_numpy(resized_img)[None,]
        return (resized_img,)

# ImageResizeToSDXL
class ImageResizeToSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("resized_16x", "resized_original_ratio")
    FUNCTION = "resize_image"
    CATEGORY = "IAT"

    def resize_image(self, image):
        max_pixels = 1024 * 1024
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        width, height = img.size
        current_pixels = width * height
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
            new_width_16x = int(width * scale)
            new_height_16x = int(height * scale)
            new_width_16x = (new_width_16x // 16) * 16
            new_height_16x = (new_height_16x // 16) * 16
            img_16x = img.resize((new_width_16x, new_height_16x), Image.Resampling.LANCZOS)
            new_width_ratio = int(width * scale)
            new_height_ratio = int(height * scale)
            img_ratio = img.resize((new_width_ratio, new_height_ratio), Image.Resampling.LANCZOS)
        else:
            img_16x = img
            img_ratio = img
        img_16x = np.array(img_16x).astype(np.float32) / 255.0
        img_16x = torch.from_numpy(img_16x)[None,]
        img_ratio = np.array(img_ratio).astype(np.float32) / 255.0
        img_ratio = torch.from_numpy(img_ratio)[None,]
        return (img_16x, img_ratio)

# ImageSizeNode
class ImageSizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "IAT"

    def get_size(self, image):
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        width, height = img.size
        return (width, height)

# IntInputNode
class IntInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "IAT/Input"

    def get_int(self, value):
        return (value,)

# QwenTranslator
class QwenTranslator:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen", "Qwen2.5-3B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "请输入要翻译的文本"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "IAT"

    def detect_language(self, text):
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'ja' if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text) else 'zh'
        if re.search(r'[a-zA-Z]', text):
            return 'en'
        return None

    def translate(self, text):
        lang = self.detect_language(text)
        if not lang:
            return ("请输入中文、日文或英文",)
        if lang == 'en':
            return (text,)
        messages = [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"将以下{lang}文本翻译为英文：{text}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return (response,)

# TextInputNode
class TextInputNode:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "请输入文本"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_text"
    CATEGORY = "IAT/Input"

    def get_text(self, text):
        return (text,)

# SeedGeneratorNode
class SeedGeneratorNode:
    def __init__(self):
        self.counter = 0
        self.current_seed = 0  # 存储当前种子值

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999999999,
                    "step": 1
                }),
                "control_after_generate": ("BOOLEAN", {
                    "default": True  # 保持默认为True，符合UI中的状态
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_seed"
    CATEGORY = "IAT/Input"

    def generate_seed(self, seed, control_after_generate):
        # 先保存当前显示的种子值用于返回
        seed_to_return = seed
        
        # 如果需要随机化，则在返回当前种子后，为下一次准备一个新的随机种子
        if control_after_generate:
            # 生成新的随机种子并存储，但不立即返回
            self.current_seed = random.randint(0, 9999999999999999)

        return (seed_to_return,)

#浮点数除4节点
class FloatDivideByFourNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_float": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000000.0,
                    "max": 1000000.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("output_float",)
    FUNCTION = "divide_by_four"
    CATEGORY = "IAT/Math"

    def divide_by_four(self, input_float):
        # 将输入的浮点数除以 4
        output_float = input_float / 4.0
        return (output_float,)

# 导出所有节点
NODE_CLASS_MAPPINGS = {
    "Base64ToImageNode by IAT": Base64ToImageNode,
    "FloatInputNode by IAT": FloatInputNode,
    "ImageMatchSize by IAT": ImageMatchSize,
    "ImageResizeLongestSide by IAT": ImageResizeLongestSideNode, 
    "ImageResizeToSDXL by IAT": ImageResizeToSDXL,
    "ImageSize by IAT": ImageSizeNode, 
    "IntInputNode by IAT": IntInputNode,
    "QwenTranslator by IAT": QwenTranslator,
    "TextInputNode by IAT": TextInputNode,
    "SeedGeneratorNode by IAT": SeedGeneratorNode,
    "FloatDivideByFourNode by IAT": FloatDivideByFourNode, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Base64ToImageNode by IAT": "Base64 to Image by IAT",
    "FloatInputNode by IAT": "Float Input by IAT",
    "ImageMatchSize by IAT": "Image Match Size by IAT",
    "ImageResizeLongestSide by IAT": "Image Resize Longest Side by IAT",  
    "ImageResizeToSDXL by IAT": "ImageResizeToSDXL by IAT",
    "ImageSize by IAT": "Image Size by IAT", 
    "IntInputNode by IAT": "Integer Input by IAT",
    "QwenTranslator by IAT": "Qwen Translator by IAT",
    "TextInputNode by IAT": "Text Input by IAT",
    "SeedGeneratorNode by IAT": "Seed Generator by IAT",
    "FloatDivideByFourNode by IAT": "Float Divide by 4 by IAT"
}
