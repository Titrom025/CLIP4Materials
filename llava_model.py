import os
import re
import sys
import torch
from transformers import logging as hf_logging
from PIL import Image
import requests
from io import BytesIO
try:
    LLAVA_PYTHON_PATH = os.environ["LLAVA_PYTHON_PATH"]
except KeyError:
    LLAVA_PYTHON_PATH = "../LLaVA"
    print(f'Using default LLAVA_PYTHON_PATH: {LLAVA_PYTHON_PATH}')
    
sys.path.append(LLAVA_PYTHON_PATH)
torch.autograd.set_grad_enabled(False)

# Set logging verbosity for the transformers package to only log errors
hf_logging.set_verbosity_error()

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

class LLaVaChat(object):
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b", conv_mode="llava_v1"):
        disable_torch_init()

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name
        )
        self.conv_mode = conv_mode

        self.reset()
        
    def reset(self):        
        # Cache for image features
        pass

    def __call__(self, query, image_features, image_sizes):
        # Given this query, and the image_featurese, prompt LLaVA with the query,
        # using the image_features as context.
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        elif "llava-v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        else:
            conv_mode = "llava_v0"        

        if self.conv_mode is not None and conv_mode != self.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = conv_mode

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda())
    
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 512
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_features,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                image_sizes = image_sizes, 
            )

        outputs_new = self.tokenizer.batch_decode(
            output_ids[:, 0:], skip_special_tokens=True
        )[0]
        outputs_new = outputs_new.strip()
        if outputs_new.endswith(stop_str):
            outputs_new = outputs_new[: -len(stop_str)]
        outputs_new = outputs_new.strip()

        return outputs_new
    
    def preprocess_image(self, images):
        return process_images(
            images,
            self.image_processor,
            self.model.config).to(self.model.device, dtype=torch.float16)


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

if __name__ == "__main__":

    model_path = "liuhaotian/llava-v1.6-vicuna-7b"
    chat = LLaVaChat(model_path)
    print("LLaVA chat initialized...")

    query = "List the set of objects in this image."
    image_features = load_images(["https://llava-vl.github.io/static/images/view.jpg"])
    image_sizes = [image.size for image in image_features]
    image_features = chat.preprocess_image(image_features)

    outputs = chat(query=query, image_features=image_features, image_sizes=image_sizes)
    print(outputs)
