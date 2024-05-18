import argparse
import base64
import cv2
import json
import os
import re
import requests
import time
import torch
import shutil

from io import BytesIO
from tqdm import tqdm

from PIL import Image
from llava_model import LLaVaChat

DEFAULT_REGION_FEA_TOKEN = "<region_fea>"
COORDS_TAG = "<target_coords>"

IMAGE_W = 1000
IMAGE_H = 1000
LOG_FILE = None


def reset_log():
    with open(LOG_FILE, 'w'):
        pass


def print_and_log(message):
    with open(LOG_FILE, 'a') as f_clip:
        f_clip.write(message + '\n')
    print(message)


def prepare_out_path(filename, obj_id=None):
    path_parts = filename.split(os.path.sep)
    new_first_folder = "frames_with_bboxes"
    path_parts[0] = new_first_folder
    if obj_id is not None:
        path_parts[-1] = path_parts[-1].replace('.jpg', f'_{obj_id}.jpg')
    return os.path.join(*path_parts)


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


def describe_images(image_paths, prompt_text, log_status=False):
    image_draw_path = prepare_out_path(image_paths[0])
    target_image_folder = os.path.dirname(image_draw_path)
    if os.path.exists(target_image_folder):
        shutil.rmtree(target_image_folder)
    os.makedirs(target_image_folder)
    
    if log_status:
        data2iterate = tqdm(image_paths)
    else:
        data2iterate = image_paths

    generated_descriptions = []
    for image_idx, image_path in enumerate(data2iterate, start=1):
        if log_status:
            print_and_log(f'Describing {image_path}')
    
        image_features = load_images([image_path])
        image_sizes = [image.size for image in image_features]
        image_features = llava_chat.preprocess_image(image_features)

        outputs = llava_chat(query=prompt_text, image_features=image_features, image_sizes=image_sizes)
        if log_status:
            print_and_log(outputs)
            print_and_log('')
        
        generated_descriptions.append(outputs)

    return generated_descriptions

def read_json_markup(json_file):
    with open(json_file, 'r') as file:
        markup_data = json.load(file)
    
    return markup_data


def init_llava():
    global LOG_FILE
    global llava_chat
    global worker_addr
    global model_name
    
    if LOG_FILE is not None:
        print("Model was already initialized")
        return
    
    LOG_FILE = 'llava_log.txt'
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"
    llava_chat = LLaVaChat(model_path)