import argparse
import cv2
import json
import os
import random
import time

from tqdm import tqdm

from model_apis import describe_images, init_llava

DESCRIPTION_PROMPT = "Describe the object in details"

def describe_with_llava(objects, processed_markup_path):
    init_llava()
   
    runtimes = []
    for object_id, image_path in enumerate(tqdm(objects, desc="Processing objects with LLaVa")):
        object_names = ",".join(objects[image_path]["name"])
        object_materials = ",".join(objects[image_path]["material"])
        prompts = [
            f'Describe the ({object_names}) made of ({object_materials}) ' \
            f'in the image including color and material in one sentence.' \
            f'There is an example of the answer format: ' \
            f'"The trash can is black and made of plastic"',
        ]
        for prompt_idx, prompt in enumerate(prompts):
            ts = time.time()
            generated_descriptions = describe_images([image_path], prompt)
            te = time.time()
            runtimes.append(te-ts)
            objects[image_path][f"description_{prompt_idx}"] = generated_descriptions[0]

        if object_id % 10 == 0:
            with open(processed_markup_path, 'w') as json_file:
                json.dump(objects, json_file, indent=4)
            print(f'Markup file saved with objects {object_id+1}/{len(objects)}')    
    
    with open(processed_markup_path, 'w') as json_file:
        json.dump(objects, json_file, indent=4)
    print(f'Processed markup file saved to {processed_markup_path}')
    print(f'Mean llava_1.6 runtime: {round(sum(runtimes) / len(runtimes), 2)}sec')


def main():
    dataset_path = "material_dataset_135_scenes_v2"
    coco_markup_path = os.path.join(dataset_path, "processed_materials.json")

    with open(coco_markup_path, 'r') as json_file:
        objects = json.load(json_file)

    processed_markup_path = os.path.join(dataset_path, "processed_materials_with_llava.json")
    describe_with_llava(objects, processed_markup_path)


if __name__ == "__main__":
    main()