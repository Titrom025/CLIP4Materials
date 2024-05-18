import argparse
import json
import os
import shutil

import torch

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def draw_material(image_path, result, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    text = f"{result['material']} ({result['confidence']:.2f})"
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    image.save(output_path)


def evaluate_clip_model(model_path, dataset_root, material_markup, classes, debug=False):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    if not os.path.isdir(model_path):
        # model from huggingface
        model_path = os.path.join("experiments", model_path)
        os.makedirs(model_path, exist_ok=True)

    draw_folder = "drawed_materials"
    draw_output_folder = os.path.join(model_path, draw_folder)
    if os.path.exists(draw_output_folder):
        shutil.rmtree(draw_output_folder)
    os.makedirs(draw_output_folder)

    total_count = 0
    correct_count = 0

    with open(material_markup, 'r') as file:
        material_references = json.load(file)
    
    output_markup = os.path.join(model_path, 'evaluate_stats.json')

    for image_path in tqdm(material_references):
        image_full_path = os.path.join(dataset_root, image_path)

        image = Image.open(image_full_path).convert("RGB")
        inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs.to(device))
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        max_idx = probs.argmax().item()
        confidence = probs.max().item()
        best_class = classes[max_idx]

        result = {"material": best_class, "confidence": confidence}
        if debug:
            result["all_materials"] = {cls: float(prob) for cls, prob in zip(classes, probs[0])}
        
        material_references[image_path]['predicted_material'] = result['material']
        material_references[image_path]['predicted_confidence'] = result['confidence']
        if debug:
            material_references[image_path]['all_materials'] = result['all_materials']

        draw_material(image_full_path, result, os.path.join(draw_output_folder, image_path))

        total_count += 1
        is_correct = any(mat in material_references[image_path]['material'] for mat in [result['material']])
        if is_correct:
            correct_count += 1
        
        material_references[image_path]['is_correct'] = is_correct
    
    accuracy = round(correct_count / total_count if total_count > 0 else 0, 2)
    print("Accuracy:", accuracy)

    material_references['info'] = {
        'model': model_path,
        'accuracy': accuracy
    }

    with open(output_markup, 'w') as f:
        json.dump(material_references, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP model for material classification.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained CLIP model or checkpoint.")
    parser.add_argument("--dataset_root", type=str, default="../ml-hypersim/project/", help="Root directory of the dataset.")
    parser.add_argument("--material_markup", type=str, default="../ml-hypersim/project/material_dataset/processed_materials.json", help="Path to the material markup file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output.")

    args = parser.parse_args()

    material_classes = ["textile", "glass", "paper", "wood", "plastic", "ceramics", "metal"]

    evaluate_clip_model(args.model_path, args.dataset_root, args.material_markup, material_classes, args.debug)

if __name__ == "__main__":
    main()
