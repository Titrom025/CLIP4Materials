import argparse
import json
import os
import shutil

import torch

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def draw_material(image_path, result, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font = ImageFont.load_default(size=20)

    text = f"{result['material']} ({result['confidence']:.2f})"
    draw.text((10, 10), text, font=font, fill=(255, 0, 0))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    image.save(output_path)


def evaluate_clip_model(model_path, dataset_root, material_markup, material_classes, debug=False):
    output_markup = os.path.join(model_path, 'evaluate_clip_v2.json')
    if os.path.exists(output_markup):
        print(f'Markup already exists: {output_markup}')
        return
    print(f'Evaluating model {model_path}')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    if not os.path.isdir(model_path):
        # model from huggingface
        model_path = os.path.join("experiments", model_path)
        output_markup = os.path.join("experiments", output_markup)
        os.makedirs(model_path, exist_ok=True)

    if debug:
        draw_folder = "drawed_materials"
        draw_output_folder = os.path.join(model_path, draw_folder)
        if os.path.exists(draw_output_folder):
            shutil.rmtree(draw_output_folder)
        os.makedirs(draw_output_folder)

    total_count = 0
    correct_count = 0

    with open(material_markup, 'r') as file:
        material_references = json.load(file)

    # total_count = defaultdict(int)
    # correct_count = defaultdict(int)
    true_positives = defaultdict(int)
    predicted_positives = defaultdict(int)
    actual_positives = defaultdict(int)

    dict_predictions = defaultdict(list)
    dict_labels = defaultdict(list)

    for material in material_classes:
        true_positives[material] = 0
        predicted_positives[material] = 0
        actual_positives[material] = 0

    import random
    random.seed(10)
    material_references_keys = list(material_references.keys())
    random.shuffle(material_references_keys)
    for idx, image_path in enumerate(tqdm(material_references_keys)):
        # if idx == 100:
        #     break
        object_name = ','.join(material_references[image_path]["name"]).strip()
        prompts = [f"The {object_name} is made of {material}" for material in material_classes]
        # prompts = [f'a photo of a {material} {object_name}' for material in material_classes]
        # prompts = [f'the {material} {object_name}' for material in material_classes]
        image_full_path = os.path.join(dataset_root, image_path)

        image = Image.open(image_full_path).convert("RGB")
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        # inputs = processor(text=material_classes, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs.to(device))
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        max_idx = probs.argmax().item()
        confidence = probs.max().item()
        confidence = probs.max().item()
        predicted_class = material_classes[max_idx]

        result = {"material": predicted_class, "confidence": confidence, "prompt": prompts[max_idx]}
        result["all_materials"] = {cls: float(prob) for cls, prob in zip(material_classes, probs[0])}
        
        material_references[image_path]['predicted_material'] = result['material']
        material_references[image_path]['predicted_confidence'] = result['confidence']
        material_references[image_path]['prompt'] = result['prompt']
        material_references[image_path]['all_materials'] = result['all_materials']

        if debug:
            draw_material(image_full_path, result, os.path.join(draw_output_folder, image_path))

        total_count += 1
        is_correct = any(mat in material_references[image_path]['material'] for mat in [result['material']])
        if is_correct:
            correct_count += 1
        
        material_references[image_path]['is_correct'] = is_correct

        true_labels = material_references[image_path]['material']
        for material in material_classes:
            dict_labels[material].append(material in true_labels)
            dict_predictions[material].append(material == predicted_class)
    
    accuracy = round(correct_count / total_count if total_count > 0 else 0, 2)
    print("Accuracy:", accuracy)

    precision = {}
    recall = {}
    f1_scores = {}
    for material in material_classes:
        precision[material], recall[material], f1_scores[material], _ = precision_recall_fscore_support(
            dict_labels[material],
            dict_predictions[material],
            average='binary')

    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    for material in material_classes:
        mean_precision += precision[material]
        mean_recall += recall[material]
        mean_f1 += f1_scores[material]
        print(f"{material}: precision {precision[material]:.2f}, recall {recall[material]:.2f}, f1 {f1_scores[material]:.2f}, support {sum(dict_labels[material])}")

    print(f'Mean precision: {round(mean_precision / len(material_classes), 2)}')
    print(f'Mean recall: {round(mean_recall / len(material_classes), 2)}')
    print(f'Mean f1: {round(mean_f1 / len(material_classes), 2)}')
        

    material_references['info'] = {
        'model': model_path,
        'accuracy': accuracy
    }

    with open(output_markup, 'w') as f:
        json.dump(material_references, f, indent=4)
    print(f'Results saved to {output_markup}')

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP model for material classification.")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained CLIP model or checkpoint.")
    parser.add_argument("--dataset-root", type=str, default="./", help="Root directory of the dataset.")
    parser.add_argument("--material-markup", type=str, default="material_dataset_135_scenes_v2/test_data.json", help="Path to the material markup file.")
    parser.add_argument("--models-dir", type=str, help="Path to the directory with models to test")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output.")

    args = parser.parse_args()

    assert args.model_path or args.models_dir, "You must specify --model-path or --models-dir"

    material_classes = ["textile", "glass", "paper", "wood", "plastic", "ceramics", "metal"]

    if args.models_dir is not None:
        for model_dir in tqdm(os.listdir(args.models_dir)):
            model_path = os.path.join(args.models_dir, model_dir)
            if not os.path.isdir(model_path) or "openai" in model_path:
                continue

            evaluate_clip_model(model_path, args.dataset_root, args.material_markup, material_classes, args.debug)
    else:
        evaluate_clip_model(args.model_path, args.dataset_root, args.material_markup, material_classes, args.debug)

if __name__ == "__main__":
    main()
