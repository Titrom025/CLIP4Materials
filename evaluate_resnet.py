import argparse
import json
import os
import shutil

import torch

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from torchvision import models, transforms

from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def draw_material(image_path, result, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font = ImageFont.load_default()

    text = f"{result['material']} ({result['confidence']:.2f})"
    draw.text((10, 10), text, font=font, fill=(255, 0, 0))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    image.save(output_path)


def evaluate_resnet_model(model_path, dataset_root, material_markup, material_classes, debug=False):
    output_markup = os.path.join(os.path.dirname(model_path), 'evaluate_resnet_v2.json')
    if os.path.exists(output_markup):
        print(f'Markup already exists: {output_markup}')
        return
    print(f'Evaluating model {model_path}')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = models.resnet18(pretrained=False)
    # model = models.resnet34(pretrained=False)
    # model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(material_classes))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        object_name = ','.join(material_references[image_path]["name"]).strip()
        image_full_path = os.path.join(dataset_root, image_path)

        image = Image.open(image_full_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_idx = probs.argmax().item()
            confidence = probs.max().item()
            predicted_class = material_classes[max_idx]

        result = {"material": predicted_class, "confidence": confidence}
        result["all_materials"] = {cls: float(prob) for cls, prob in zip(material_classes, probs[0])}
        
        material_references[image_path]['predicted_material'] = result['material']
        material_references[image_path]['predicted_confidence'] = result['confidence']
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
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 model for material classification.")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained ResNet-50 model or checkpoint.")
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

            evaluate_resnet_model(model_path, args.dataset_root, args.material_markup, material_classes, args.debug)
    else:
        evaluate_resnet_model(args.model_path, args.dataset_root, args.material_markup, material_classes, args.debug)

if __name__ == "__main__":
    main()