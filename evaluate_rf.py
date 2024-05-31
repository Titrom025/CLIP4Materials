import json
import os
import shutil
import random
from tqdm import tqdm
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import joblib
import numpy as np

def calculate_edge_quality(image):
    # image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    # Применяем фильтр Собеля для обнаружения границ
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисляем градиентную амплитуду и угол
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_angle = np.arctan2(sobel_y, sobel_x)

    # Оцениваем четкость границ (высокая амплитуда градиента = более четкие границы)
    edge_sharpness = np.mean(gradient_magnitude)

    # Вычисляем текстурность (большой разброс углов = более грубая текстура)
    angle_variance = np.var(gradient_angle)
    texture_roughness = 1 / (1 + angle_variance)

    return edge_sharpness, texture_roughness


def apply_custom_gabor_transform(image):
    # image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    # Создание собственного ядра Габора с указанным углом
    means = []
    for angle in range(0, 90, 5):
        kernel_size = 11
        sigma = 5
        theta = angle
        lambda_ = 10
        gamma = 0.5
        psi = 0
        gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma, psi)

        # Применение фильтра Габора к изображению
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
        means.append(np.std(filtered_image))

    filtered_image = np.mean(means)
    return filtered_image


def get_statistics(image, eps=1e-10):
    # Преобразование изображения в одномерный массив интенсивностей пикселей
    intensities = image.ravel()

    # Вычисление эксцесса распределения интенсивности пикселей и станд. откл.
    mean_intensity = np.mean(intensities)
    std_dev = np.std(intensities)
    kurtosis = (np.mean((intensities - mean_intensity)**4) / (std_dev**4 + eps)) - 3

    hist = np.zeros(256)
    idxs, values = np.unique(image, return_counts=True)
    hist[idxs] = values
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + eps))
    # print(f"Эксцесс: {kurtosis}\nСтандартное отклонение: {std_dev}\nЭнтропия: {entropy}")
    return kurtosis, std_dev, entropy


def evaluate_random_forest_model(model_path, dataset_root, material_markup, material_classes, debug=False):
    output_markup = os.path.join(dataset_root, 'evaluate_rf_v2.json')
    if os.path.exists(output_markup):
        print(f'Markup already exists: {output_markup}')
        return
    print(f'Evaluating Random Forest model')

    label_map = {0: 'ceramics', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'textile', 6: 'wood'}
    
    rf_model = joblib.load(model_path)
    print(rf_model)
    print(f"Model loaded from {model_path}")

    if not os.path.isdir(dataset_root):
        os.makedirs(dataset_root, exist_ok=True)

    if debug:
        draw_folder = "drawed_materials_rf"
        draw_output_folder = os.path.join(dataset_root, draw_folder)
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

    small_test_features = []

    material_references_keys = list(material_references.keys())
    random.shuffle(material_references_keys)
    for idx, image_path in enumerate(tqdm(material_references_keys)):
        # if idx > 10:
        #     break
        image_full_path = os.path.join(dataset_root, image_path)
        image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
        target_size = (256, 256)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        edge_sharpness, texture_roughness = calculate_edge_quality(image)
        fragility = apply_custom_gabor_transform(image)
        kurtosis, std, entropy = get_statistics(image)
        small_test_features.append([image_path, edge_sharpness, texture_roughness, fragility, kurtosis, std, entropy])

    small_test_features_df = pd.DataFrame(small_test_features, columns=['image_name', 'edge_sharpness', 'texture_roughness', 'fragility', 'kurtosis', 'std', 'entropy'])

    predicts = rf_model.predict(small_test_features_df.iloc[:, 1:].values)
    total_count = len(predicts)

    for i in range(total_count):
        predicted_class = label_map[round(predicts[i])]
        true_labels = material_references[small_test_features_df.iloc[i, 0]]['material']
        material_references[small_test_features_df.iloc[i, 0]]['predicted_material'] = predicted_class

        is_correct = any(mat in true_labels for mat in [predicted_class])
        material_references[small_test_features_df.iloc[i, 0]]['is_correct'] = is_correct

        if is_correct:
            correct_count += 1

        for material in material_classes:
            dict_labels[material].append(material in true_labels)
            dict_predictions[material].append(material == predicted_class)

    accuracy = round(correct_count / total_count if total_count > 0 else 0, 3)
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

    print(f'Mean precision: {round(mean_precision / len(material_classes), 3)}')
    print(f'Mean recall: {round(mean_recall / len(material_classes), 3)}')
    print(f'Mean f1: {round(mean_f1 / len(material_classes), 3)}')

    material_references['info'] = {
        'model': 'Random Forest',
        'accuracy': accuracy
    }

    with open(output_markup, 'w') as f:
        json.dump(material_references, f, indent=4)
    print(f'Results saved to {output_markup}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Random Forest model for material classification.")
    parser.add_argument("--model-path", type=str, help="Path to the pre-trained Random Forest model or checkpoint.")
    parser.add_argument("--dataset-root", type=str, default="./", help="Root directory of the dataset.")
    parser.add_argument("--material-markup", type=str, default="material_dataset_135_scenes_v2/test_data.json", help="Path to the material markup file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for more verbose output.")

    args = parser.parse_args()

    material_classes = ["textile", "glass", "paper", "wood", "plastic", "ceramics", "metal"]


    evaluate_random_forest_model(args.model_path, args.dataset_root, args.material_markup, material_classes, args.debug)

if __name__ == "__main__":
    main()