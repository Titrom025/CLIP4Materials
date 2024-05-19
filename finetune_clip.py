import logging
import json
import os

import hydra
import torch

from datetime import datetime
from tqdm import tqdm

from clearml import Task
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from collections import defaultdict

class CustomCLIPDataset(Dataset):
    def __init__(self, root, data_file, transform=None):
        with open(os.path.join(root, data_file), 'r') as f:
            data = json.load(f)
        self.images = []
        self.texts = []
        for image_path, value in data.items():
            image_path = '/'.join(image_path.split('/')[1:])
            image_path = os.path.join(root, image_path)
            if 'mirror' in image_path or 'list' in image_path:
                print(f'Skipping: {image_path}')
                continue
            for i in range(3):
                desc_key = f"description_{i}"
                if desc_key in value:
                    self.images.append(image_path)
                    self.texts.append(value[desc_key])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]
        return image, text


def group_by_scene(data):
    scenes = defaultdict(dict)
    for image_path, value in data.items():
        scene = os.path.dirname(image_path)
        scenes[scene][image_path] = value
    return scenes

def split_scenes(scenes, seed=42, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    random.seed(seed)
    scene_names = list(scenes.keys())
    random.shuffle(scene_names)

    n_total = len(scene_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_scenes = scene_names[:n_train]
    val_scenes = scene_names[n_train:n_train + n_val]
    test_scenes = scene_names[n_train + n_val:]

    train_data = {image: value for scene in train_scenes for image, value in scenes[scene].items()}
    val_data = {image: value for scene in val_scenes for image, value in scenes[scene].items()}
    test_data = {image: value for scene in test_scenes for image, value in scenes[scene].items()}

    return train_data, val_data, test_data

def save_split_data(root, train_data, val_data, test_data):
    with open(os.path.join(root, 'train_data.json'), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(root, 'val_data.json'), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(root, 'test_data.json'), 'w') as f:
        json.dump(test_data, f)

def create_datasets(root, transform=None):
    data_file = 'processed_materials_with_llava_combined.json'
    with open(os.path.join(root, data_file), 'r') as f:
        data = json.load(f)

    if not os.path.exists(os.path.join(root, 'train_data.json')):
        scenes = group_by_scene(data)
        train_data, val_data, test_data = split_scenes(scenes)
        save_split_data(root, train_data, val_data, test_data)

    train_dataset = CustomCLIPDataset(root, 'train_data.json', transform=transform)
    val_dataset = CustomCLIPDataset(root, 'val_data.json', transform=transform)
    test_dataset = CustomCLIPDataset(root, 'test_data.json', transform=transform)

    return train_dataset, val_dataset, test_dataset

def contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    logits = torch.matmul(image_features, text_features.t())
    labels = torch.arange(len(logits), device=logits.device)

    loss = (F.cross_entropy(logits / temperature, labels) +
            F.cross_entropy(logits.t() / temperature, labels)) / 2
    return loss


def calculate_accuracy(logits):
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        correct = (predictions == torch.arange(logits.size(0), device=logits.device)).sum()
        acc = correct.float() / logits.size(0)
    return acc


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task = Task.init(project_name='CLIP Material Classification', task_name=f'Clip_{date_str}', task_type=Task.TaskTypes.training)

    script_root = hydra.utils.get_original_cwd()

    logger_ = logging.getLogger(__name__)
    logger_.info('Hydra config:')
    logger_.info(OmegaConf.to_yaml(cfg))

    clearml_config = OmegaConf.to_container(cfg, resolve=True)
    task.connect(clearml_config)
    logger_clearml = task.get_logger()

    experiments_dir = os.path.join(script_root, cfg.experiments_dir)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    experiment_path = f'{experiments_dir}/exp_train_clip_{date_str}'

    dataset_root = experiments_dir = os.path.join(script_root, 'material_dataset_135_scenes/')
    output_model_dir = os.path.join(experiment_path)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if cfg.freeze_image_encoder:
        for name, param in model.vision_model.named_parameters():
            logger_.info(f'Freeze {name}')
            param.requires_grad = False

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    transform = Compose([
        Resize((256, 256), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    train_dataset, val_dataset, test_dataset = create_datasets(dataset_root, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    for epoch in range(cfg.num_epochs):
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        tqdm_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")

        for images, texts in tqdm_bar:
            images = images.to(device)
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

            image_features = model.get_image_features(images)
            text_features = model.get_text_features(**inputs)

            loss = contrastive_loss(image_features, text_features)
            logits = torch.matmul(F.normalize(image_features, p=2, dim=-1), F.normalize(text_features, p=2, dim=-1).t())
            accuracy = calculate_accuracy(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy
            total_batches += 1
            
            tqdm_bar.set_postfix(loss=f"{total_loss / total_batches:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        logger_.info(f"Epoch {epoch + 1} - Average Loss: {average_loss:.3f}, Average Accuracy: {average_accuracy:.3f}")
    
        logger_clearml.report_scalar("Loss", "train", iteration=epoch, value=average_loss)
        logger_clearml.report_scalar("Accuracy", "train", iteration=epoch, value=average_accuracy)

        val_total_loss = 0
        val_total_accuracy = 0
        val_total_batches = 0
        model.eval()
        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validation"):
                images = images.to(device)
                inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)

                image_features = model.get_image_features(images)
                text_features = model.get_text_features(**inputs)

                loss = contrastive_loss(image_features, text_features)
                logits = torch.matmul(F.normalize(image_features, p=2, dim=-1), F.normalize(text_features, p=2, dim=-1).t())
                accuracy = calculate_accuracy(logits)

                val_total_loss += loss.item()
                val_total_accuracy += accuracy
                val_total_batches += 1

        val_average_loss = val_total_loss / val_total_batches
        val_average_accuracy = val_total_accuracy / val_total_batches
        logger_.info(f"Validation - Average Loss: {val_average_loss:.3f}, Average Accuracy: {val_average_accuracy:.3f}")

        logger_clearml.report_scalar("Loss", "val", iteration=epoch, value=val_average_loss)
        logger_clearml.report_scalar("Accuracy", "val", iteration=epoch, value=val_average_accuracy)
        
        model.train()

    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)

if __name__ == "__main__":
    main()
