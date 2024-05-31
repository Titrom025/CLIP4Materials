import logging
import json
import os

import hydra
import torch

from datetime import datetime
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torch.optim import Adam, lr_scheduler
import random
from collections import defaultdict

from torch.nn import CrossEntropyLoss, Dropout
from collections import Counter

try:
    from clearml import Task
    clearml_available = True
except ImportError:
    clearml_available = False

CLASS_MAP = {'textile': 0, 'glass': 1, 'paper': 2, 'wood': 3, 'plastic': 4, 'ceramics': 5, 'metal': 6}

class CustomDataset(Dataset):
    def __init__(self, root, data_file, transform=None):
        with open(os.path.join(root, data_file), 'r') as f:
            data = json.load(f)
        self.images = []
        self.labels = []
        self.idx_to_material = []
        current_idx = 0
        for image_path, value in data.items():
            image_path = '/'.join(image_path.split('/')[1:])
            image_path = os.path.join(root, image_path)
            if 'mirror' in image_path or 'list' in image_path:
                print(f'Skipping: {image_path}')
                continue
            material = value['material'][0]  # Use only the first material
            self.images.append(image_path)
            self.labels.append(CLASS_MAP[material])
        
        
        counter = Counter(self.labels)
        print(f"Unique elements and their counts in {self.labels}:")
        for element, count in counter.items():
            print(f"{element}: {count}")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def group_by_scene(data):
    scenes = defaultdict(dict)
    for image_path, value in data.items():
        scene = os.path.dirname(image_path)
        scenes[scene][image_path] = value
    return scenes

def split_scenes(scenes, seed=42, train_ratio=0.75, val_ratio=0.15):
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

def create_datasets(root, train_transform, val_transform):
    data_file = 'processed_materials_with_llava_combined.json'
    with open(os.path.join(root, data_file), 'r') as f:
        data = json.load(f)

    if not os.path.exists(os.path.join(root, 'train_data.json')):
        scenes = group_by_scene(data)
        train_data, val_data, test_data = split_scenes(scenes)
        save_split_data(root, train_data, val_data, test_data)

    train_dataset = CustomDataset(root, 'train_data.json', transform=train_transform)
    val_dataset = CustomDataset(root, 'val_data.json', transform=val_transform)
    test_dataset = CustomDataset(root, 'test_data.json', transform=val_transform)

    return train_dataset, val_dataset, test_dataset

@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    use_clearml = cfg.use_clearml
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if use_clearml and clearml_available:
        task = Task.init(project_name='ResNet Material Classification', task_name=f'ResNet_{date_str}', task_type=Task.TaskTypes.training)
    else:
        task = None

    script_root = hydra.utils.get_original_cwd()

    logger_ = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger_.info('Hydra config:')
    logger_.info(OmegaConf.to_yaml(cfg))

    if use_clearml and clearml_available:
        clearml_config = OmegaConf.to_container(cfg, resolve=True)
        task.connect(clearml_config)
        logger_clearml = task.get_logger()
    else:
        logger_clearml = None

    experiments_dir = os.path.join(script_root, cfg.experiments_dir)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    
    experiment_path = f'{experiments_dir}/exp_train_resnet_{date_str}'

    dataset_root = os.path.join(script_root, cfg.dataset_path)
    output_model_dir = os.path.join(experiment_path)

    model = models.resnet18(pretrained=True)
    # model = models.resnet34(pretrained=True)
    # model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 7)  # Assuming there are 7 material classes

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    train_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.Pad((0, 0, 288, 288)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset, val_dataset, test_dataset = create_datasets(dataset_root, train_transform, val_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    criterion = CrossEntropyLoss()
    dropout = Dropout(p=0.3)

    for epoch in range(1, cfg.num_epochs + 1):
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        tqdm_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}")

        for images, labels in tqdm_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(dropout(images))
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == labels.data) / len(labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_batches += 1
            
            tqdm_bar.set_postfix(loss=f"{total_loss / total_batches:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        logger_.info(f"Epoch {epoch} - Average Loss: {average_loss:.3f}, Average Accuracy: {average_accuracy:.3f}")

        if logger_clearml:
            logger_clearml.report_scalar("Loss", "train", iteration=epoch, value=average_loss)
            logger_clearml.report_scalar("Accuracy", "train", iteration=epoch, value=average_accuracy)

        val_total_loss = 0
        val_total_accuracy = 0
        val_total_batches = 0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                accuracy = torch.sum(preds == labels.data) / len(labels)

                val_total_loss += loss.item()
                val_total_accuracy += accuracy.item()
                val_total_batches += 1

        val_average_loss = val_total_loss / val_total_batches
        val_average_accuracy = val_total_accuracy / val_total_batches
        logger_.info(f"Validation - Average Loss: {val_average_loss:.3f}, Average Accuracy: {val_average_accuracy:.3f}")

        if logger_clearml:
            logger_clearml.report_scalar("Loss", "val", iteration=epoch, value=val_average_loss)
            logger_clearml.report_scalar("Accuracy", "val", iteration=epoch, value=val_average_accuracy)
                                         
    model.train()
    torch.save(model.state_dict(), os.path.join(output_model_dir, "resnet50_material_classification.pth"))

if __name__ == "__main__":
    main()