import os
import json

import torch

from datetime import datetime
from tqdm import tqdm

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

class CustomCLIPDataset(Dataset):
    def __init__(self, dataset_root, data_file, transform=None):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.images = []
        self.texts = []
        for key, value in data.items():
            image_path = os.path.join(dataset_root, key)
            if 'mirror' in key or 'list' in key:
                print(f'Skipping: {key}')
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

def main():
    config = {
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 1e-5,
        "freeze_image_encoder": True,
        "scheduler": "CosineAnnealingLR",
        "experiments_dir": "experiments/"
    }

    if not os.path.exists(config["experiments_dir"]):
        os.makedirs(config["experiments_dir"])
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_path = f'{config["experiments_dir"]}/exp_train_clip_{date_str}'

    dataset_root = '../ml-hypersim/project'
    data_file = '..//ml-hypersim/project/material_dataset/processed_materials_with_llava.json'
    output_model_dir = os.path.join(experiment_path)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if config["freeze_image_encoder"]:
        for name, param in model.vision_model.named_parameters():
            print(f'Freeze {name}')
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=0) 
    else:
        raise ValueError(f'Unexpected sheduler: {config["scheduler"]}')

    transform = Compose([
        Resize((256, 256), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = CustomCLIPDataset(dataset_root, data_file, transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    with open(os.path.join(output_model_dir, 'config_params.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

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
            
            # Update tqdm bar to include the current learning rate
            tqdm_bar.set_postfix(loss=f"{total_loss / total_batches:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()  # Step the scheduler after each epoch

        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        print(f"Epoch {epoch + 1} - Average Loss: {average_loss:.3f}, Average Accuracy: {average_accuracy:.3f}")
    
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)

if __name__ == "__main__":
    main()
