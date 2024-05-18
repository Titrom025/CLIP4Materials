import logging
import json
import os

import hydra
import torch

from datetime import datetime
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR


class CustomCLIPDataset(Dataset):
    def __init__(self, root, data_file, transform=None):
        with open(os.path.join(root, data_file), 'r') as f:
            data = json.load(f)
        self.images = []
        self.texts = []
        for image_path, value in data.items():
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
    script_root = hydra.utils.get_original_cwd()
    logger = logging.getLogger(__name__)

    logger.info('Hydra config:')
    logger.info(OmegaConf.to_yaml(cfg))

    if not os.path.exists(cfg.experiments_dir):
        os.makedirs(cfg.experiments_dir)
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_path = f'{cfg.experiments_dir}/exp_train_clip_{date_str}'

    data_file = 'material_dataset_v2/processed_materials_with_llava.json'
    output_model_dir = os.path.join(experiment_path)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if cfg.freeze_image_encoder:
        for name, param in model.vision_model.named_parameters():
            logger.info(f'Freeze {name}')
            param.requires_grad = False

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    transform = Compose([
        Resize((256, 256), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = CustomCLIPDataset(script_root, data_file, transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    for epoch in range(cfg.num_epochs):
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        tqdm_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")

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
        logger.info(f"Epoch {epoch + 1} - Average Loss: {average_loss:.3f}, Average Accuracy: {average_accuracy:.3f}")
    
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)

if __name__ == "__main__":
    main()
