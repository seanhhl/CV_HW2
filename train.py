import os
import json
import torch
import gdown
import tarfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import albumentations as A

# Disable cuDNN benchmark due to dynamic image shapes in DETR to prevent slowdowns.
torch.backends.cudnn.benchmark = False

def install_requirements():
    print("Installing required packages...")
    subprocess.run(["pip", "install", "-q", "transformers", "timm", "pycocotools", "albumentations"], check=True)

install_requirements()

from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/Colab_files/CV_HW2'
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'checkpoint.pth')

DATASET_TAR_ID = '13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5'
DATASET_TAR_PATH = '/content/dataset.tar.gz'
DATASET_EXTRACT_PATH = '/content/dataset'

EPOCHS = 20
BATCH_SIZE = 8 
LR_BACKBONE = 1e-5
LR_TRANSFORMER = 1e-4

if not os.path.exists(DATASET_EXTRACT_PATH):
    print("Downloading dataset...")
    gdown.download(id=DATASET_TAR_ID, output=DATASET_TAR_PATH, quiet=False)
    
    print("Extracting dataset...")
    with tarfile.open(DATASET_TAR_PATH, 'r:gz') as tar_ref:
        tar_ref.extractall(DATASET_EXTRACT_PATH)
    print("Dataset preparation complete!")

TRAIN_IMG_DIR = os.path.join(DATASET_EXTRACT_PATH, 'nycu-hw2-data/train')
TRAIN_ANN_FILE = os.path.join(DATASET_EXTRACT_PATH, 'nycu-hw2-data/train.json')
VAL_IMG_DIR = os.path.join(DATASET_EXTRACT_PATH, 'nycu-hw2-data/valid')
VAL_ANN_FILE = os.path.join(DATASET_EXTRACT_PATH, 'nycu-hw2-data/valid.json')
TEST_IMG_DIR = os.path.join(DATASET_EXTRACT_PATH, 'nycu-hw2-data/test')

# Data augmentation designed for digit detection (flipping is strictly prohibited)
train_transform = A.Compose([
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.2),
    A.RandomSizedBBoxSafeCrop(width=640, height=480, erosion_rate=0.0, p=0.3)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

class DetrCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, processor, coco_to_model_id, transform=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.coco_to_model_id = coco_to_model_id
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        
        # Ensure image is RGB to avoid NumPy shape anomalies with grayscale images
        image_np = np.array(img.convert("RGB"))
        
        bboxes = [anno['bbox'] for anno in target]
        category_ids = [anno['category_id'] for anno in target]
        
        if self.transform is not None and len(bboxes) > 0:
            h, w = image_np.shape[:2]
            valid_bboxes, valid_cat_ids = [], []
            for bbox, cat_id in zip(bboxes, category_ids):
                x_min, y_min, bw, bh = bbox
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_min + bw), min(h, y_min + bh)
                if x_max > x_min and y_max > y_min:
                    valid_bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
                    valid_cat_ids.append(cat_id)
            
            if len(valid_bboxes) > 0:
                transformed = self.transform(image=image_np, bboxes=valid_bboxes, category_ids=valid_cat_ids)
                image_np = transformed['image']
                bboxes = transformed['bboxes']
                category_ids = transformed['category_ids']
            
        mapped_target = []
        for bbox, cat_id in zip(bboxes, category_ids):
            if bbox[2] > 0 and bbox[3] > 0:
                mapped_id = self.coco_to_model_id.get(cat_id)
                if mapped_id is not None:
                    mapped_target.append({
                        'bbox': list(bbox), 
                        'category_id': mapped_id,
                        'area': bbox[2] * bbox[3],
                        'iscrowd': 0
                    })
            
        annotations = {'image_id': image_id, 'annotations': mapped_target}
        
        encoding = self.processor(images=image_np, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        # Prevent errors if crop results in no objects
        if "labels" in encoding and len(encoding["labels"]) > 0:
            target_out = encoding["labels"][0]
        else:
            target_out = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "class_labels": torch.empty((0,), dtype=torch.long)
            }
        
        return pixel_values, target_out

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    batch_dict = processor.pad(pixel_values, return_tensors="pt")
    
    return {
        'pixel_values': batch_dict['pixel_values'],
        'pixel_mask': batch_dict['pixel_mask'],
        'labels': labels
    }

print("Initializing model and processor...")
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    size={"shortest_edge": 480, "longest_edge": 640}
)

with open(TRAIN_ANN_FILE, 'r') as f:
    coco_data = json.load(f)
categories = coco_data['categories']

# Map COCO IDs (usually 1-indexed) to Model IDs (must be 0-indexed and continuous)
coco_id_to_model_id = {cat['id']: i for i, cat in enumerate(categories)}
model_id_to_coco_id = {i: cat['id'] for i, cat in enumerate(categories)}

print(f"\n[Check] Category ID Mapping (Internal model ID -> Output required COCO ID): {model_id_to_coco_id}\n")

id2label = {i: cat['name'] for i, cat in enumerate(categories)}
label2id = {cat['name']: i for i, cat in enumerate(categories)}
num_classes = len(categories)

train_dataset = DetrCocoDataset(img_folder=TRAIN_IMG_DIR, ann_file=TRAIN_ANN_FILE, processor=processor, coco_to_model_id=coco_id_to_model_id, transform=train_transform)
val_dataset = DetrCocoDataset(img_folder=VAL_IMG_DIR, ann_file=VAL_ANN_FILE, processor=processor, coco_to_model_id=coco_id_to_model_id, transform=None)

workers = min(4, os.cpu_count() or 2)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)

print("Initializing model from config (Transformer trained from scratch, backbone using pretrained weights)...")
config = DetrConfig.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
    use_pretrained_backbone=True 
)

model = DetrForObjectDetection(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current compute device: {device}")
model.to(device)

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": LR_BACKBONE,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=LR_TRANSFORMER, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n[Resume] Found previous checkpoint: {CHECKPOINT_PATH}")
    print("Restoring model, optimizer, and scheduler states...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Successfully restored! Resuming from Epoch {start_epoch+1} to Epoch {EPOCHS}.\n")
else:
    print("\n[New Training] No checkpoint found. Starting from Epoch 1.\n")

print(f"Starting training, expected to run for {EPOCHS} Epochs...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["labels"]]
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
    lr_scheduler.step()
    
    avg_loss = train_loss / len(train_dataloader)
    print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Epoch {epoch+1} state securely backed up to Drive!")

model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"Model weights saved to: {SAVE_DIR}")

print("Starting predictions on test set...")
model.eval()
predictions = []

test_images = [img for img in os.listdir(TEST_IMG_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

test_images.sort()

for img_name in tqdm(test_images, desc="Predicting"):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    
    try:
        image_id = int(''.join(filter(str.isdigit, img_name)))
    except ValueError:
        import sys
        image_id = hash(img_name) % ((sys.maxsize + 1) * 2) 

    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(**inputs)
        
    target_sizes = torch.tensor([image.size[::-1]])
    
    # Lower threshold to 0.1 for better COCO mAP computation on PR-Curve
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = box.tolist()
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        
        coco_category_id = model_id_to_coco_id[label.item()]
        
        pred_dict = {
            "image_id": image_id,
            "bbox": [x_min, y_min, w, h],
            "score": score.item(),
            "category_id": coco_category_id
        }
        predictions.append(pred_dict)

pred_json_path = 'pred.json'
drive_pred_path = os.path.join(SAVE_DIR, 'pred.json')

with open(pred_json_path, 'w') as f:
    json.dump(predictions, f, indent=4)
    
import shutil
shutil.copy(pred_json_path, drive_pred_path)

print(f"Prediction complete! Result file saved as {pred_json_path}")
print(f"Simultaneously backed up to your Google Drive: {drive_pred_path}")
