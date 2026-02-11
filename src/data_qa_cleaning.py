#!/usr/bin/env python3
"""
Script de Data QA & Cleaning pour dataset YOLO Pokémon (émulateur PyBoy)
- Visualisation de contrôle (bounding boxes)
- Déduplication d'images
- Statistiques de classes
- Split train/val
"""
import os
import cv2
import random
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

# Chemins
IMG_DIR = Path('data/dataset/raw/images')
LBL_DIR = Path('data/dataset/raw/labels')
DEBUG_DIR = Path('debug_output')
TRAIN_DIR = Path('data/dataset/train')
VAL_DIR = Path('data/dataset/val')
IMG_SIZE = (160, 144)

random.seed(42)

def yolo_to_bbox(yolo_line, img_w, img_h):
    parts = yolo_line.strip().split()
    class_id, x_c, y_c, w, h = map(float, parts)
    x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h
    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)
    return int(class_id), x1, y1, x2, y2

def draw_boxes(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Image non trouvée: {img_path}")
        return
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x1, y1, x2, y2 = yolo_to_bbox(line, *IMG_SIZE)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(class_id), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out_path), img)

def visualisation_controle():
    print("[INFO] Visualisation de contrôle...")
    DEBUG_DIR.mkdir(exist_ok=True)
    images = sorted(list(IMG_DIR.glob('*.jpg')))
    if len(images) < 20:
        print("[WARN] Moins de 20 images disponibles.")
    sample_imgs = random.sample(images, min(20, len(images)))
    for img_path in sample_imgs:
        label_path = LBL_DIR / (img_path.stem + '.txt')
        out_path = DEBUG_DIR / img_path.name
        if label_path.exists():
            draw_boxes(img_path, label_path, out_path)
        else:
            print(f"[WARN] Label manquant pour {img_path.name}")
    print(f"[INFO] Images annotées sauvegardées dans {DEBUG_DIR}/")

def hash_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return hashlib.md5(img.tobytes()).hexdigest()

def deduplication():
    print("[INFO] Déduplication d'images...")
    hashes = defaultdict(list)
    for img_path in IMG_DIR.glob('*.jpg'):
        h = hash_image(img_path)
        if h:
            hashes[h].append(img_path)
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicates:
        print("[INFO] Doublons trouvés :")
        for group in duplicates:
            print("  - ", ', '.join(str(p.name) for p in group))
    else:
        print("[INFO] Aucun doublon détecté.")
    return duplicates

def statistiques_classes():
    print("[INFO] Statistiques de classes...")
    class_counts = defaultdict(int)
    for lbl_path in LBL_DIR.glob('*.txt'):
        with open(lbl_path, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
    print("[INFO] Résumé des classes :")
    for cid, count in sorted(class_counts.items()):
        print(f"  Classe {cid}: {count}")
    return class_counts

def split_train_val():
    print("[INFO] Split train/val...")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(list(IMG_DIR.glob('*.jpg')))
    random.shuffle(images)
    n_val = int(0.2 * len(images))
    val_imgs = set(images[:n_val])
    for img_path in images:
        label_path = LBL_DIR / (img_path.stem + '.txt')
        if img_path in val_imgs:
            dest_img = VAL_DIR / img_path.name
            dest_lbl = VAL_DIR / (img_path.stem + '.txt')
        else:
            dest_img = TRAIN_DIR / img_path.name
            dest_lbl = TRAIN_DIR / (img_path.stem + '.txt')
        shutil.copy2(img_path, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_lbl)
        else:
            print(f"[WARN] Label manquant pour {img_path.name}")
    print(f"[INFO] Split terminé : {len(images)-n_val} train, {n_val} val.")

def main():
    visualisation_controle()
    deduplication()
    statistiques_classes()
    split_train_val()
    print("[INFO] Pipeline Data QA & Cleaning terminée.")

if __name__ == '__main__':
    main()
