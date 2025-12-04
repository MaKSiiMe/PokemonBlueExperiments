import os
import shutil
import random
import glob

# CONFIGURATION FIXE
SOURCE_DIR = os.path.join("data", "dataset", "raw")
# Le dossier "dataset" à la racine est requis par YOLO par défaut, on le laisse là
DEST_DIR = os.path.join("data", "dataset", "pokemon") 

def main():
    print(f"📦 Organisation depuis {SOURCE_DIR}...")
    
    # Nettoyage si on relance
    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, split, 'labels'), exist_ok=True)
        
    images = glob.glob(os.path.join(SOURCE_DIR, "images", "*.jpg"))
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    splits = {'train': images[:split_idx], 'val': images[split_idx:]}
    
    for split_name, imgs in splits.items():
        for img_path in imgs:
            img_name = os.path.basename(img_path)
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = os.path.join(SOURCE_DIR, "labels", txt_name)
            
            if os.path.exists(txt_path):
                shutil.copy(img_path, os.path.join(DEST_DIR, split_name, 'images', img_name))
                shutil.copy(txt_path, os.path.join(DEST_DIR, split_name, 'labels', txt_name))

    print("✅ Données prêtes pour YOLO.")

if __name__ == "__main__":
    main()
