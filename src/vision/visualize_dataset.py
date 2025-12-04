import cv2
import os
import glob
import random


# CONFIGURATION
BASE_DIR = "data/dataset/raw"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR = os.path.join("data", "debug_vis")

# Couleurs et labels par classe (BGR pour OpenCV)
CLASS_COLORS = {
    0: (255, 0, 0),    # Player - Bleu
    1: (0, 0, 255),    # NPC - Rouge
    2: (0, 255, 255),  # Door - Jaune
    3: (0, 255, 0),    # Sign - Vert
}

CLASS_LABELS = {
    0: "Player",
    1: "NPC",
    2: "Door",
    3: "Sign",
}

# Création du dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_yolo_box(img, line):
    """Convertit YOLO (x_center, y_center, w, h) en Pixels et dessine"""
    height, width, _ = img.shape
    
    parts = line.split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    
    # Récupérer couleur et label selon la classe
    color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Blanc par défaut
    label = CLASS_LABELS.get(class_id, f"Class {class_id}")
    
    # Conversion YOLO -> Pixels (Top-Left, Bottom-Right)
    x_min = int((x_center - w/2) * width)
    y_min = int((y_center - h/2) * height)
    x_max = int((x_center + w/2) * width)
    y_max = int((y_center + h/2) * height)
    
    # Dessiner le rectangle
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
    
    # Ajouter le texte
    cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def main():
    print("🕵️  Vérification du dataset...")
    
    # On prend toutes les images jpg
    image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    
    if not image_paths:
        print("❌ Aucune image trouvée !")
        return

    # On en prend 10 au hasard pour tester
    samples = random.sample(image_paths, min(len(image_paths), 20))
    
    for img_path in samples:
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Trouver le fichier label correspondant
        basename = os.path.basename(img_path).replace(".jpg", ".txt")
        label_path = os.path.join(LABEL_DIR, basename)
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    draw_yolo_box(img, line)
        
        # Sauvegarder l'image debug
        output_path = os.path.join(OUTPUT_DIR, "debug_" + basename.replace(".txt", ".jpg"))
        cv2.imwrite(output_path, img)
        print(f"✅ Image générée : {output_path}")

    print(f"\n📂 Va voir dans le dossier '{OUTPUT_DIR}' si les carrés verts sont bien placés !")

if __name__ == "__main__":
    main()
