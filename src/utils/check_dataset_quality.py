import cv2
import os
import glob
import random

IMG_DIR = "data/dataset/pokemon/train/images"
LABEL_DIR = "data/dataset/pokemon/train/labels"

COLORS = {
    0: (255, 0, 0),   # Player (Bleu)
    1: (0, 0, 255),   # NPC (Rouge)
    2: (0, 255, 255), # Door (Jaune)
    3: (255, 0, 255)  # Sign (Violet)
}

def main():
    print("🕵️  Inspection de la Qualité du Dataset...")
    
    img_files = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not img_files:
        print("❌ Erreur : Dossier vide. As-tu fait le split ?")
        return

    samples = random.sample(img_files, 50)

    for img_path in samples:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        basename = os.path.basename(img_path)
        txt_name = basename.replace(".jpg", ".txt")
        txt_path = os.path.join(LABEL_DIR, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    color = COLORS.get(cls, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Dataset Audit (Appuie sur une touche)", cv2.resize(img, (640, 576), interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(0)
        if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
