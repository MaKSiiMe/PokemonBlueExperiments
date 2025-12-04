from ultralytics import YOLO
import torch
import os
import yaml

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🥊 Démarrage de l'entraînement sur : {device.upper()}")

    os.environ.setdefault("ULTRALYTICS_WORKERS", "0")

    base = "/home/maxime/Holberton/C#25-ML_Portfolio/PokemonBlueExperiments/data/dataset/pokemon"
    val_images = os.path.join(base, "val", "images")
    if not os.path.isdir(val_images):
        print(f"⚠️ Aucun dossier de validation trouvé à: {val_images}. Utilisation de train/images (ou images) pour la validation.")

    # Vérification explicite des chemins dans data.yaml
    with open("data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("train", "val"):
        p = cfg.get(key)
        if not p or not os.path.isdir(p):
            print(f"❌ Chemin '{key}' introuvable: {p}. Corrigez data.yaml pour pointer vers un dossier d'images existant.")
            return

    model = YOLO("yolov8n.pt")
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=160,
        batch=32,
        device=device,
        project="models",
        name="yolo_pokemon_doors",
        plots=True,
        verbose=True,
        cache="ram",
        seed=42
    )

    print("\n✅ TERMINÉ !")
    print("💾 Poids: models/yolo_pokemon/weights/best.pt")

if __name__ == "__main__":
    main()