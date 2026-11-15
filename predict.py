"""Classe une image : fichier local ou image récupérée depuis l'ESP32-CAM.

Exemples :
    python predict.py --image feuille.jpg
    python predict.py --url http://192.168.1.50/capture
    python predict.py --url http://192.168.1.50/capture --loop 2     # une image toutes les 2 s
"""
import argparse
import io
import json
import os
import time

import numpy as np
import requests
import tensorflow as tf
from PIL import Image


def load_image(args, size):
    if args.url:
        r = requests.get(args.url, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(args.image)
    img = img.convert("RGB").resize((size, size))
    return np.expand_dims(np.asarray(img, dtype=np.float32), 0)   # 0..255, la normalisation est dans le modèle


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--url", help="ex. http://<IP de l'ESP32-CAM>/capture")
    p.add_argument("--model_dir", default="out")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--loop", type=float, default=0, help="secondes entre deux images (0 = une seule)")
    args = p.parse_args()

    model = tf.keras.models.load_model(os.path.join(args.model_dir, "model.keras"))
    with open(os.path.join(args.model_dir, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)

    while True:
        probs = model.predict(load_image(args, args.img_size), verbose=0)[0]
        best = np.argsort(probs)[::-1][: args.top]
        print(" | ".join(f"{labels[i]} {probs[i]:.1%}" for i in best))
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
