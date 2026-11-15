"""Entraîne un classifieur de maladies de feuilles (MobileNetV2, transfert d'apprentissage).

Jeu de données attendu : PlantVillage, un sous-dossier par classe
    data/PlantVillage/<classe>/*.jpg

Exemple :
    python train_mobilenet.py --data_dir data/PlantVillage --epochs 8 --finetune_epochs 4

Sorties dans --out_dir : model.keras, model.tflite, labels.json, report.txt
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def load_splits(data_dir, img_size, batch, seed):
    """70 % entraînement / 15 % validation / 15 % test."""
    common = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch,
        seed=seed,
        validation_split=0.30,
    )
    train = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    # Ordre fixe pour pouvoir couper proprement le reste en validation / test.
    rest = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **common)
    class_names = train.class_names
    n = tf.data.experimental.cardinality(rest).numpy()
    val = rest.take(n // 2)
    test = rest.skip(n // 2)
    prefetch = tf.data.AUTOTUNE
    return train.prefetch(prefetch), val.prefetch(prefetch), test.prefetch(prefetch), class_names


def build_model(num_classes, img_size):
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))   # pixels 0..255
    x = augment(inputs)                                       # actif seulement à l'entraînement
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # équivalent de preprocess_input
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--finetune_epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train, val, test, class_names = load_splits(args.data_dir, args.img_size, args.batch, args.seed)
    print(f"{len(class_names)} classes : {class_names}")

    model, base = build_model(len(class_names), args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)

    if args.finetune_epochs > 0:
        # Dégel des dernières couches de MobileNetV2 avec un petit pas d'apprentissage.
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train, validation_data=val, epochs=args.finetune_epochs, callbacks=callbacks)

    # Évaluation sur le jeu de test (jamais vu pendant l'entraînement)
    y_true = np.concatenate([y.numpy() for _, y in test])
    y_prob = model.predict(test.map(lambda x, y: x), verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = float(np.mean(y_true == y_pred))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nExactitude test : {acc:.3f}\n{report}")
    with open(os.path.join(args.out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Exactitude test : {acc:.4f}\n\n{report}\nMatrice de confusion :\n{cm}\n")

    model.save(os.path.join(args.out_dir, "model.keras"))
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Export TensorFlow Lite (pour un usage embarqué ultérieur)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(os.path.join(args.out_dir, "model.tflite"), "wb") as f:
        f.write(converter.convert())
    print(f"Modèle enregistré dans {args.out_dir}/")


if __name__ == "__main__":
    main()
