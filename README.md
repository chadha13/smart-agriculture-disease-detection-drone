# Agriculture intelligente : drone de détection des maladies des cultures

Prototype de drone qui capture des images de feuilles avec une **ESP32-CAM** et les classe avec un modèle **MobileNetV2** entraîné sur **PlantVillage**. Le vol est assuré par un contrôleur de vol configuré avec **Betaflight**.

> Démo : _[ajouter le lien vidéo ou un GIF]_

## Fonctionnement

```
 ESP32-CAM (Wi-Fi) ── /capture ou :81/stream ──► PC ── predict.py ──► maladie + confiance
 (embarquée sur le drone)                          (MobileNetV2, PlantVillage)
```

1. L'ESP32-CAM se connecte au Wi-Fi et expose une image JPEG (`/capture`) et un flux MJPEG (`:81/stream`).
2. Le script `predict.py` récupère une image, la redimensionne en 224 × 224 et la classe.
3. Le contrôleur de vol (Betaflight) est indépendant : il ne commande que le vol.

## Structure

```
firmware/esp32cam_server/   sketch Arduino de l'ESP32-CAM (AI Thinker)
model/train_mobilenet.py    entraînement, évaluation et export (Keras + TFLite)
model/predict.py            classification d'un fichier ou d'une image de l'ESP32-CAM
docs/                       schémas, photos du montage, configuration Betaflight
```

## Utilisation

**1. Firmware**
- Arduino IDE, carte « AI Thinker ESP32-CAM » (paquet *esp32* d'Espressif).
- Copier `secrets.h.example` en `secrets.h` et renseigner le Wi-Fi, puis téléverser.
- Le moniteur série (115200) affiche l'adresse IP de la caméra.

**2. Modèle**
```bash
cd model
pip install -r requirements.txt
python train_mobilenet.py --data_dir ../data/PlantVillage --epochs 8 --finetune_epochs 4
```
Le jeu de données PlantVillage est à télécharger séparément (non inclus ici, voir sa licence).

**3. Prédiction**
```bash
python predict.py --url http://<IP>/capture
python predict.py --image feuille.jpg
```

## Résultats
_[À remplir avec les chiffres réels de `out/report.txt` après l'entraînement : exactitude sur le jeu de test, matrice de confusion, quelques exemples.]_

## Limites
- PlantVillage contient des feuilles photographiées sur fond uniforme, et plusieurs images viennent de la même feuille : le découpage aléatoire peut gonfler l'exactitude, et la performance peut baisser sur des images prises par le drone en extérieur.
- Vibrations, éclairage, flou de mouvement et résolution de l'ESP32-CAM ne sont pas encore évalués.
- La classification est faite sur un PC relié en Wi-Fi, pas sur le drone.

## Pistes d'amélioration
- Évaluer sur des images réelles prises par le drone
- Embarquer le modèle (TFLite) pour une détection en vol
- Géolocaliser les zones malades (GPS)

## Licence
_[à choisir, par exemple MIT]_
