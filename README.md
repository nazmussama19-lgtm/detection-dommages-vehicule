# Détection de dommages sur véhicule

À partir de la photo d'une voiture, repérer si l'avant ou l'arrière est endommagé et qualifier le dommage (pièce cassée, carrosserie enfoncée ou aucun dommage). Cas d'usage visé : un premier tri automatique des photos envoyées lors d'une déclaration de sinistre.

**[→ Tester l'application](https://LIEN-A-COMPLETER.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-1D4E47)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

![Aperçu de l'application](assets/apercu.jpg)

---

## Résultats

Le modèle retenu est un **ResNet50 pré-entraîné sur ImageNet**, dont le dernier bloc a été réentraîné sur les photos de véhicules. Mesuré sur 575 images de validation :

| Indicateur | Valeur |
|---|---|
| Précision globale | **79 %** |
| F1 macro | **0,79** |
| Zone (avant / arrière) correctement identifiée | **99,3 %** (4 erreurs sur 575) |
| Dommages arrière classés « aucun dommage » | 13 % (21 sur 161) |
| Dommages avant classés « aucun dommage » | 1,8 % (4 sur 217) |
| Véhicules intacts signalés comme endommagés | 14 % (27 sur 197) |

Détail par classe :

| Classe | Précision | Rappel | F1 | Images |
|---|---|---|---|---|
| Avant · pièce cassée | 0,73 | 0,93 | 0,82 | 119 |
| Avant · carrosserie enfoncée | 0,82 | 0,69 | 0,75 | 98 |
| Avant · aucun dommage | 0,96 | 0,81 | 0,88 | 122 |
| Arrière · pièce cassée | 0,84 | 0,65 | 0,74 | 81 |
| Arrière · carrosserie enfoncée | 0,70 | 0,71 | 0,71 | 80 |
| Arrière · aucun dommage | 0,75 | 0,92 | 0,83 | 75 |

### Comparaison des modèles

Tous entraînés 10 époques avec Adam, sur le même découpage :

| Modèle | Précision en validation |
|---|---|
| CNN à 3 couches, entraîné de zéro | 57,7 % |
| Même CNN + BatchNorm, dropout et weight decay | 50,4 % |
| EfficientNet-B0 pré-entraîné, seule la tête entraînée | 65,7 % |
| ResNet50 pré-entraîné, dernier bloc réentraîné (lr 0,001, dropout 0,5) | 76,7 % |
| **ResNet50, lr 0,005, dropout 0,2 (modèle retenu)** | **79,5 %** |

## Ce que l'analyse a montré

**Avec 1 725 images d'entraînement, partir de zéro ne suffit pas.** Le CNN simple plafonne sous 58 %, et la régularisation le fait descendre à 50 %. Un ResNet50 pré-entraîné fait 22 points de mieux.

**Le modèle sait où regarder, pas toujours ce qu'il voit.** Il ne confond presque jamais l'avant et l'arrière. Plus de la moitié des erreurs (62 sur 118) portent sur la différence entre pièce cassée et carrosserie enfoncée : 28 % des « avant enfoncé » sont pris pour des « avant cassé », et 25 % des « arrière cassé » pour des « arrière enfoncé ».

**Le vrai risque métier est à l'arrière.** Un dommage sur 8 à l'arrière passe pour un véhicule intact, contre moins de 2 % à l'avant. Pour un tri de sinistres, c'est l'erreur qui coûte le plus, et c'est là qu'il faudrait des images en plus.

**Le réglage des hyperparamètres a peu d'effet.** Sur 20 essais Optuna, ceux menés à terme avec un taux d'apprentissage entre 2·10⁻⁴ et 8·10⁻³ donnent tous entre 76 et 80 %, un écart comparable à la variation d'une époque à l'autre. Seuls les taux trop faibles (moins de 5·10⁻⁵) décrochent nettement, à 60–65 %.

**Le modèle surapprend dès la 3ᵉ époque.** La perte d'entraînement passe de 1,00 à 0,11, alors que la précision en validation ne progresse plus et oscille entre 74 et 81 % à partir de la 3ᵉ époque.

## Démarche

1. **Données** : 2 300 photos de véhicules vues de trois quarts, rangées en 6 classes, découpées aléatoirement en 1 725 images d'entraînement et 575 de validation.
2. **Prétraitement** : redimensionnement en 224 × 224 et normalisation ImageNet. Augmentation de données à l'entraînement : retournement horizontal, rotation de ±10°, luminosité et contraste de ±20 %.
3. **Modèles de référence** : un CNN entraîné de zéro, puis sa version régularisée.
4. **Transfert d'apprentissage** : EfficientNet-B0, puis ResNet50 avec toutes les couches gelées sauf le dernier bloc (`layer4`) et une nouvelle tête (dropout + couche linéaire à 6 sorties).
5. **Hyperparamètres** : recherche Optuna sur le taux d'apprentissage (10⁻⁵ à 10⁻²) et le dropout (0,2 à 0,7), 20 essais de 3 époques, arrêt des essais peu prometteurs.
6. **Entraînement final** : 10 époques, Adam, entropie croisée, lots de 32 images, sur GPU (environ 17 minutes).
7. **Évaluation** : rapport de classification et matrice de confusion.
8. **Mise en production** : export des poids en float16, application Streamlit sur CPU.

## Structure

```
├── app.py                              interface Streamlit
├── prediction.py                       architecture du modèle, chargement et prédiction
├── modele/classifieur_resnet50.pth     poids du modèle entraîné
├── assets/apercu.jpg
├── .streamlit/config.toml              thème
└── requirements.txt
```

## Lancer le projet

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

## Limites

- **Angle de prise de vue imposé.** Le modèle a été entraîné sur des vues de trois quarts avant ou arrière : une photo de profil ou un gros plan donnera un résultat peu fiable.
- **Pas de classe « hors sujet ».** Une image qui ne montre pas de voiture sera tout de même classée dans l'une des 6 catégories.
- **Un score un peu optimiste.** Il n'y a pas de jeu de test séparé : les mêmes 575 images ont servi à comparer les modèles, régler les hyperparamètres et mesurer le résultat final. L'augmentation de données s'appliquait aussi à la validation et aucune graine aléatoire n'était fixée, d'où des scores qui varient de ±3 points d'une époque à l'autre.
- **Pas d'arrêt anticipé.** C'est la 10ᵉ époque qui est conservée, alors que la 5ᵉ atteignait 81,2 %.
- **Une comparaison inégale.** EfficientNet-B0 a été testé entièrement gelé, ResNet50 avec son dernier bloc réentraîné. L'avance de ResNet vient en partie de là.

## Auteur

**Nazmus Sama** · [GitHub](https://github.com/nazmussama19-lgtm)
