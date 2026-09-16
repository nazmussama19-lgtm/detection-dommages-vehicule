# Détection de dommages sur véhicule

À partir de la photo d'une voiture, repérer si l'avant ou l'arrière est endommagé et qualifier le dommage (pièce cassée, carrosserie enfoncée ou aucun dommage). Cas d'usage visé : un premier tri automatique des photos envoyées lors d'une déclaration de sinistre.

**[→ Tester l'application](https://LIEN-A-COMPLETER.streamlit.app)**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

![Aperçu de l'application](assets/apercu.jpg)

---

## Modèle

- **Apprentissage par transfert** à partir d'un ResNet50 pré-entraîné sur ImageNet : les couches convolutives sont gelées sauf le dernier bloc (`layer4`), et la couche finale est remplacée par une tête à 6 classes (dropout 0,2 + couche linéaire).
- **Environ 1 700 images** réparties en 6 classes :

| Zone | Classes |
|---|---|
| Avant | pièce cassée · carrosserie enfoncée · aucun dommage |
| Arrière | pièce cassée · carrosserie enfoncée · aucun dommage |

- **Précision d'environ 80 %** sur le jeu de validation.

L'application affiche la classe prédite et la probabilité associée à chacune des 6 classes, ce qui permet de repérer les cas où le modèle hésite.

## Choix techniques

**Un modèle léger à déployer.** Les poids sont stockés en demi-précision (float16), ce qui divise la taille du fichier par deux (environ 47 Mo au lieu de 91 Mo) et évite de passer par Git LFS. Ils sont reconvertis en float32 au chargement : la classe prédite reste la même qu'avec le modèle d'origine, et les probabilités diffèrent de moins de 0,2 point.

**Inférence sur CPU.** Le modèle a été entraîné sur GPU mais se charge sur CPU, et `requirements.txt` installe la version CPU de PyTorch, bien plus légère. L'application tourne ainsi sur Streamlit Community Cloud.

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
- **Évaluation sommaire.** Seule la précision globale est connue ; une matrice de confusion permettrait de voir quelles classes sont confondues (par exemple « cassée » et « enfoncée »).

## Données

Jeu d'images de véhicules issu d'un cas pratique de formation (Codebasics).

## Auteur

**Nazmus Sama** · [GitHub](https://github.com/nazmussama19-lgtm)
