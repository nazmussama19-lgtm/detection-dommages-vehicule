from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CHEMIN_MODELE = Path(__file__).parent / "modele" / "classifieur_resnet50.pth"

# Ordre identique à celui des classes utilisé à l'entraînement
CLASSES = ["Front Breakage", "Front Crushed", "Front Normal", "Rear Breakage", "Rear Crushed", "Rear Normal"]

LIBELLES = {
    "Front Breakage": "Avant · pièce cassée",
    "Front Crushed": "Avant · carrosserie enfoncée",
    "Front Normal": "Avant · aucun dommage",
    "Rear Breakage": "Arrière · pièce cassée",
    "Rear Crushed": "Arrière · carrosserie enfoncée",
    "Rear Normal": "Arrière · aucun dommage",
}

TRANSFORMATION = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ClassifieurResNet(nn.Module):
    """ResNet50 dont la dernière couche est remplacée par une tête à 6 classes."""

    def __init__(self, nb_classes: int = len(CLASSES)):
        super().__init__()
        # Pas de poids ImageNet à télécharger : ils sont écrasés par ceux du modèle entraîné
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, nb_classes),
        )

    def forward(self, x):
        return self.model(x)


def charger_modele() -> ClassifieurResNet:
    modele = ClassifieurResNet()
    poids = torch.load(CHEMIN_MODELE, map_location="cpu", weights_only=True)
    modele.load_state_dict(poids)
    modele.eval()
    return modele


def predire(modele: ClassifieurResNet, image: Image.Image) -> list[tuple[str, float]]:
    """Renvoie les classes triées par probabilité décroissante."""
    tenseur = TRANSFORMATION(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilites = torch.softmax(modele(tenseur), dim=1)[0]
    resultats = [(classe, p.item()) for classe, p in zip(CLASSES, probabilites)]
    return sorted(resultats, key=lambda r: r[1], reverse=True)
