import streamlit as st
from PIL import Image, UnidentifiedImageError

from prediction import LIBELLES, charger_modele, predire

st.set_page_config(page_title="Détection de dommages sur véhicule", page_icon="🚗", layout="centered")


@st.cache_resource
def modele():
    return charger_modele()


st.title("Détection de dommages sur véhicule")
st.write(
    "Déposez la photo d'une voiture vue de trois quarts avant ou arrière : "
    "le modèle indique la zone concernée et le type de dommage."
)

fichier = st.file_uploader("Photo du véhicule", type=["jpg", "jpeg", "png"])

if fichier:
    try:
        image = Image.open(fichier)
    except UnidentifiedImageError:
        st.error("Ce fichier n'est pas une image lisible.")
        st.stop()

    colonne_image, colonne_resultat = st.columns([3, 2], gap="large")
    colonne_image.image(image, width="stretch")

    with st.spinner("Analyse en cours…"):
        resultats = predire(modele(), image)

    classe, probabilite = resultats[0]
    colonne_resultat.caption("Diagnostic")
    colonne_resultat.subheader(LIBELLES[classe], anchor=False)
    colonne_resultat.write(f"Confiance : **{probabilite:.0%}**")
    colonne_resultat.divider()
    colonne_resultat.caption("Probabilité par classe")
    for classe, probabilite in resultats:
        colonne_resultat.progress(probabilite, text=f"{LIBELLES[classe]} · {probabilite:.0%}")
