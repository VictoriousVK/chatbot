import cv2
import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle

# Choisir UNE des deux bibliothèques pour l'encodage
# import face_recognition
from deepface import DeepFace

# Charger le classificateur pour la détection initiale
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dossier pour sauvegarder les visages enrôlés
DATA_DIR = "face_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Charger la base de données des visages connus
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    # On suppose que les données sont sauvegardées dans un fichier pickle
    if os.path.exists(os.path.join(DATA_DIR, "faces.pkl")):
        with open(os.path.join(DATA_DIR, "faces.pkl"), 'rb') as f:
            known_data = pickle.load(f)
            known_face_encodings = known_data["encodings"]
            known_face_names = known_data["names"]
    return known_face_encodings, known_face_names

def enroll_new_face(name):
    cap = cv2.VideoCapture(0)
    st.warning("Veuillez vous positionner face à la caméra. Appuyez sur 's' pour sauvegarder, 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Enrôlement - Prenez une photo', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(faces) == 1:
            # On capture le visage détecté
            face_img = frame[y:y+h, x:x+w]
            # On le convertit en RGB (format attendu par les bibliothèques)
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # === METHODE AVEC DEEPFACE ===
            try:
                # DeepFace permet aussi de trouver l'embedding
                embedding_obj = DeepFace.represent(rgb_face, model_name='VGG-Face', enforce_detection=False)
                embedding = embedding_obj[0]["embedding"]
            except Exception as e:
                st.error(f"Erreur lors de la création de l'empreinte : {e}")
                break

            # Charger les données existantes, ajouter les nouvelles et sauvegarder
            known_encodings, known_names = load_known_faces()
            known_encodings.append(embedding)
            known_names.append(name)

            with open(os.path.join(DATA_DIR, "faces.pkl"), 'wb') as f:
                pickle.dump({"encodings": known_encodings, "names": known_names}, f)

            st.success(f"Visage de {name} enrôlé avec succès !")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    known_face_encodings, known_face_names = load_known_faces()
    if len(known_face_encodings) == 0:
        st.error("Aucun visage enrôlé dans la base de données. Veuillez d'abord enrôler des visages.")
        return

    cap = cv2.VideoCapture(0)
    st.write("Reconnaissance en cours... Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # Réduction pour plus de vitesse
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Détecter les visages avec la cascade (plus rapide)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        name = "Inconnu"
        for (x, y, w, h) in faces:
            # Agrandir les coordonnées pour les ramener à la taille originale de l'image
            x, y, w, h = x*2, y*2, w*2, h*2
            face_img = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                # === COMPARAISON AVEC DEEPFACE ===
                result = DeepFace.find(rgb_face, db_path=DATA_DIR, model_name='VGG-Face', enforce_detection=False, silent=True)
                if result and not result[0].empty:
                    best_match_distance = result[0]['VGG-Face_cosine'].iloc[0]
                    if best_match_distance < 0.5: # Seuil de tolérance à ajuster
                        # Trouver le nom correspondant dans la base de données
                        # (Cette partie nécessite de lier l'embedding au nom, plus complexe.
                        # Pour simplifier, on affichera "Reconnu").
                        name = "Reconnu!"
                    else:
                        name = "Inconnu"
                else:
                    name = "Inconnu"
            except:
                name = "Inconnu"

            # Dessiner le rectangle et le nom
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher le flux avec Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Système Personnalisé de Reconnaissance Faciale")
    st.markdown("""
    **Fonctionnalités :**
    - **Détection :** Trouve les visages dans le flux vidéo.
    - **Enrôlement :** Enregistre un nouveau visage dans la base de données.
    - **Reconnaissance :** Identifie les personnes enrôlées.
    """)

    menu = st.sidebar.selectbox("Menu", ["Détection Simple", "Enrôlement", "Reconnaissance"])

    if menu == "Détection Simple":
        st.header("Détection de Visages (Algorithme de Viola-Jones)")
        if st.button("Démarrer la Détection"):
            detect_faces() # Votre fonction originale

    elif menu == "Enrôlement":
        st.header("Enrôler un Nouveau Visage")
        name = st.text_input("Entrez le nom de la personne :")
        if name and st.button("Commencer l'Enrôlement"):
            enroll_new_face(name)

    elif menu == "Reconnaissance":
        st.header("Reconnaissance Faciale")
        if st.button("Démarrer la Reconnaissance"):
            recognize_faces()

if __name__ == "__main__":
    app()