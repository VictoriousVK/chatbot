import cv2
import streamlit as st
import os

# Charger le classificateur HaarCascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Créer un dossier pour sauvegarder les visages
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

def detect_faces(save_images=False):
    cap = cv2.VideoCapture(0)
    face_id = 0  # compteur de visages sauvegardés

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Impossible d'accéder à la webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if save_images:
                face_id += 1
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"captured_faces/face_{face_id}.jpg", face_img)

        cv2.imshow("Détection de visages", frame)

        # 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("🧑‍💻 Reconnaissance Faciale Personnalisée")
    st.write("Ce système utilise l'algorithme Viola-Jones pour détecter des visages en temps réel")

    option = st.radio("Choisissez une option :", ["Détection simple", "Détection et sauvegarde des visages"])

    if st.button("Démarrer la webcam"):
        if option == "Détection simple":
            detect_faces(save_images=False)
        else:
            detect_faces(save_images=True)

    st.info("Appuyez sur la touche 'q' dans la fenêtre vidéo pour arrêter la détection.")


if __name__ == "__main__":
    app()
