import streamlit as st
import speech_recognition as sr

# -----------------------------
# Étape 1 : Fonction de reconnaissance vocale
# -----------------------------
def transcribe_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("🎤 Veuillez parler maintenant...")
        # Ajuster le bruit ambiant pour améliorer la précision
        r.adjust_for_ambient_noise(source, duration=1)

        try:
            audio_text = r.listen(source, timeout=5, phrase_time_limit=10)
            st.info("⏳ Transcription en cours...")

            # Utilisation de l’API Google
            text = r.recognize_google(audio_text, language="fr-FR")
            return text
        except sr.WaitTimeoutError:
            return "⚠️ Aucun son détecté, veuillez réessayer."
        except sr.UnknownValueError:
            return "❌ Désolé, je n’ai pas compris."
        except sr.RequestError:
            return "🚨 Erreur : Impossible d’accéder au service Google."
        except Exception as e:
            return f"Erreur inattendue : {e}"


# -----------------------------
# Étape 2 : Interface principale avec Streamlit
# -----------------------------
def main():
    st.set_page_config(page_title="Reconnaissance Vocale", page_icon="🎤")
    st.title("🎙️ Application de Reconnaissance Vocale")
    st.write("Cliquez sur le bouton ci-dessous et commencez à parler.")

    if st.button("🎤 Démarrer l’enregistrement"):
        text = transcribe_speech()
        st.success("✅ Transcription terminée :")
        st.write(f"**{text}**")

        # Bonus : enregistrer l’historique
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(text)

    # Affichage de l’historique des transcriptions
    if "history" in st.session_state and len(st.session_state.history) > 0:
        st.subheader("📝 Historique des transcriptions")
        for i, t in enumerate(st.session_state.history, 1):
            st.write(f"{i}. {t}")


# -----------------------------
# Étape 3 : Lancer l’application
# -----------------------------
if __name__ == "__main__":
    main()
