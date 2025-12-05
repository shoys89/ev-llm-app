import os
import sys
import streamlit as st

# Aseguramos que el proyecto raíz esté en sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.nlp.llm_ev_assistant import run_llm_assistant


def main():
    st.set_page_config(page_title="Asistente EV con LLM", page_icon="🤖")
    st.title("🤖 Asistente de Carga de Vehículos Eléctricos (LLM + Modelo HF)")
    st.write(
        "Este asistente usa un LLM (Groq + Qwen) para entender tu mensaje, "
        "completa la información de la sesión de carga y llama a tu modelo "
        "de predicción en Hugging Face para estimar la energía cargada."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Mostrar historial
    for role, msg in st.session_state.history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)

    # Input del usuario (estilo chat)
    user_msg = st.chat_input("Describe tu sesión de carga...")
    if user_msg:
        # Añadimos al historial
        st.session_state.history.append(("user", user_msg))
        st.chat_message("user").markdown(user_msg)

        with st.spinner("Pensando..."):
            try:
                answer = run_llm_assistant(user_msg)
            except Exception as exc:
                answer = (
                    "Ocurrió un error al procesar tu mensaje:\n\n"
                    f"`{exc}`\n\n"
                    "Verifica que las variables de entorno HF_TOKEN y GROQ_API_KEY "
                    "estén configuradas y que EV-DB.csv existe en la carpeta data/."
                )

        st.session_state.history.append(("assistant", answer))
        st.chat_message("assistant").markdown(answer)


if __name__ == "__main__":
    main()
