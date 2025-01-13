# example_streamlit_app.py
import os

import streamlit as st
from toolboxv2.mods.FastApi.fast_lit import verify_streamlit_session, inject_custom_css

# Pfad zur CSS-Datei
css_file_path = "./web/assets/styles.css"

# CSS in die Streamlit-App injizieren
inject_custom_css(css_file_path)

# Beispielinhalt der Streamlit-App
st.title(f"{os.system('dir')}")
st.title("Meine Streamlit-App mit benutzerdefiniertem Styling")
st.write("Dieser Text sollte gemäß dem benutzerdefinierten CSS gestylt sein.")

# Beispiel für eine Komponente, die das CSS verwendet
st.markdown("""
<div class="container">
    <h1>Dies ist ein Beispiel-Titel</h1>
    <p>Dies ist ein Beispieltext, der mit dem benutzerdefinierten CSS gestylt wird.</p>
    <button>Ein Beispiel-Button</button>
</div>
""", unsafe_allow_html=True)

# Verify session at start
if verify_streamlit_session():
    # Access user data
    user_data = st.session_state.user_data
    st.write(f"Welcome, {user_data['user_name']}")
    st.write(user_data)


else:
    st.write("Not logged in")
