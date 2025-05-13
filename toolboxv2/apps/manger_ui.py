import streamlit as st
import requests
from typing import Optional
from pathlib import Path

# FastAPI Backend URL
API_URL = "http://localhost:5000/apps"


class AppManagerClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        # Session für Request-Wiederverwendung
        self.session = requests.Session()

    def register_app(self, name: str, file_path: str) -> dict:
        response = self.session.post(
            f"{self.api_url}/apps/",
            json={"name": name, "file_path": file_path}
        )
        return response.json()

    def start_app(self, name: str) -> dict:
        response = self.session.post(f"{self.api_url}/apps/{name}/start")
        return response.json()

    def stop_app(self, name: str) -> dict:
        response = self.session.post(f"{self.api_url}/apps/{name}/stop")
        return response.json()

    def list_apps(self) -> list:
        response = self.session.get(f"{self.api_url}/apps/")
        return response.json()

    def delete_app(self, name: str) -> dict:
        response = self.session.delete(f"{self.api_url}/apps/{name}")
        return response.json()


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Speichert eine hochgeladene Datei und gibt den Pfad zurück"""
    if uploaded_file is None:
        return None

    # Speicherverzeichnis erstellen
    save_dir = Path("uploaded_apps")
    save_dir.mkdir(exist_ok=True)

    # Datei speichern
    file_path = save_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path.absolute())


def main():
    st.set_page_config(page_title="Streamlit App Manager", layout="wide")
    st.title("Streamlit App Manager")

    # Initialize API client
    client = AppManagerClient(API_URL)

    # Sidebar für App-Registration
    with st.sidebar:
        st.header("Neue App registrieren")

        # App-Name Eingabe
        app_name = st.text_input("App Name")

        # File Upload
        uploaded_file = st.file_uploader("Streamlit App File (.py)", type=['py'])

        if st.button("App registrieren") and app_name and uploaded_file:
            try:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    response = client.register_app(app_name, file_path)
                    st.success(f"App '{app_name}' erfolgreich registriert!")
            except requests.exceptions.RequestException as e:
                st.error(f"Fehler bei der Registrierung: {str(e)}")

    # Hauptbereich: App-Übersicht
    try:
        apps = client.list_apps()

        # Apps in Karten darstellen
        cols = st.columns(3)
        for idx, app in enumerate(apps):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"### {app['name']}")
                    st.text(f"Status: {'Aktiv' if app['running'] else 'Gestoppt'}")
                    st.text(f"Port: {app['port']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if not app['running']:
                            if st.button("Start", key=f"start_{app['name']}"):
                                response = client.start_app(app['name'])
                                st.success(response['message'])
                                st.rerun()
                        else:
                            if st.button("Stop", key=f"stop_{app['name']}"):
                                response = client.stop_app(app['name'])
                                st.success(response['message'])
                                st.rerun()

                    with col2:
                        if st.button("Löschen", key=f"delete_{app['name']}"):
                            response = client.delete_app(app['name'])
                            st.success(response['message'])
                            st.rerun()

                    if app['running']:
                        st.markdown(f"🔗 [App öffnen](http://localhost:{app['port']})")

                    # Trennlinie
                    st.markdown("---")

    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Laden der Apps: {str(e)}")


if __name__ == "__main__":
    main()
