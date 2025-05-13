import importlib
import os
from importlib.machinery import ModuleSpec

import requests


class GistLoader:
    def __init__(self, gist_url):
        self.gist_url = gist_url
        self.module_code = None

    def load_module(self, module_name):
        """Lädt das Modul mit dem gegebenen Namen."""
        if self.module_code is None:
            self.module_code = self._fetch_gist_content()

        # Erstelle ein neues Modul
        module = importlib.util.module_from_spec(self.get_spec(module_name))
        exec(self.module_code, module.__dict__)
        return module

    def get_spec(self, module_name):
        """Gibt die Modul-Specifikation zurück."""
        return ModuleSpec(module_name, self)

    def get_filename(self, module_name):
        return f"<gist:{self.gist_url}>"

    def _fetch_gist_content(self):
        """Lädt den Inhalt des Gists von der GitHub API herunter."""
        gist_id = self.gist_url.split('/')[-1]
        api_url = f"https://api.github.com/gists/{gist_id}"

        response = requests.get(api_url)

        if response.status_code == 200:
            gist_data = response.json()
            first_file = next(iter(gist_data['files'].values()))
            return first_file['content']
        else:
            raise Exception(f"Failed to fetch gist: {response.status_code}")


def create_or_update_gist(file_path, description="", public=True, gist_id=None, token=None):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the content of the file
    with open(file_path) as file:
        content = file.read()

    if token is None:
        token = os.getenv("GITHUB_TOKEN_GIST")

    # Prepare the data for the request
    file_name = os.path.basename(file_path)
    gist_data = {
        "description": description,
        "public": public,
        "files": {
            file_name: {
                "content": content
            }
        }
    }

    # Prepare the headers, including the authorization token if provided
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    if gist_id:
        # Update an existing Gist
        url = f"https://api.github.com/gists/{gist_id}"
        response = requests.patch(url, json=gist_data, headers=headers)
    else:
        # Create a new Gist
        url = "https://api.github.com/gists"
        response = requests.post(url, json=gist_data, headers=headers)

    # Check if the request was successful
    if response.status_code in (200, 201):
        gist_url = response.json().get('html_url')
        print(f"Gist {'updated' if gist_id else 'created'} successfully: {gist_url}")
        return response.json()
    else:
        print(f"Failed to {'update' if gist_id else 'create'} the Gist. Status Code: {response.status_code}")
        print(response.json())
        return None
