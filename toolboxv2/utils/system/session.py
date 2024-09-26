import asyncio
import atexit
import dataclasses
import os
import sys
import time
import socket
from requests import Response
from typing import Optional

import requests

from ..extras.blobs import BlobFile
from ..singelton_class import Singleton
from .getting_and_closing_app import get_app, get_logger

from aiohttp import ClientSession, ClientResponse, FormData

from ... import Code
from ...tests.a_util import async_test

from .types import Result


# @dataclasses.dataclass
# class LocalUser:
#    name:str
#    uid:str

class Session(metaclass=Singleton):

    # user: LocalUser

    def __init__(self, username, base=None):
        self.username = username
        self.session: Optional[ClientSession] = None
        self.valid = False
        if base is None:
            base = os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app")
        if base is not None and base.endswith("/api/"):
            base = base.replace("api/", "")
        self.base = base
        self.base = base.rstrip('/')  # Ensure no trailing slash

        async def helper():
            await self.session.close() if self.session is not None else None

        atexit.register(async_test(helper))

    async def init_log_in_mk_link(self, mak_link):
        from urllib.parse import urlparse, parse_qs
        await asyncio.sleep(0.1)

        if self.username is None or self.username == "":
            print("Please enter a username")
            return False

        print("Step (1/7)")
        pub_key, prv_key = Code.generate_asymmetric_keys()
        parsed_url = urlparse(mak_link)
        params = parse_qs(parsed_url.query)
        invitation = params.get('key', [None])[0]

        if not invitation:
            print('Invalid LoginKey')
            return False

        print("Step (2/7)")
        res = await get_app("Session.InitLogin").run_http("CloudM.AuthManager", "add_user_device", method="POST",
                                                          name=self.username, pub_key=pub_key, invitation=invitation,
                                                          web_data=False, as_base64=False)
        res = Result.result_from_dict(**res).print()
        if res.is_error():
            return res
        await asyncio.sleep(0.1)

        print("Step (3/7)")
        challenge = await get_app("Session.InitLogin").run_http('CloudM.AuthManager', 'get_to_sing_data', method="POST",
                                                                args_='username=' + self.username + '&personal_key=False')

        challenge = Result.result_from_dict(**challenge).print()
        if challenge.is_error():
            return challenge

        await asyncio.sleep(0.1)
        print("Step (4/7)")
        claim_data = await get_app("Session.InitLogin").run_http('CloudM.AuthManager', 'validate_device',
                                                                 username=self.username,
                                                                 signature=Code.create_signature(challenge.get("challenge"), prv_key, salt_length=32),
                                                                 method="POST")
        claim_data = Result.result_from_dict(**claim_data).print()

        claim = claim_data.get("key")

        print("claim:", claim)
        if claim is None:
            return claim_data
        await asyncio.sleep(0.1)
        print("Step (5/7)")
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
            blob.clear()
            blob.write(claim.encode())
        print("Step (6/7)")
        await asyncio.sleep(0.1)
        # Do something with the data or perform further actions
        res = await self.login()
        print("Step (7/7)")
        return res

    async def login(self):
        if self.session is None:
            self.session = ClientSession()
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="r") as blob:
            claim = blob.read()
            if claim == b'Error decoding':
                blob.clear()
                claim = b''
        if not claim:
            return False
        print("Claim :",claim)
        async with self.session.request("GET", url=f"{self.base}/validateSession", json={'Jwt_claim': claim.decode(),
                                                                                         'Username': self.username}) as response:
            if response.status == 200:
                print("Successfully Connected 2 TBxN")
                get_logger().info("LogIn successful")
                self.valid = True
                return True
            get_logger().warning("LogIn failed")
            return False

    async def download_file(self, url, dest_folder="mods_sto"):
        if not self.session:
            raise Exception("Session not initialized. Please login first.")
        # Sicherstellen, dass das Zielverzeichnis existiert
        os.makedirs(dest_folder, exist_ok=True)

        # Analyse der URL, um den Dateinamen zu extrahieren
        filename = url.split('/')[-1]

        # Bereinigen des Dateinamens von Sonderzeichen
        valid_chars = '-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        filename = ''.join(char for char in filename if char in valid_chars)

        # Konstruieren des vollständigen Dateipfads
        file_path = os.path.join(dest_folder, filename)
        if isinstance(url, str):
            url = self.base + url
        async with self.session.get(url) as response:
            if response.status == 200:
                with open(file_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f'File downloaded: {file_path}')
                return True
            else:
                print(f'Failed to download file: {url}. Status code: {response.status}')
        return False

    async def logout(self) -> bool:
        if self.session:
            async with self.session.post(f'{self.base}/web/logoutS') as response:
                await self.session.close()
                self.session = None
                return response.status == 200
        return False

    async def fetch(self, url: str, method: str = 'GET', data=None) -> ClientResponse or Response:
        if isinstance(url, str):
            url = self.base + url
        if self.session:
            if method.upper() == 'POST':
                return await self.session.post(url, json=data)
            else:
                return await self.session.get(url)
        else:
            print(f"Could not find session using request on {url}")
            if method.upper() == 'POST':
                return requests.request(method, url, json=data)
            return requests.request(method, url, data=data)
            # raise Exception("Session not initialized. Please login first.")

    async def upload_file(self, file_path: str, upload_url: str):
        # Prüfe, ob die Datei existiert
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Datei {file_path} nicht gefunden.")

        # Initialisiere die Session, falls sie nicht bereits gestartet ist
        if self.session is None:
            self.session = ClientSession()

        # Bereite die Datei für den Upload vor
        form = FormData()
        form.add_field('file',
                       open(file_path, 'rb'),
                       filename=os.path.basename(file_path),
                       content_type='application/octet-stream')

        # Erstelle die vollständige URL
        full_url = self.base + upload_url

        # Sende die Datei
        async with self.session.post(full_url, data=form) as response:
            # Prüfe, ob der Upload erfolgreich war
            if response.status == 200:
                print(f"Datei {file_path} erfolgreich hochgeladen.")
                return await response.json()
            else:
                print(f"Fehler beim Hochladen der Datei {file_path}. Status: {response.status}")
                return None

    def exit(self):
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
            blob.clear()


async def helper_session_invalid():
    s = Session('root')

    t = await s.init_log_in_mk_link("/")
    print(t)
    t1 = await s.login()
    print(t1)
    assert t1 == False


def test_session_invalid():
    import asyncio

    asyncio.run(helper_session_invalid())


def test_session_invalid_log_in():
    import asyncio
    async def helper():
        s = Session('root')
        t1 = await s.login()
        print(t1)
        assert t1 == False

    asyncio.run(helper())


def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        ip_address = response.json()['ip']
        return ip_address
    except Exception as e:
        print(f"Fehler beim Ermitteln der öffentlichen IP-Adresse: {e}")
        return None


def get_local_ip():
    try:
        # Erstellt einen Socket, um eine Verbindung mit einem öffentlichen DNS-Server zu simulieren
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Verwendet Google's öffentlichen DNS-Server als Ziel, ohne tatsächlich eine Verbindung herzustellen
            s.connect(("8.8.8.8", 80))
            # Ermittelt die lokale IP-Adresse, die für die Verbindung verwendet würde
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception as e:
        print(f"Fehler beim Ermitteln der lokalen IP-Adresse: {e}")
        return None
