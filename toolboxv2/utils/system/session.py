import asyncio
import atexit
import os
import socket

try:
    from fastapi import Response
except ImportError:
    class Response:
        pass

import requests
from aiohttp import ClientResponse, ClientSession, MultipartWriter

from ... import Code
from ...tests.a_util import async_test
from ..extras.blobs import BlobFile
from ..singelton_class import Singleton
from .getting_and_closing_app import get_app, get_logger
from .types import Result


class RequestSession(Response):

    def __init__(self, session, body, json, row):
        super().__init__()
        self.session = session
        self._body = body
        self._json = json
        self.row = row

    def body(self):
        return self._body

    def json(self):
        if isinstance(self._json, dict):
            return self._json
        return self._json()

from aiohttp import ClientConnectorError, ClientError


class Session(metaclass=Singleton):

    # user: LocalUser

    def __init__(self, username, base=None):
        self.username = username
        self.session: ClientSession | None = None
        self.valid = False
        if base is None:
            base = os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app")
        if base is not None and base.endswith("/api/"):
            base = base.replace("api/", "")
        self.base = base
        self.base = base.rstrip('/')  # Ensure no trailing slash

        async def helper():
            try:
                await self.session.close() if self.session is not None else None
            except ClientConnectorError as e:
                print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
                return False
            except socket.gaierror as e:
                print(f"DNS-Auflösung fehlgeschlagen: {e}")
                return False
            except ClientError as e:
                print(f"Allgemeiner Client-Fehler: {e}")
                return False
            except Exception:
                print("No session to close for", self.username)

        atexit.register(async_test(helper))

    async def init_log_in_mk_link(self, mak_link):
        from urllib.parse import parse_qs, urlparse
        await asyncio.sleep(0.1)

        pub_key, prv_key = Code.generate_asymmetric_keys()
        Code.save_keys_to_files(pub_key, prv_key, get_app('sys.session').info_dir.replace(get_app('sys.session').id, ''))
        parsed_url = urlparse(mak_link)
        params = parse_qs(parsed_url.query)
        invitation = params.get('key', [None])[0]
        self.username = params.get('name', [get_app('sys.session').get_username()])[0]
        if not invitation:
            print('Invalid LoginKey')
            return False

        res = await get_app("Session.InitLogin").run_http("CloudM.AuthManager", "add_user_device", method="POST",
                                                          name=self.username, pub_key=pub_key, invitation=invitation,
                                                          web_data=False, as_base64=False)
        res = Result.result_from_dict(**res)
        if res.is_error():
            return res
        await asyncio.sleep(0.1)
        return await self.auth_with_prv_key()

    @staticmethod
    def get_prv_key():
        pub_key, prv_key = Code.load_keys_from_files(get_app('sys.session.get_prv_key').info_dir.replace(get_app('sys.session.get_prv_key').id, ''))
        return prv_key

    def if_key(self):
        return len(self.get_prv_key()) > 0

    async def auth_with_prv_key(self):
        prv_key = self.get_prv_key()
        challenge = await get_app("Session.InitLogin").run_http('CloudM.AuthManager', 'get_to_sing_data', method="POST",
                                                                args_='username=' + self.username + '&personal_key=False')

        challenge = Result.result_from_dict(**challenge)
        if challenge.is_error():
            return challenge.lazy_return(-1, "Server returned invalid data")

        await asyncio.sleep(0.1)
        claim_data = await get_app("Session.InitLogin").run_http('CloudM.AuthManager', 'validate_device',
                                                                 username=self.username,
                                                                 signature=Code.create_signature(
                                                                     challenge.get("challenge"), prv_key,
                                                                     salt_length=32),
                                                                 method="POST")
        claim_data = Result.result_from_dict(**claim_data)
        if claim_data.is_error():
            return claim_data.lazy_return(-1, "Server returned invalid data")

        claim = claim_data.get("key")

        if claim is None:
            return claim_data
        await asyncio.sleep(0.1)
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
            blob.clear()
            blob.write(claim.encode())
        await asyncio.sleep(0.1)
        # Do something with the data or perform further actions
        res = await self.login()
        return res

    def init(self):
        if self.session is None:
            self.session = ClientSession()

    async def login(self):
        if self.session is None:
            self.session = ClientSession()
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="r") as blob:
            claim = blob.read()
            if claim == b'Error decoding':
                blob.clear()
                claim = b''
        if not claim:
            res = await self.auth_with_prv_key()
            print(str(res))
            return res is True
        try:
            async with self.session.request("GET", url=f"{self.base}/validateSession", json={'Jwt_claim': claim.decode(),
                                                                                             'Username': self.username}) as response:
                print(response.status, "status")
                if response.status == 200:
                    print("Successfully Connected 2 TBxN")
                    get_logger().info("LogIn successful")
                    self.valid = True
                    return True
                if response.status == 401 and self.if_key():
                    return await self.auth_with_prv_key()
                print(response)
                get_logger().warning("LogIn failed")
                return False
        except ClientConnectorError as e:
            print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
            return False
        except socket.gaierror as e:
            print(f"DNS-Auflösung fehlgeschlagen: {e}")
            return False
        except ClientError as e:
            print(f"Allgemeiner Client-Fehler: {e}")
            return False
        except Exception as e:
            print("Connection error", self.username, e)
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
        try:
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
        except ClientConnectorError as e:
            print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
            return False
        except socket.gaierror as e:
            print(f"DNS-Auflösung fehlgeschlagen: {e}")
            return False
        except ClientError as e:
            print(f"Allgemeiner Client-Fehler: {e}")
            return False
        except Exception as e:
            print("Error:",e, self.username)
        return False

    async def logout(self) -> bool:
        if self.session:
            try:
                async with self.session.post(f'{self.base}/web/logoutS') as response:
                    await self.session.close()
                    self.session = None
                    return response.status == 200
            except ClientConnectorError as e:
                print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
                return False
            except socket.gaierror as e:
                print(f"DNS-Auflösung fehlgeschlagen: {e}")
                return False
            except ClientError as e:
                print(f"Allgemeiner Client-Fehler: {e}")
                return False
            except Exception as e:
                print("Error session logout:", e, self.username)
        return False

    async def fetch(self, url: str, method: str = 'GET', data=None) -> ClientResponse or Response:
        if isinstance(url, str):
            url = self.base + url
        if self.session:
            try:
                if method.upper() == 'POST':
                    return await self.session.post(url, json=data)
                else:
                    return await self.session.get(url)
            except ClientConnectorError as e:
                print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
                return False
            except socket.gaierror as e:
                print(f"DNS-Auflösung fehlgeschlagen: {e}")
                return False
            except ClientError as e:
                print(f"Allgemeiner Client-Fehler: {e}")
                return False
            except Exception as e:
                print("Error session fetch:", e, self.username)
                return requests.request(method, url, data=data)
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
        upload_url = self.base + upload_url
        # headers = {'accept': '*/*',
        #            'Content-Type': 'multipart/form-data'}
        # file_dict = {"file": open(file_path, "rb")}
        # response = await self.session.post(upload_url, headers=headers, data=file_dict)
        with open(file_path, 'rb') as f:
            file_data = f.read()

        with MultipartWriter('form-data') as mpwriter:
            part = mpwriter.append(file_data)
            part.set_content_disposition('form-data', name='file', filename=os.path.basename(file_path))
            try:
                async with self.session.post(upload_url, data=mpwriter, timeout=20000) as response:

                    # Prüfe, ob der Upload erfolgreich war
                    if response.status == 200:
                        print(f"Datei {file_path} erfolgreich hochgeladen.")
                        return await response.json()
                    else:
                        print(f"Fehler beim Hochladen der Datei {file_path}. Status: {response.status}")
                        print(await response.text())
                        return None
            except ClientConnectorError as e:
                print(f"Server nicht erreichbar (DNS oder Verbindung): {e}")
                return False
            except socket.gaierror as e:
                print(f"DNS-Auflösung fehlgeschlagen: {e}")
                return False
            except ClientError as e:
                print(f"Allgemeiner Client-Fehler: {e}")
                return False
            except Exception as e:
                print(f"Error session Fehler beim Hochladen der Datei {file_path}:", e, self.username)



    def exit(self):
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
            blob.clear()


async def helper_session_invalid():
    s = Session('root')

    t = await s.init_log_in_mk_link("/")
    print(t)
    t1 = await s.login()
    print(t1)
    assert not t1


def test_session_invalid():
    import asyncio

    asyncio.run(helper_session_invalid())


def test_session_invalid_log_in():
    import asyncio
    async def helper():
        s = Session('root')
        t1 = await s.login()
        print(t1)
        assert not t1

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
