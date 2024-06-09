import asyncio
import atexit
import dataclasses
import os
import sys
import time
import socket
from typing import Optional

import requests
from requests import Response

from ..extras.blobs import BlobFile
from ..singelton_class import Singleton
from ..toolbox import App
from .getting_and_closing_app import get_app, get_logger
from . import all_functions_enums as tbef

from aiohttp import ClientSession, ClientResponse
from yarl import URL

from ... import Code, Spinner
from ...tests.a_util import async_test


# @dataclasses.dataclass
# class LocalUser:
#    name:str
#    uid:str

class Session(metaclass=Singleton):

    # user: LocalUser

    def __init__(self, username, base=None):
        self.username = username
        self.session: Optional[ClientSession] = None
        if base is None:
            base = os.environ.get("TOOLBOXV2_REMOTE_BASE", "https://simplecore.app")
        if base is not None and base.endswith("/api/"):
            base = base.replace("api/", "")
        self.base = base
        self.base = base.rstrip('/')  # Ensure no trailing slash

        async def helper():
            await self.session.close() if self.session is not None else None

        atexit.register(async_test(helper))

    async def init_log_in_mk_link(self, mak_link, download=True, b_name="chromium", headless=False):
        from playwright.async_api import async_playwright
        await asyncio.sleep(0.1)
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(
                    headless=headless)  # Set headless=False if you want to see the browser UI
            except Exception as e:
                if download and "Executable doesn't exist at" in str(e):
                    print("starting installation")
                    os.system(sys.executable + ' -m playwright install ' + b_name + ' --with-deps --force')
                if not download:
                    return "install a browser"
                browser = await playwright.chromium.launch(
                    headless=headless)
            context = await browser.new_context()

            # Open a new page
            page = await context.new_page()

            # Navigate to a URL that sets something in localStorage
            if mak_link.startswith(self.base):
                mak_link = mak_link.replace(self.base, "")

            await page.goto(f"{self.base}/{mak_link}")  # Replace with the actual URL that uses localStorage
            # Retrieve data from localStorage
            await asyncio.sleep(1)
            await page.wait_for_load_state("networkidle", timeout=40 * 60)
            started = await page.evaluate("localStorage.getItem('StartMLogIN')")
            if started is None:
                get_logger().error("Could not found the startMLogIN flag")
                await browser.close()
                return False
            print("Step (1/7)")
            with Spinner("Waiting for Log in", count_down=True, time_in_s=6):
                await asyncio.sleep(6)
            print("Step (2/7)")
            await page.wait_for_load_state("networkidle", timeout=240 * 60)
            claim = await page.evaluate("localStorage.getItem('jwt_claim_device')")
            print("claim: ", len(claim))
            print("Step (3/7)")
            if claim is None:
                get_logger().error("No claim Received")
                await browser.close()
                return False
            print("Step (4/7)")
            with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
                blob.clear()
                blob.write(claim.encode())
            print("Step (5/7)")
            # Do something with the data or perform further actions

            # Close browser
            await browser.close()
        print("Step (6/7)")
        res = await self.login()
        print("Step (7/7)")
        return res

    async def login(self):
        if self.session is None:
            self.session = ClientSession()
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="r") as blob:
            claim = blob.read()
        if not claim:
            return False

        async with self.session.request("GET", url=f"{self.base}/validateSession", json={'Jwt_claim': claim.decode(),
                                                                                         'Username': self.username}) as response:
            if response.status == 200:
                print("Successfully Connected 2 TBxN")
                get_logger().info("LogIn successful")
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
            url = URL(self.base + url)
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

    async def fetch(self, url: URL or str, method: str = 'GET', data=None) -> ClientResponse:
        if isinstance(url, str):
            url = URL(self.base + url)
        if self.session:
            if method.upper() == 'POST':
                return await self.session.post(url, data=data)
            else:
                return await self.session.get(url)
        else:
            raise Exception("Session not initialized. Please login first.")

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
