import dataclasses
import os
import time
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

from ... import Code


#@dataclasses.dataclass
#class LocalUser:
#    name:str
#    uid:str

class Session(metaclass=Singleton):

    # user: LocalUser

    def __init__(self, username):
        self.username = username
        self.session: Optional[ClientSession] = None
        self.base = os.environ.get("TOOLBOXV2_REMOTE_BASE")

    async def init_log_in_mk_link(self, mak_link):
        from playwright.async_api import async_playwright
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=False)  # Set headless=False if you want to see the browser UI
            context = await browser.new_context()

            # Open a new page
            page = await context.new_page()

            # Navigate to a URL that sets something in localStorage
            if mak_link.startswith(self.base):
                mak_link = mak_link.replace(self.base, "")
            await page.goto(f"{self.base}/{mak_link}")  # Replace with the actual URL that uses localStorage
            # Retrieve data from localStorage
            time.sleep(3)
            started = await page.evaluate("localStorage.getItem('StartMLogIN')")
            if started is None:
                get_logger().error("Could not found the startMLogIN flag")
                await browser.close()
                return False
            await page.wait_for_url(url=f"{self.base}/web/dashboard", wait_until="commit", timeout=60 * 2.5 * 1000)
            claim = await page.evaluate("localStorage.getItem('jwt_claim_device')")
            if claim is None:
                get_logger().error("No claim Received")
                await browser.close()
                return False
            with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="w") as blob:
                blob.clear()
                blob.write(claim.encode())

            # Do something with the data or perform further actions

            # Close browser
            await browser.close()

        return await self.login()

    async def login(self):
        self.session = ClientSession()
        with BlobFile(f"claim/{self.username}/jwt.c", key=Code.DK()(), mode="r") as blob:
            claim = blob.read()
            print("Claim:", claim)

        async with self.session.post(f"{self.base}/validateSession",
                                     data={'Jwt_claim': claim, 'Username': self.username}) as response:
            print(response.status)
            if response.status == 200:
                json_response = await response.json()
                print(json_response)
                get_logger().info("LogIn successful")
                return True
            get_logger().warning("LogIn failed")
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
            url = URL(url)
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


def test_session_invalid():
    import asyncio
    async def helper():
        s = Session('root')

        t = await s.init_log_in_mk_link("/")
        print(t)
        t1 = await s.login()
        print(t1)
        assert t1 == False

    asyncio.run(helper())
