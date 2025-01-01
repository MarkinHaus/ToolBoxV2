from dataclasses import dataclass
import aiohttp
from typing import List, Dict, Union, Optional, AsyncIterator, BinaryIO, Iterator
import numpy as np
from pathlib import Path
import requests
import json


@dataclass
class SonarConfig:
    api_key: str
    base_url: str = "http://localhost:8080"
    batch_size: int = 32
    timeout: int = 30


class SonarAPIError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"API Error ({status}): {message}")


class ASonarClient:
    def __init__(self, config: SonarConfig):
        self.config = config
        self._token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_session(self):
        if not self._token:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/token",
                    json={"api_key": self.config.api_key},
                    timeout=self.config.timeout
                ) as resp:
                    if resp.status != 200:
                        raise SonarAPIError(resp.status, await resp.text())
                    data = await resp.json()
                    self._token = data["token"]

        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._token}"}
        )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _check_session(self):
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' or await init_session()")

    async def text_to_embeddings(
        self,
        texts: List[str],
        source_lang: str = "eng_Latn",
        stream: bool = False
    ) -> Union[np.ndarray, AsyncIterator[Dict]]:
        self._check_session()

        if stream:
            return self._stream_embeddings("/api/embeddings/text", {
                "texts": texts,
                "source_lang": source_lang,
                "stream": True,
                "batch_size": self.config.batch_size
            })

        async with self._session.post(
            f"{self.config.base_url}/api/embeddings/text",
            json={
                "texts": texts,
                "source_lang": source_lang,
                "batch_size": self.config.batch_size
            },
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return np.array(data["embeddings"])

    async def embeddings_to_text(
        self,
        embeddings: np.ndarray,
        target_lang: str = "eng_Latn",
        max_seq_len: int = 512
    ) -> List[str]:
        self._check_session()

        async with self._session.post(
            f"{self.config.base_url}/api/text/from_embeddings",
            json={
                "embeddings": embeddings.tolist(),
                "target_lang": target_lang,
                "max_seq_len": max_seq_len
            },
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return data["texts"]

    async def translate_text(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        self._check_session()

        async with self._session.post(
            f"{self.config.base_url}/api/translate/text",
            json={
                "texts": texts,
                "source_lang": source_lang,
                "target_lang": target_lang
            },
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return data["translations"]

    async def speech_to_embeddings(
        self,
        audio_files: List[Union[str, Path, BinaryIO]],
        stream: bool = False
    ) -> Union[np.ndarray, AsyncIterator[Dict]]:
        self._check_session()

        data = aiohttp.FormData()
        for i, file in enumerate(audio_files):
            if isinstance(file, (str, Path)):
                data.add_field(f"file{i}", open(file, "rb"))
            else:
                data.add_field(f"file{i}", file)

        data.add_field("stream", str(stream).lower())
        data.add_field("batch_size", str(self.config.batch_size))

        if stream:
            return self._stream_embeddings("/api/embeddings/speech", data)

        async with self._session.post(
            f"{self.config.base_url}/api/embeddings/speech",
            data=data,
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return np.array(data["embeddings"])

    async def speech_to_text(
        self,
        audio_files: List[Union[str, Path, BinaryIO]],
        target_lang: str = "eng_Latn"
    ) -> List[str]:
        self._check_session()

        data = aiohttp.FormData()
        for i, file in enumerate(audio_files):
            if isinstance(file, (str, Path)):
                data.add_field(f"file{i}", open(file, "rb"))
            else:
                data.add_field(f"file{i}", file)

        data.add_field("target_lang", target_lang)

        async with self._session.post(
            f"{self.config.base_url}/api/speech/to_text",
            data=data,
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return data["texts"]

    async def compute_similarity(
        self,
        source_texts: List[str],
        translated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        source_lang: str = "eng_Latn",
        target_lang: str = "eng_Latn"
    ) -> np.ndarray:
        self._check_session()

        async with self._session.post(
            f"{self.config.base_url}/api/similarity/blaser",
            json={
                "source_texts": source_texts,
                "translated_texts": translated_texts,
                "reference_texts": reference_texts,
                "source_lang": source_lang,
                "target_lang": target_lang
            },
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())
            data = await resp.json()
            return np.array(data["scores"])

    async def _stream_embeddings(
        self,
        endpoint: str,
        payload: Union[Dict, aiohttp.FormData]
    ) -> AsyncIterator[Dict]:
        async with self._session.post(
            f"{self.config.base_url}{endpoint}",
            json=payload if isinstance(payload, dict) else None,
            data=payload if isinstance(payload, aiohttp.FormData) else None,
            timeout=self.config.timeout
        ) as resp:
            if resp.status != 200:
                raise SonarAPIError(resp.status, await resp.text())

            async for line in resp.content:
                if line:
                    yield json.loads(line)


class SonarClient:
    def __init__(self, config):
        self.config = config
        self._token: Optional[str] = None
        self._session: Optional[requests.Session] = None

    def __enter__(self):
        self.init_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_session(self):
        if not self._token:
            with requests.Session() as session:
                response = session.post(
                    f"{self.config.base_url}/api/token",
                    json={"api_key": self.config.api_key},
                    timeout=self.config.timeout
                )
                if response.status_code != 200:
                    raise SonarAPIError(response.status_code, response.text)
                data = response.json()
                self._token = data["token"]

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def _check_session(self):
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'with' or call init_session().")

    def text_to_embeddings(
        self,
        texts: List[str],
        source_lang: str = "eng_Latn",
        stream: bool = False
    ) -> Union[np.ndarray, Iterator[Dict]]:
        self._check_session()

        payload = {
            "texts": texts,
            "source_lang": source_lang,
            "stream": stream,
            "batch_size": self.config.batch_size
        }

        if stream:
            return self._stream_embeddings("/api/embeddings/text", payload)

        response = self._session.post(
            f"{self.config.base_url}/api/embeddings/text",
            json=payload,
            timeout=self.config.timeout
        )
        if response.status_code != 200:
            raise SonarAPIError(response.status_code, response.text)

        data = response.json()
        return np.array(data["embeddings"])

    def _stream_embeddings(
        self,
        endpoint: str,
        payload: Dict
    ) -> Iterator[Dict]:
        with self._session.post(
            f"{self.config.base_url}{endpoint}",
            json=payload,
            stream=True,
            timeout=self.config.timeout
        ) as response:
            if response.status_code != 200:
                raise SonarAPIError(response.status_code, response.text)

            for line in response.iter_lines():
                if line:
                    yield json.loads(line)

    def embeddings_to_text(
        self,
        embeddings: np.ndarray,
        target_lang: str = "eng_Latn",
        max_seq_len: int = 512
    ) -> List[str]:
        self._check_session()

        payload = {
            "embeddings": embeddings.tolist(),
            "target_lang": target_lang,
            "max_seq_len": max_seq_len
        }

        response = self._session.post(
            f"{self.config.base_url}/api/text/from_embeddings",
            json=payload,
            timeout=self.config.timeout
        )
        if response.status_code != 200:
            raise SonarAPIError(response.status_code, response.text)

        data = response.json()
        return data["texts"]

    def translate_text(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        self._check_session()

        payload = {
            "texts": texts,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

        response = self._session.post(
            f"{self.config.base_url}/api/translate/text",
            json=payload,
            timeout=self.config.timeout
        )
        if response.status_code != 200:
            raise SonarAPIError(response.status_code, response.text)

        data = response.json()
        return data["translations"]

    def speech_to_embeddings(
        self,
        audio_files: List[Union[str, Path, BinaryIO]],
        stream: bool = False
    ) -> Union[np.ndarray, Iterator[Dict]]:
        self._check_session()

        data = {}
        for i, file in enumerate(audio_files):
            if isinstance(file, (str, Path)):
                data[f"file{i}"] = open(file, "rb")
            else:
                data[f"file{i}"] = file

        data["stream"] = str(stream).lower()
        data["batch_size"] = str(self.config.batch_size)

        if stream:
            return self._stream_embeddings("/api/embeddings/speech", data)

        response = self._session.post(
            f"{self.config.base_url}/api/embeddings/speech",
            files=data,
            timeout=self.config.timeout
        )
        if response.status_code != 200:
            raise SonarAPIError(response.status_code, response.text)

        data = response.json()
        return np.array(data["embeddings"])

    def speech_to_text(
        self,
        audio_files: List[Union[str, Path, BinaryIO]],
        target_lang: str = "eng_Latn"
    ) -> List[str]:
        self._check_session()

        data = {}
        for i, file in enumerate(audio_files):
            if isinstance(file, (str, Path)):
                data[f"file{i}"] = open(file, "rb")
            else:
                data[f"file{i}"] = file

        data["target_lang"] = target_lang

        response = self._session.post(
            f"{self.config.base_url}/api/speech/to_text",
            files=data,
            timeout=self.config.timeout
        )
        if response.status_code != 200:
            raise SonarAPIError(response.status_code, response.text)

        data = response.json()
        return data["texts"]
