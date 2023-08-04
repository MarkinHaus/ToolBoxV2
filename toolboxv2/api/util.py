from typing import List

from pydantic import BaseModel


class PostRequest(BaseModel):
    token: str
    data: dict
