from dataclasses import dataclass
from enum import Enum


@dataclass
class DatabaseModes(Enum):
    LC = "LOCAL_DICT"
    RC = "REMOTE_DICT"
    LR = "LOCAL_REDDIS"
    RR = "REMOTE_REDDIS"

    @classmethod
    def crate(cls, mode: str):
        if mode == "LC":
            return DatabaseModes.LC
        elif mode == "RC":
            return DatabaseModes.RC
        elif mode == "LR":
            return DatabaseModes.LR
        elif mode == "RR":
            return DatabaseModes.RR
        else:
            raise ValueError(f"{mode} != RR,LR,RC,LC")


@dataclass
class AuthenticationTypes(Enum):
    UserNamePassword = "password"
    Uri = "url"
    PassKey = "passkey"
    location = "location"
    none = "none"
