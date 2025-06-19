from dataclasses import dataclass
from enum import Enum


@dataclass
class DatabaseModes(Enum):
    LC = "LOCAL_DICT"
    LR = "LOCAL_REDDIS"
    RR = "REMOTE_REDDIS"
    CB = "CLUSTER_BLOB"

    @classmethod
    def crate(cls, mode: str):
        if mode == "LC":
            return DatabaseModes.LC
        elif mode == "LR":
            return DatabaseModes.LR
        elif mode == "RR":
            return DatabaseModes.RR
        elif mode == "CB":
            return DatabaseModes.CB
        else:
            raise ValueError(f"{mode} != RR,LR,LC,CB")


@dataclass
class AuthenticationTypes(Enum):
    UserNamePassword = "password"
    Uri = "url"
    PassKey = "passkey"
    location = "location"
    none = "none"
