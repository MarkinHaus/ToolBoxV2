from dataclasses import dataclass
from enum import Enum


@dataclass
class DatabaseModes(Enum):
    LC = "LOCAL_DICT"
    RC = "REMOTE_DICT"
    LR = "LOCAL_REDDIS"
    RR = "REMOTE_REDDIS"


@dataclass
class AuthenticationTypes(Enum):
    UserNamePassword = "password"
    Uri = "url"
    PassKey = "passkey"
    location = "location"
    none = "none"
