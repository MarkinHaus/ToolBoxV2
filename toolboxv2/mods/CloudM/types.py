import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from toolboxv2 import Code


@dataclass
class UserPersonaPubKey:
    public_key: bytes
    sign_count: int
    credential_id: bytes
    rawId: str
    attestation_object: bytes


@dataclass
class User:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    pub_key: str = field(default="")
    email: str = field(default="")
    name: str = field(default="")
    user_pass_pub: str = field(default="")
    user_pass_pub_persona: Dict[str, str or bytes] = field(default_factory=lambda: ({}))
    user_pass_pub_devices: List[str] = field(default_factory=lambda: ([]))
    user_pass_pri: str = field(default="")
    user_pass_sync: str = field(default="")
    creation_time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d::%H:%M:%S", time.localtime()))
    challenge: str = field(default="")
    is_persona: bool = field(default=False)
    level: int = field(default=0)


@dataclass
class UserCreator(User):
    def __post_init__(self):
        self.user_pass_pub, self.user_pass_pri = Code.generate_asymmetric_keys()
        self.user_pass_sync = Code.generate_symmetric_key()
        self.challenge = Code.encrypt_asymmetric(str(uuid.uuid4()), self.user_pass_pub)
