import hashlib
import json
from typing import Mapping

import numpy as np


def named_rng(seed: int, concern: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{concern}".encode()).digest()
    child_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(child_seed)


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
