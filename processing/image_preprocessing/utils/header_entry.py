from dataclasses import dataclass
from typing import Callable


@dataclass
class HeaderEntry:
    tag : str
    conversion: Callable
