REGISTRY = {}

from .basic_controller import BasicMAC
from .new_controller import NewMAC

REGISTRY["basic_mac"] = BasicMAC

REGISTRY["new_mac"] = NewMAC
