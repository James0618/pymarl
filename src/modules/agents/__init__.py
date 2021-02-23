REGISTRY = {}

from .rnn_agent import RNNAgent
from .new_agent import NewAgent

REGISTRY["rnn"] = RNNAgent

REGISTRY["new"] = NewAgent
