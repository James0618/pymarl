REGISTRY = {}

from .rnn_agent import RNNAgent
from .new_agent import NewAgent, TransitionModel, OpponentModel

REGISTRY["rnn"] = RNNAgent

REGISTRY["new"] = NewAgent, TransitionModel, OpponentModel
