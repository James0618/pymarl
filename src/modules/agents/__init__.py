REGISTRY = {}

from .rnn_agent import RNNAgent
from .new_agent import StateEstimation, TransitionModel, OpponentModel

REGISTRY["rnn"] = RNNAgent

REGISTRY["new"] = StateEstimation, TransitionModel, OpponentModel
