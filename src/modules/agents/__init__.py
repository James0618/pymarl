REGISTRY = {}

from .rnn_agent import RNNAgent
from .new_agent import StateEstimation, ValueEstimation, TransitionModel, OpponentModel

REGISTRY["rnn"] = RNNAgent

REGISTRY["new"] = StateEstimation, ValueEstimation, TransitionModel, OpponentModel
