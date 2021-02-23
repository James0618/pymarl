import torch.nn as nn
import torch.nn.functional as F


class NewAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NewAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class TransitionModel(nn.Module):
    def __init__(self, observation_shape, args):
        """
        first, set the hidden state as the initial observation,
        then, take the hidden state as the agents' state,
        and, the rnn output the prediction of observations.
        """
        super(TransitionModel, self).__init__()
        self.args = args

        # observation here contain "observation" of agents and their actions
        self.from_actions = nn.Linear(observation_shape, args.rnn_hidden_dim)
        self.transition = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # predict the "observation" of agents
        self.pred_observation = nn.Linear(args.rnn_hidden_dim, observation_shape - args.n_agents - args.n_actions)

    def forward(self, observations, last_state):
        features = F.relu(self.from_actions(observations))
        last_state = last_state.reshape(-1, self.args.rnn_hidden_dim)
        current_state = self.transition(features, last_state)
        reconstruction = self.pred_observation(current_state)

        return reconstruction, current_state

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


class OpponentModel(nn.Module):
    def __init__(self, args):
        """
        take the state generated by transition model as opponents' converted state,
        and transfer the opponents' actions
        """
        super(OpponentModel, self).__init__()
        self.args = args

        self.opponent = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )

    def forward(self, state):
        # TODO: Add GMM to the opponent model!
        latent_actions = self.opponent(state)

        return latent_actions


class ConditionalPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super(ConditionalPolicy, self).__init__()

    def forward(self, state, opponent_actions):
        pass