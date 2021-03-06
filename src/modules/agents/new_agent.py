import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionModel(nn.Module):
    def __init__(self, observation_shape, args):
        """
        first, take the hidden state as the agents' state,
        and, the rnn output the prediction of observations.
        """
        super(TransitionModel, self).__init__()
        self.args = args

        # observation here contain "observation" of agents and their actions
        self.from_observations = nn.Linear(observation_shape, args.rnn_hidden_dim)
        self.from_actions = nn.Linear(args.n_actions + args.latent_action_shape, args.rnn_hidden_dim)
        self.features = nn.Linear(2 * args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.transition = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # predict the "observation" of agents and state value
        self.pred_observation = nn.Linear(args.rnn_hidden_dim, observation_shape)
        self.pred_reward = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, observations, hidden_state, actions):
        from_observations = F.relu(self.from_observations(observations)).reshape(-1, self.args.rnn_hidden_dim)
        from_actions = F.relu(self.from_actions(actions)).reshape(-1, self.args.rnn_hidden_dim)

        observations_and_actions = torch.cat((from_observations, from_actions), dim=-1)
        features = F.relu(self.features(observations_and_actions)).reshape(-1, self.args.rnn_hidden_dim)
        hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        next_hidden_state = self.transition(features, hidden_state).reshape(-1, self.args.n_actions,
                                                                            self.args.rnn_hidden_dim)
        pred_reward = self.pred_reward(next_hidden_state)
        pred_observation = self.pred_observation(next_hidden_state)

        return pred_observation, pred_reward, next_hidden_state

    def init_hidden(self):
        # make hidden states on same device as model
        return self.from_observations.weight.new(1, self.args.rnn_hidden_dim).zero_()


class OpponentModel(nn.Module):
    def __init__(self, args):
        """
        take the state generated by transition model as opponents' converted state,
        and transfer the opponents' actions
        """
        super(OpponentModel, self).__init__()
        self.args = args

        self.opponent = nn.Sequential(
            nn.Linear(args.state_shape, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.latent_action_shape),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        # TODO: Add GMM to the opponent model!
        latent_actions = self.opponent(state)

        return latent_actions


class StateEstimation(nn.Module):
    def __init__(self, observation_shape, args):
        super(StateEstimation, self).__init__()
        self.args = args

        self.estimate_state = nn.Linear(args.rnn_hidden_dim + observation_shape, args.state_shape)
        self.estimate_value = nn.Linear(args.state_shape, 1)

    def forward(self, hidden_state, observation):
        # hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        inputs = torch.cat((hidden_state, observation), dim=-1)

        state = F.relu(self.estimate_state(inputs))
        state_value = self.estimate_value(state)

        return state, state_value
