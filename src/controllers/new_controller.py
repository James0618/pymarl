from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class NewMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # TODO: 2-step rollout
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs, next_hidden_state, _, _, pred_reward = self.get_agent_outs(ep_batch, t_ep)
        agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, -1)

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        self.update_hidden_states(chosen_actions, next_hidden_state, ep_batch.batch_size)

        return chosen_actions

    def get_agent_outs(self, ep_batch, t):
        batch_size = ep_batch.batch_size

        actions = th.eye(self.args.n_actions).expand(batch_size * self.n_agents, self.args.n_actions, -1).cuda()

        # self.hidden_state: Tensor[bs * n_agents, 64]
        hidden_state = self.hidden_states.expand(self.args.n_actions, batch_size * self.n_agents, -1).transpose(0, 1)

        agent_inputs = self._build_inputs(ep_batch, t).expand(self.args.n_actions,
                                                              batch_size * self.n_agents, -1).transpose(0, 1)

        next_hidden_state, states, next_states, pred_reward, pred_value = self.forward(
            agent_inputs=agent_inputs, hidden_state=hidden_state, actions=actions)

        agent_outputs = (pred_reward + pred_value * self.args.gamma)

        return agent_outputs, next_hidden_state, states[:, 0], next_states, pred_reward

    def update_hidden_states(self, chosen_action, next_hidden_state, batch_size):
        chosen_action = chosen_action.contiguous().view(batch_size * self.n_agents, 1).cpu()
        index = th.arange(batch_size * self.n_agents).unsqueeze(-1)
        index = th.cat((index, chosen_action), dim=-1).t()

        self.hidden_states = next_hidden_state[index.tolist()]

    def forward(self, agent_inputs, hidden_state, actions):
        # Rollout part
        states, next_hidden_state = self.state_estimation.forward(hidden_state=hidden_state, observation=agent_inputs)
        opponent_actions = self.opponent_model.forward(states)

        next_states, pred_reward = self.transition_model.forward(states, actions, opponent_actions)
        pred_value = self.value_estimation(next_states)

        return next_hidden_state, states, next_states, pred_reward, pred_value

    def init_hidden(self, batch_size):
        hidden_states = self.state_estimation.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.hidden_states = hidden_states.view(batch_size * self.n_agents, -1)

    def parameters(self):
        # return self.agent.parameters()
        return list(self.transition_model.parameters()) + list(self.opponent_model.parameters()) + \
               list(self.state_estimation.parameters()) + list(self.value_estimation.parameters())

    def transition_parameters(self):
        return list(self.transition_model.parameters()) + list(self.opponent_model.parameters()) + \
               list(self.state_estimation.parameters())

    def value_parameters(self):
        return list(self.value_estimation.parameters())

    def load_state(self, other_mac):
        self.transition_model.load_state_dict(other_mac.transition_model.state_dict())
        self.opponent_model.load_state_dict(other_mac.opponent_model.state_dict())
        self.state_estimation.load_state_dict(other_mac.state_estimation.state_dict())
        self.value_estimation.load_state_dict(other_mac.value_estimation.state_dict())

    def cuda(self):
        self.transition_model.cuda()
        self.opponent_model.cuda()
        self.state_estimation.cuda()
        self.value_estimation.cuda()

    def save_models(self, path):
        th.save(self.transition_model.state_dict(), "{}/transition_model.th".format(path))
        th.save(self.opponent_model.state_dict(), "{}/opponent_model.th".format(path))
        th.save(self.state_estimation.state_dict(), "{}/state_estimation.th".format(path))
        th.save(self.value_estimation.state_dict(), "{}/value_estimation.th".format(path))

    def load_models(self, path):
        self.transition_model.load_state_dict(th.load("{}/transition_model.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.opponent_model.load_state_dict(th.load("{}/opponent_model.th".format(path),
                                                    map_location=lambda storage, loc: storage))
        self.state_estimation.load_state_dict(th.load("{}/state_estimation.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.value_estimation.load_state_dict(th.load("{}/value_estimation.th".format(path),
                                                      map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        # build the agent, transition model, and opponent model
        StateEstimation, ValueEstimation, TransitionModel, OpponentModel = agent_REGISTRY[self.args.agent]
        self.transition_model = TransitionModel(self.args)
        self.opponent_model = OpponentModel(self.args)
        self.state_estimation = StateEstimation(input_shape, self.args)
        self.value_estimation = ValueEstimation(self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
