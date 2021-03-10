import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop


class NewLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.transition_params, self.value_params = list(mac.transition_parameters()), list(mac.value_parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.transition_optimiser = RMSprop(params=self.transition_params, lr=5*args.lr, alpha=args.optim_alpha,
                                            eps=args.optim_eps)
        self.value_optimiser = RMSprop(params=self.value_params, lr=args.lr, alpha=args.optim_alpha,
                                       eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mode='transition'):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        cur_q_out, mac_out, cur_s, next_s, pred_rs = [], [], [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, next_hidden_state, state, next_state, pred_reward = self.mac.get_agent_outs(batch, t=t)
            cur_s.append(state.view(batch.batch_size, self.args.n_agents, -1))

            if t == batch.max_seq_length - 1:
                mac_out.append(agent_outs.view(batch.batch_size, self.args.n_agents, -1))

            else:
                chosen_action = actions[:, t].contiguous().view(batch.batch_size * self.args.n_agents, 1).cpu()
                index = th.arange(batch.batch_size * self.args.n_agents).unsqueeze(-1)
                index = th.cat((index, chosen_action), dim=-1).t()

                self.mac.update_hidden_states(chosen_action, next_hidden_state, batch.batch_size)
                # Pick the Q-Values for the actions taken by each agent
                cur_q_out.append(agent_outs[index.tolist()].view(batch.batch_size, self.args.n_agents))
                next_s.append(next_state[index.tolist()].view(batch.batch_size, self.args.n_agents, -1))
                pred_rs.append(pred_reward[index.tolist()].view(batch.batch_size, self.args.n_agents))

                mac_out.append(agent_outs.view(batch.batch_size, self.args.n_agents, -1))

        # Concat over time
        cur_q_out, next_s, pred_rs = th.stack(cur_q_out, dim=1), th.stack(next_s, dim=1), th.stack(pred_rs, dim=1)
        mac_out, cur_s = th.stack(mac_out, dim=1), th.stack(cur_s[:-1], dim=1)

        pred_sum_rs = th.sum(pred_rs, dim=2, keepdim=True)              # sum the prediction of rewards for all agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            target_agent_outs, _, _, _, _ = self.target_mac.get_agent_outs(batch, t=t)
            target_mac_out.append(target_agent_outs.view(batch.batch_size, self.args.n_agents, -1))

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals = th.sum(cur_q_out, dim=2, keepdim=True)
        target_max_qvals = th.sum(target_max_qvals, dim=2, keepdim=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        transition_loss = ((next_s - cur_s.detach()) ** 2).sum(-1).sum(-1) / (self.args.n_agents * cur_s.shape[-1])
        masked_transition_loss = (transition_loss.unsqueeze(-1) * mask).sum() / mask.sum()

        pred_sum_rs_loss = (((rewards - pred_sum_rs) * mask) ** 2).sum() / mask.sum()

        q_loss = (masked_td_error ** 2).sum() / mask.sum()

        index = rewards.nonzero()
        if index.shape[0] != 0:
            index = index.t()
            nonzero_rewards = rewards[index[0], index[1]]
            pred_rs_nz = pred_sum_rs[index[0], index[1]]
            nonzero_loss = ((nonzero_rewards - pred_rs_nz) ** 2).sum() / index.shape[1]

            # Normal L2 loss, take mean over actual data
            loss = q_loss + masked_transition_loss + pred_sum_rs_loss + nonzero_loss

        else:
            nonzero_loss = 0
            # Normal L2 loss, take mean over actual data
            loss = q_loss + masked_transition_loss + pred_sum_rs_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("transition_loss", masked_transition_loss.item(), t_env)
            self.logger.log_stat("nonzero_loss", float(nonzero_loss), t_env)
            self.logger.log_stat("reward_loss", pred_sum_rs_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
