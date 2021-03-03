import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class NewLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        test = rewards.cpu().numpy()
        # Calculate estimated Q-Values
        mac_out, pred_obs, pred_rs = [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            chosen_action = actions[:, t].contiguous().view(batch.batch_size * self.args.n_agents, 1).cpu()
            index = th.arange(batch.batch_size * self.args.n_agents).unsqueeze(-1)
            index = th.cat((index, chosen_action), dim=-1).t()

            agent_outs, next_hidden_state, pred_observation, pred_reward = self.mac.get_agent_outs(batch, t=t)
            self.mac.update_hidden_states(chosen_action, next_hidden_state, batch.batch_size)

            # Pick the Q-Values for the actions taken by each agent
            mac_out.append(agent_outs[index.tolist()].view(batch.batch_size, self.args.n_agents))
            pred_obs.append(pred_observation[index.tolist()].view(batch.batch_size, self.args.n_agents, -1))
            pred_rs.append(pred_reward[index.tolist()].view(batch.batch_size, self.args.n_agents))

        # Concat over time
        mac_out, pred_obs, pred_rs = th.stack(mac_out, dim=1), th.stack(pred_obs, dim=1), th.stack(pred_rs, dim=1)
        pred_sum_rs = th.sum(pred_rs, dim=2, keepdim=True)              # sum the prediction of rewards for all agents

        target_observation = []
        for t in range(1, batch.max_seq_length):
            temp = self.mac._build_inputs(batch, t)
            target_observation.append(temp.view(batch.batch_size, self.args.n_agents, -1))

        target_obs = th.stack(target_observation, dim=1).cuda()
        target_rs = rewards.cuda()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(1, batch.max_seq_length):
            target_agent_outs, _, _, _ = self.target_mac.get_agent_outs(batch, t=t)
            target_mac_out.append(target_agent_outs.view(batch.batch_size, self.args.n_agents, -1))

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals = th.sum(mac_out, dim=2, keepdim=True)
        target_max_qvals = th.sum(target_max_qvals, dim=2, keepdim=True)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        pred_obs_loss = ((target_obs - pred_obs) ** 2).sum(-1).sum(-1) / (self.args.n_agents * pred_obs.shape[-1])
        masked_obs_loss = (pred_obs_loss.unsqueeze(-1) * mask).sum() / mask.sum()

        pred_sum_rs_loss = (((rewards - pred_sum_rs) * mask) ** 2).sum() / mask.sum()

        q_loss = (masked_td_error ** 2).sum() / mask.sum()

        # Normal L2 loss, take mean over actual data
        loss = q_loss + masked_obs_loss, pred_sum_rs_loss

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
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
