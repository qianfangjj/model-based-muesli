from utils import get_network, get_optimizer, get_lr_scheduler, get_loss_functions, set_all_seeds
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import numpy as np
import datetime
import torch
import time
import pytz
import ray
import os


@ray.remote(num_gpus=1)
class Learner(Logger):

  def __init__(self, config, storage, replay_buffer, state=None):
    set_all_seeds(config.seed)

    self.run_tag = config.run_tag
    self.group_tag = config.group_tag
    self.worker_id = 'learner'
    self.replay_buffer = replay_buffer
    self.storage = storage
    self.config = deepcopy(config)

    if "learner" in self.config.use_gpu_for:
      if torch.cuda.is_available():
        if self.config.learner_gpu_device_id is not None:
          device_id = self.config.learner_gpu_device_id
          self.device = torch.device("cuda:{}".format(device_id))
        else:
          self.device = torch.device("cuda")
      else:
        raise RuntimeError("GPU was requested but torch.cuda.is_available() is False.")
    else:
      self.device = torch.device("cpu")
    
    self.network = get_network(config, self.device)
    self.network.to(self.device)
    self.network.train()

    initial_weights = self.network.get_weights()
    self.target_network = get_network(config, self.device)
    self.target_network.load_weights(initial_weights)
    self.target_network.to(self.device)


    self.optimizer = get_optimizer(config, self.network.parameters())
    self.lr_scheduler = get_lr_scheduler(config, self.optimizer)
    self.scalar_loss_fn, self.policy_loss_fn = get_loss_functions(config)

    self.training_step = 0
    self.losses_to_log = {'policy_gradient': 0., 'cmpo': 0.,'reward': 0., 'value': 0., 'policy': 0.}

    self.throughput = {'total_frames': 0, 'total_games': 0, 'training_step': 0, 'time': {'ups': 0, 'fps': 0}}

    if self.config.norm_obs:
      self.obs_min = np.array(self.config.obs_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.obs_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

    if state is not None:
      self.load_state(state)

    Logger.__init__(self)

  def load_state(self, state):
    self.run_tag = os.path.join(self.run_tag, 'resumed', '{}'.format(state['training_step']))
    self.network.load_state_dict(state['weights'])
    self.optimizer.load_state_dict(state['optimizer'])

    self.replay_buffer.add_initial_throughput.remote(state['total_frames'], state['total_games'])
    self.throughput['total_frames'] = state['total_frames']
    self.throughput['training_step'] = state['training_step']
    self.training_step = state['training_step'] 

  def save_state(self):
    actor_games = ray.get(self.storage.get_stats.remote('actor_games'))
    state = {'dirs': self.dirs,
             'config': self.config,
    		 		 'weights': self.network.get_weights(),
    		     'optimizer': self.optimizer.state_dict(),
             'training_step': self.training_step,
             'total_games': self.throughput['total_games'],
             'total_frames': self.throughput['total_frames'],
             'actor_games': actor_games}
    path = os.path.join(self.dirs['saves'], str(state['training_step']))
    torch.save(state, path)

  def send_weights(self):
    self.storage.store_weights.remote(self.network.get_weights(), self.training_step)

  def log_throughput(self):
    data = ray.get(self.replay_buffer.get_throughput.remote())

    self.throughput['total_games'] = data['games']
    self.log_scalar(tag='games/finished', value=data['games'], i=self.training_step)

    new_frames = data['frames'] - self.throughput['total_frames']
    if new_frames > self.config.frames_before_fps_log:

      current_time = time.time()
      new_updates = self.training_step - self.throughput['training_step']
      ups = new_updates / (current_time - self.throughput['time']['ups'])
      fps = new_frames / (current_time - self.throughput['time']['fps'])
      replay_ratio = ups / fps
      sample_ratio = self.config.batch_size * replay_ratio

      self.throughput['total_frames'] = data['frames']
      self.throughput['training_step'] = self.training_step
      self.throughput['time']['ups'] = current_time
      self.throughput['time']['fps'] = current_time

      self.log_scalar(tag='throughput/frames_per_second', value=fps, i=self.training_step)
      self.log_scalar(tag='throughput/updates_per_second', value=ups, i=self.training_step)
      self.log_scalar(tag='throughput/replay_ratio', value=replay_ratio, i=self.training_step)
      self.log_scalar(tag='throughput/sample_ratio', value=sample_ratio, i=self.training_step)
      self.log_scalar(tag='throughput/total_frames', value=data['frames'], i=self.training_step)

  def learn(self):
    self.send_weights()

    self.throughput['time']['fps'] = time.time() 
    while ray.get(self.replay_buffer.size.remote()) < self.config.stored_before_train:
      time.sleep(1)
      # print(f"\n buffer_size < 500")

    self.throughput['time']['ups'] = time.time() 
    while self.training_step < self.config.training_steps:
      not_ready_batches = [self.replay_buffer.sample_batch.remote() for _ in range(self.config.batches_per_fetch)]
      while len(not_ready_batches) > 0:
        ready_batches, not_ready_batches = ray.wait(not_ready_batches, num_returns=1)

        batch = ray.get(ready_batches[0])
        # self.update_weights(batch)
        self.update_weights_muesli(batch)
        self.training_step += 1
        # soft update target network params
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
          target_param.data.copy_(
            target_param.data * (1.0 - self.config.alpha_target) + param.data * self.config.alpha_target
          )

        if self.training_step % self.config.send_weights_frequency == 0:
          self.send_weights()

        if self.training_step % self.config.save_state_frequency == 0:
          self.save_state()

        if self.training_step % self.config.learner_log_frequency == 0:
          policy_gradient_loss = self.losses_to_log['policy_gradient'] / self.config.learner_log_frequency
          cmpo_loss = self.losses_to_log['cmpo'] / self.config.learner_log_frequency
          reward_loss = self.losses_to_log['reward'] / self.config.learner_log_frequency
          value_loss = self.losses_to_log['value'] / self.config.learner_log_frequency
          policy_loss = self.losses_to_log['policy'] / self.config.learner_log_frequency

          self.losses_to_log['policy_gradient'] = 0
          self.losses_to_log['cmpo'] = 0
          self.losses_to_log['reward'] = 0
          self.losses_to_log['value'] = 0
          self.losses_to_log['policy'] = 0

          self.log_scalar(tag='loss/policy_gradient', value=policy_gradient_loss, i=self.training_step)
          self.log_scalar(tag='loss/cmpo', value=cmpo_loss, i=self.training_step)
          self.log_scalar(tag='loss/reward', value=reward_loss, i=self.training_step)
          self.log_scalar(tag='loss/value', value=value_loss, i=self.training_step)
          self.log_scalar(tag='loss/policy', value=policy_loss, i=self.training_step)
          self.log_throughput()

          if self.lr_scheduler is not None:
            self.log_scalar(tag='loss/learning_rate', value=self.lr_scheduler.lr, i=self.training_step)

          if self.config.debug:
            total_grad_norm = 0
            for name, weights in self.network.named_parameters():
              self.log_histogram(weights.grad.data.cpu().numpy(), 'gradients' + '/' + name + '_grad', self.training_step)
              self.log_histogram(weights.data.cpu().numpy(), 'network_weights' + '/' + name, self.training_step)
              total_grad_norm += weights.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            self.log_scalar(tag='total_gradient_norm', value=total_grad_norm, i=self.training_step)

  def update_weights(self, batch):
    batch, idxs, is_weights = batch
    observations, actions, targets = batch

    target_rewards, target_values, target_policies = targets

    if self.config.norm_obs:
      observations = (observations - self.obs_min) / self.obs_range
    observations = torch.from_numpy(observations).to(self.device)

    value, reward, policy_logits, hidden_state = self.network.initial_inference(observations)

    with torch.no_grad():
      target_policies = torch.from_numpy(target_policies).to(self.device)
      target_values = torch.from_numpy(target_values).to(self.device)
      target_rewards = torch.from_numpy(target_rewards).to(self.device)
      is_weights = torch.from_numpy(is_weights).to(self.device)

      init_value = self.config.inverse_value_transform(value) if not self.config.no_support else value
      new_errors = (init_value.squeeze() - target_values[:, 0]).cpu().numpy()
      self.replay_buffer.update.remote(idxs, new_errors)

      if not self.config.no_target_transform:
        target_values = self.config.scalar_transform(target_values)
        target_rewards = self.config.scalar_transform(target_rewards)

      # scalar to support
      if not self.config.no_support:
        target_values = self.config.value_phi(target_values)
        target_rewards = self.config.reward_phi(target_rewards)

    reward_loss = 0
    value_loss = self.scalar_loss_fn(value.squeeze(), target_values[:, 0])
    policy_loss = self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, 0])

    for i, action in enumerate(zip(*actions), 1):
      value, reward, policy_logits, hidden_state = self.network.recurrent_inference(hidden_state, action)
      hidden_state.register_hook(lambda grad: grad * 0.5)

      reward_loss += self.scalar_loss_fn(reward.squeeze(), target_rewards[:, i])

      value_loss += self.scalar_loss_fn(value.squeeze(), target_values[:, i])
      
      policy_loss += self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, i])

    reward_loss = (is_weights * reward_loss).mean()
    value_loss = (is_weights * value_loss).mean()
    policy_loss = (is_weights * policy_loss).mean()

    full_weighted_loss = reward_loss + value_loss + policy_loss

    full_weighted_loss.register_hook(lambda grad: grad * (1/self.config.num_unroll_steps))

    self.optimizer.zero_grad()

    full_weighted_loss.backward()

    if self.config.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.clip_grad)

    self.optimizer.step()

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    # self.losses_to_log['policy_gradient'] += policy_gradient_loss.detach().cpu().item()
    # self.losses_to_log['cmpo'] += cmpo_loss.detach().cpu().item()
    self.losses_to_log['reward'] += reward_loss.detach().cpu().item()
    self.losses_to_log['value'] += value_loss.detach().cpu().item()
    self.losses_to_log['policy'] += policy_loss.detach().cpu().item()

  def target_network_predictions(self, observation_batch, action_batch):
    # Target network preditions
    p_prior = []
    v_prior = []
    root_value, _, root_policy_logits, root_hidden_state = self.target_network.initial_inference(
      observation_batch[:, 0]
    )
    root_policy_logits = torch.softmax(root_policy_logits, dim=1)
    p_prior.append(root_policy_logits)
    if not self.config.no_support:
        root_value = self.config.inverse_value_transform(root_value)
    v_prior.append(root_value)
    # calculate q_prior use one-step look-ahead
    v1, r1, _, _ = self.target_network.recurrent_inference(
      root_hidden_state, action_batch[:, 0]
    )
    if not self.config.no_support:
        r1 = self.config.inverse_value_transform(r1)
        v1 = self.config.inverse_value_transform(v1)
    q_prior = r1 + self.config.discount * v1
    prior_adv = q_prior - root_value

    # unroll K step
    for i in range(1, action_batch.shape[1] + 1):
      value_, _, policy_logits_, rp = self.target_network.initial_inference(
        observation_batch[:, i]
      )
      if not self.config.no_support:
          value_ = self.config.inverse_value_transform(value_)
      # compute adv for each action to construct p_cmpo
      exp_adv_list = []
      z_cmpo = 0
      for a in range(self.config.action_space):
        # calculate adv for every action
        action = torch.full(action_batch[:, 0].size(), a).to(self.device)
        qs = 0
        for t in range(self.config.simulation_depth):
          v, r, p, rp = self.target_network.recurrent_inference(
            rp, action
          )
          p = torch.softmax(p, dim=1)
          if not self.config.no_support:
              r = self.config.inverse_value_transform(r)
              v = self.config.inverse_value_transform(v)
          q = r + self.config.discount * v
          qs += q
          action = torch.multinomial(p, 1, replacement=False)
        qs = qs / self.config.simulation_depth
        exp_adv = torch.exp(torch.clamp(qs - value_, -self.config.threshold_c, self.config.threshold_c))
        z_cmpo += exp_adv
        exp_adv_list.append(exp_adv)
      z_cmpo = z_cmpo / self.config.action_space
      exp_advs = torch.cat(exp_adv_list, dim=1)
      policy_cmpo = policy_logits_ * exp_advs / z_cmpo
      policy_cmpo = torch.softmax(policy_cmpo, dim=1)
      p_prior.append(policy_cmpo)
      # policy_logits_ = torch.softmax(policy_logits_, dim=1)
      # p_prior.append(policy_logits_)
      v_prior.append(value_)

    return prior_adv, p_prior, v_prior

  def cal_prob_fuc(self, logits):
    logits_sub_max = torch.clamp(logits - torch.max(logits, dim=1, keepdim=True)[0], -10.0 ** 20.0, 1)
    exp_logits = torch.exp(logits_sub_max) + 0.0000001
    exp_logits_sum = torch.sum(exp_logits)
    softmax_logits = exp_logits / exp_logits_sum
    return softmax_logits

  def policy_gradient_loss(self, policy_logits, old_logits, action, advantages):
    new_logps = self.cal_prob_fuc(policy_logits)
    new_logps = torch.log(new_logps.gather(1, action))
    old_logps = torch.log(old_logits.gather(1, action))
    ratio = torch.exp(new_logps - old_logps)
    clip_ratio = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
    pg_loss = (-torch.min(ratio * advantages, clip_ratio * advantages)).sum(1)
    return pg_loss

  def cmpo_loss(self, policy_logits, observation):
    # use target policy to sample action
    root_value, _, root_policy_logits, root_hidden_state = self.target_network.initial_inference(
      observation
    )
    root_policy_logits = torch.softmax(root_policy_logits, dim=1)
    if not self.config.no_support:
        root_value = self.config.inverse_value_transform(root_value)
    # calculate cmpo loss
    actions, exadvs = [], []
    exadv_sum = 0
    for i in range(self.config.num_sampled_actions):
      action = torch.multinomial(root_policy_logits, 1, replacement=False)
      actions.append(action)
      rp = root_hidden_state
      qs = 0
      for t in range(self.config.simulation_depth):
        v, r, p, rp = self.target_network.recurrent_inference(
          rp, action
        )
        p = torch.softmax(p, dim=1)
        if not self.config.no_support:
            r = self.config.inverse_value_transform(r)
            v = self.config.inverse_value_transform(v)
        q = r + self.config.discount * v
        qs += q
        action = torch.multinomial(p, 1, replacement=False)
      qm = qs / self.config.simulation_depth
      exp_adv = torch.exp(torch.clamp(qm - root_value, -self.config.threshold_c, self.config.threshold_c))
      exadv_sum += exp_adv
      exadvs.append(exp_adv)
    zs = []
    for exadv in exadvs:
      z = (1 + exadv_sum - exadv) / self.config.num_sampled_actions
      zs.append(z)
    # # cmpo loss
    cmpo_loss = 0
    logits_s = torch.softmax(policy_logits, dim=1)
    for i in range(self.config.num_sampled_actions):
      cmpo_loss += -(exadvs[i] / zs[i] * torch.log(logits_s.gather(1, actions[i])))
    cmpo_loss = cmpo_loss / self.config.num_sampled_actions
    return cmpo_loss


  def update_weights_muesli(self, batch):
    batch, idxs, is_weights = batch
    observations, actions, targets = batch

    target_rewards, target_values, target_policies = targets

    if self.config.norm_obs:
      observations = (observations - self.obs_min) / self.obs_range
    observations = torch.from_numpy(observations).to(self.device)

    value, reward, policy_logits, hidden_state = self.network.initial_inference(observations[:, 0])
    with torch.no_grad():
      target_policies = torch.from_numpy(target_policies).to(self.device)
      target_values = torch.from_numpy(target_values).to(self.device)
      target_rewards = torch.from_numpy(target_rewards).to(self.device)
      is_weights = torch.from_numpy(is_weights).to(self.device)
      action_batch = torch.tensor(actions).long().to(self.device).unsqueeze(-1)
      # print(f"\n action_batch: {action_batch.size()}")
      # print(f"\n action_batch0: {action_batch[:, 0]}")

      init_value = self.config.inverse_value_transform(value) if not self.config.no_support else value
      new_errors = (init_value.squeeze() - target_values[:, 0]).cpu().numpy()
      self.replay_buffer.update.remote(idxs, new_errors)

      if not self.config.no_target_transform:
        target_values = self.config.scalar_transform(target_values)
        target_rewards = self.config.scalar_transform(target_rewards)

      # scalar to support
      if not self.config.no_support:
        target_values = self.config.value_phi(target_values)
        target_rewards = self.config.reward_phi(target_rewards)

    prior_adv, prior_logits, prior_values = self.target_network_predictions(observations, action_batch)

    # init_value = self.config.inverse_value_transform(value) if not self.config.no_support else value
    # new_errors = (init_value.squeeze() - target_values[:, 0]).cpu().numpy()
    # self.replay_buffer.update.remote(idxs, new_errors)

    reward_loss = 0
    value_loss = self.scalar_loss_fn(value.squeeze(), target_values[:, 0])
    # policy_loss = self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, 0])
    policy_gradient_loss = self.policy_gradient_loss(policy_logits, target_policies[:, 0], action_batch[:, 0], prior_adv)
    cmpo_loss = self.cmpo_loss(policy_logits, observations[:, 0])

    policy_loss = 0
    # for i, action in enumerate(zip(*actions), 1):
    for i in range(1, action_batch.shape[1]):
      value, reward, policy_logits, hidden_state = self.network.recurrent_inference(hidden_state, action_batch[:, i-1])
      hidden_state.register_hook(lambda grad: grad * 0.5)

      reward_loss += self.scalar_loss_fn(reward.squeeze(), target_rewards[:, i])

      value_loss += self.scalar_loss_fn(value.squeeze(), target_values[:, i])

      # policy_loss += self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, i])
      policy_loss += self.policy_loss_fn(policy_logits.squeeze(), prior_logits[i])

    reward_loss = (is_weights * reward_loss).mean()
    value_loss = (is_weights * value_loss).mean()
    policy_loss = (is_weights * policy_loss).mean()
    policy_gradient_loss = (is_weights * policy_gradient_loss).mean()
    cmpo_loss = (is_weights * cmpo_loss).mean()

    full_weighted_loss = policy_gradient_loss + cmpo_loss + reward_loss + value_loss + policy_loss

    full_weighted_loss.register_hook(lambda grad: grad * (1 / self.config.num_unroll_steps))

    self.optimizer.zero_grad()

    full_weighted_loss.backward()

    if self.config.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.clip_grad)

    self.optimizer.step()

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    self.losses_to_log['policy_gradient'] += policy_gradient_loss.detach().cpu().item()
    self.losses_to_log['cmpo'] += cmpo_loss.detach().cpu().item()
    self.losses_to_log['reward'] += reward_loss.detach().cpu().item()
    self.losses_to_log['value'] += value_loss.detach().cpu().item()
    self.losses_to_log['policy'] += policy_loss.detach().cpu().item()

  def launch(self):
    print("Learner is online on {}.".format(self.device))
    self.learn()

