from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs.custom.kuavo42_legged_env import Kuavo42LeggedEnv

from humanoid.utils.terrain import  HumanoidTerrain


class Kuavo42LeggedSimpleRewardEnv(Kuavo42LeggedEnv):

  def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
    super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    self.last_feet_z = 0.

  def compute_observations(self):

    contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

    self.command_input = self.commands[:, :3] * self.commands_scale
    
    q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
    dq = self.dof_vel * self.obs_scales.dof_vel
    
    self.privileged_obs_buf = torch.cat((
      self.command_input,  # 3
      (self.dof_pos - self.default_joint_pd_target) * \
      self.obs_scales.dof_pos,  # 12
      self.dof_vel * self.obs_scales.dof_vel,  # 12
      self.actions,  # 12
      self.base_lin_vel * self.obs_scales.lin_vel,  # 3
      self.base_ang_vel * self.obs_scales.ang_vel,  # 3
      self.base_euler_xyz * self.obs_scales.quat,  # 3
      self.rand_push_force[:, :2],  # 2
      self.rand_push_torque,  # 3
      self.env_frictions,  # 1
      self.body_mass / 30.,  # 1
      contact_mask,  # 2
    ), dim=-1)

    obs_buf = torch.cat((
      self.command_input,  # 3 = 3D(vel_x, vel_y, aug_vel_yaw)
      q,  # 12D
      dq,  # 12D
      self.actions,   # 12D
      self.base_ang_vel * self.obs_scales.ang_vel,  # 3
      self.base_euler_xyz * self.obs_scales.quat,  # 3
    ), dim=-1)

    if self.cfg.terrain.measure_heights:
      heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
      self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
    
    if self.add_noise:  
      obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
    else:
      obs_now = obs_buf.clone()
    self.obs_history.append(obs_now)
    self.critic_history.append(self.privileged_obs_buf)


    obs_buf_all = torch.stack([self.obs_history[i]
                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

    self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
    self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

  def _phi(self, error, omega):
    """ Calculate exp error rewrad function
    Args:
      error: shape=(N, M)
      omega: scaler
    Returns:
      shape=(N,)
    """
    return torch.exp(-omega * torch.norm(error, dim=1))

########################### Reward ###################################
  def _reward_lin_velocity_tracking(self):
    """ Track linear velociy cammands """
    cmd = torch.cat([
      self.commands[:, :2],
      torch.zeros(self.num_envs, 1, dtype=self.commands.dtype, device=self.device)
    ], dim=1)  # (N, 3)
    rew = self._phi(cmd - self.base_lin_vel, 4)
    return rew
  
  def _reward_ang_velocity_tracking(self):
    """ Track angular velociy cammands """
    cmd = torch.cat([
      torch.zeros(self.num_envs, 2, dtype=self.commands.dtype, device=self.device),
      self.commands[:, 2:3]
    ], dim=1)
    rew = self._phi(cmd - self.base_ang_vel, 4)
    return rew
  
  def _reward_orientation(self):
    """ Track orientation (projected gravity) """
    rew = torch.norm(self.projected_gravity[:, :2], dim=1) ** 2
    return rew
  
  def _reward_energy(self):
    """ Keep low energy for torque and joint velocity """
    rew = torch.norm(self.torques * self.dof_vel, dim=1) ** 2
    return rew
  
  def _reward_base_height_tracking(self):
    """ Keep height """
    height = (
      self.root_states[:, 2] - torch.min(
        self.rigid_state[:, self.feet_indices, 2], dim=1
      )[0]
    ).unsqueeze(1)
    rew = self._phi(self.cfg.rewards.base_height_target - height, 10)
    return rew