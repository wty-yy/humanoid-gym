from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs.custom.humanoid_env import XBotLFreeEnv
from humanoid.envs.custom.xbotl_paper_config import XBotLPaperCfg

from humanoid.utils.terrain import  HumanoidTerrain


class XBotLPaperEnv(XBotLFreeEnv):

  def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
    super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    self.last_feet_z = 0.

  def compute_observations(self):

    phase = self._get_phase()
    self.compute_ref_state()

    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

    stance_mask = self._get_gait_phase()
    contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

    self.command_input = torch.cat(
      (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
    
    q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
    dq = self.dof_vel * self.obs_scales.dof_vel
    
    self.privileged_obs_buf = torch.cat((
      self.command_input,  # 2 + 3
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
      stance_mask,  # 2
      contact_mask,  # 2
    ), dim=-1)

    obs_buf = torch.cat((
      self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
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

  def _get_foot_heigh_target(self, half_phase):
    ret = torch.zeros_like(half_phase)
    for a in reversed(self.cfg.rewards.foot_traj_poly_coefs):
      ret = ret * half_phase + a
    return ret
  
  def _get_foot_vel_target(self, half_phase):
    ret = torch.zeros_like(half_phase)
    poly_coefs: list = self.cfg.rewards.foot_traj_poly_coefs
    max_order = len(poly_coefs) - 1
    for i, a in enumerate(reversed(poly_coefs[1:])):
      ret = ret * half_phase + (max_order - i) * a
    return ret
  
  def _get_phase(self):
    """ Return phase ratio in [0, 1], period = cycle_time """
    cycle_time = self.cfg.rewards.cycle_time
    phase = (self.episode_length_buf * self.dt / cycle_time) % 1.0
    return phase
  
  def _get_stance_mask(self):
    """ Return stance_mask, shape=(N, 2) """
    phase = self._get_phase()
    stance_mask = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
    stance_mask[:, 0] = phase < 0.5
    stance_mask[:, 1] = phase >= 0.5
    return stance_mask

  def compute_ref_state(self):
    """ Compute left and right foot height and velocity along z axis
    This function will call once when `compute_observations`
    Returns:
      ref_feet_height: shape=(N, 2), target left and right foot height
      ref_feet_vel: shape=(N, 2), target left and right foot velocity
    """
    phase = self._get_phase()  # range=(0, 1.0)
    half_phase = phase % 0.5   # range=(0, 0.5)
    self.ref_feet_height = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
    self.ref_feet_vel = torch.zeros_like(self.ref_feet_height)
    foot_height_target = self._get_foot_heigh_target(half_phase)
    foot_vel_target = self._get_foot_vel_target(half_phase)
    masks = [phase > 0.5, phase <= 0.5]  # left and right foot mask
    for i, mask in enumerate(masks):
      self.ref_feet_height[mask, i] = foot_height_target[mask]
      self.ref_feet_vel[mask, i] = foot_vel_target[mask]
    return self.ref_feet_height, self.ref_feet_vel
  
########################### Reward ###################################
  def _reward_lin_velocity_tracking(self):
    """ Track linear velociy cammands """
    cmd = torch.cat([
      self.commands[:, :2],
      torch.zeros(self.num_envs, 1, dtype=self.commands.dtype, device=self.device)
    ], dim=1)  # (N, 3)
    rew = self._phi(cmd - self.base_lin_vel, 5.)
    return rew
  
  def _reward_ang_velocity_tracking(self):
    """ Track angular velociy cammands """
    cmd = torch.cat([
      torch.zeros(self.num_envs, 2, dtype=self.commands.dtype, device=self.device),
      self.commands[:, 2:3]
    ], dim=1)
    rew = self._phi(cmd - self.base_ang_vel, 7.)
    return rew
  
  def _reward_orientation_tracking(self):
    """ Track orientation (roll and pitch) """
    rew = self._phi(self.base_euler_xyz[:, :2], 5)
    return rew
  
  def _reward_base_height_tracking(self):
    """ Track base height to target base height """
    stance_mask = self._get_stance_mask()
    foot_z = torch.sum(  # Only one foot should be on ground
      self.rigid_state[:, self.feet_indices, 2] * stance_mask,
      dim=1
    )
    base_height = self.root_states[:, 2] - (foot_z - 0.05)
    rew = self._phi((self.cfg.rewards.base_height_target - base_height).unsqueeze(1), 10)
    return rew
  
  def _reward_periodic_force(self):
    """ Force reward for correct stance period """
    stance_mask = self._get_stance_mask()
    feet_forces = self.contact_forces[:, self.feet_indices, :]  # (N, 2, 3)
    feet_forces = torch.norm(feet_forces, dim=2)  # (N, 2)
    rew = torch.sum(stance_mask * feet_forces, dim=1)
    return rew
  
  def _reward_periodic_velocity(self):
    """ Velocity reward for correct swing period """
    swing_mask = ~self._get_stance_mask()
    feet_vel = self.rigid_state[:, self.feet_indices, 7:10]  # (N, 2, 3)
    feet_vel = torch.norm(feet_vel, dim=2)  # (N, 2)
    rew = torch.sum(swing_mask * feet_vel, dim=1)
    return rew

  def _reward_foot_height_tracking(self):
    """ Lift left and right foot to target height """
    # Compute feet contact mask
    contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    # Get the z-position of the feet and compute the change in z-axis
    feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
    delta_z = feet_z - self.last_feet_z
    self.feet_height += delta_z
    self.last_feet_z = feet_z
    # feet hight should be close to target height
    swing_mask = ~self._get_stance_mask()  # (N,2)
    rew = self._phi((self.ref_feet_height - self.feet_height) * swing_mask, 50.)
    half_phase = self._get_phase() % 0.5
    rew *= (0.1 < half_phase) & (half_phase < 0.4)
    # Reset accumulate feet height
    self.feet_height *= ~contact
    return rew
  
  def _reward_foot_vel_tracking(self):
    """ Lift left and right foot to target velocity """
    # Get the z-velocity of the feet
    feet_z_vel = self.rigid_state[:, self.feet_indices, 9]
    # feet hight should be close to target velocity
    swing_mask = ~self._get_stance_mask()  # (N,2)
    rew = self._phi((self.ref_feet_vel - feet_z_vel) * swing_mask, 5.)
    # print("[DEBUG]", "="*50)
    # half_phase = self._get_phase() % 0.5
    # print(f"foot_vel_tracking={rew}, delta={(self.ref_feet_vel - feet_z_vel) * swing_mask}, {self.ref_feet_vel=}, {feet_z_vel}, {half_phase=}")
    return rew

  def _reward_default_joint(self):
    """ Keep joint positions close to default joint positions """
    joint_diff = self.dof_pos - self.default_joint_pd_target
    rew = self._phi(joint_diff, 2.)
    return rew
  
  def _reward_energy_cost(self):
    """ Keep low energy for torque and joint velocity """
    torque = torch.norm(self.torques, p=1, dim=1)
    joint_vels = torch.norm(self.dof_vel, p=1, dim=1)
    rew = torque * joint_vels
    return rew
  
  def _reward_action_smoothness(self):
    """ Keep action be smothness with last and last last action """
    rew = torch.norm(self.actions + self.last_last_actions - 2 * self.last_actions, dim=1)
    return rew
  
  def _reward_feet_movements(self):
    """ Keep low z-velocity and z-acceleration for both feet"""
    rew = (
      torch.norm(self.rigid_state[:, self.feet_indices, 9], dim=1) +
      torch.norm((
        self.rigid_state[:, self.feet_indices, 9] -
        self.last_rigid_state[:, self.feet_indices, 9]
      ) / self.dt, dim=1)
    )
    return rew
  
  def _reward_large_contact(self):
    """ Punish large contact forces on feet"""
    rew = torch.clamp(
      torch.norm(
        self.contact_forces[:, self.feet_indices], dim=2
      ).sum(1) - 400,
      0, 100
    )
    return rew
