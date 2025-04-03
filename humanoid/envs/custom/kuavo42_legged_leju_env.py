import os
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import Kuavo42LeggedEnv

from humanoid.utils.terrain import  HumanoidTerrain


class Kuavo42LeggedLejuEnv(Kuavo42LeggedEnv):
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        phase[self.commands[:,4].to(bool)] = 0
        return phase

    def _get_gait_phase(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        stance_mask[self.commands[:, 4].to(bool)] = 1
        return stance_mask

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :4]), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 4
            q,  # 12
            dq,  # 12
            self.actions,  # 12
            diff,  # 12
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

        base_euler_xy = self.base_euler_xyz[:, :2]
        obs_buf = torch.cat((
            self.command_input,  # 6 = 2D(sin cos) + 4D(vel_x, vel_y, aug_vel_yaw, stance)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            base_euler_xy * self.obs_scales.quat,  # 2
            self.base_lin_acc * self.obs_scales.lin_acc  # 3
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

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if not len(env_ids): return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        num_standing_envs = int(self.cfg.commands.rel_standing_envs * len(env_ids))
        num_marking_envs = int(self.cfg.commands.rel_marking_envs * len(env_ids))
        num_straight_envs = int(self.cfg.commands.rel_straight_envs * len(env_ids))
        random_perm = torch.randperm(len(env_ids))
        standing_env_ids = env_ids[random_perm[:num_standing_envs]]
        marking_env_ids = env_ids[random_perm[num_standing_envs:num_standing_envs+num_marking_envs]]
        straight_env_ids = env_ids[random_perm[num_standing_envs+num_marking_envs:num_standing_envs+num_marking_envs+num_straight_envs]]
        random_env_ids = env_ids[random_perm[num_standing_envs+num_marking_envs+num_straight_envs:]]

        # standing
        self.commands[standing_env_ids, :4] = 0
        self.commands[standing_env_ids, 4] = 1
        # marking time
        self.commands[marking_env_ids] = 0
        # straight
        self.commands[straight_env_ids] = 0
        self.commands[straight_env_ids, 0] = torch_rand_float(0, self.command_ranges["lin_vel_x"][1], (len(straight_env_ids), 1), device=self.device).squeeze(1)
        # random
        self.commands[random_env_ids, 4] = 0

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 6] = 0.  # commands
        noise_vec[6: 18] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[18: 30] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[30: 42] = 0
        noise_vec[42: 45] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[45: 47] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        noise_vec[47: 50] = noise_scales.lin_acc * self.obs_scales.lin_acc  # linear acc
        return noise_vec
