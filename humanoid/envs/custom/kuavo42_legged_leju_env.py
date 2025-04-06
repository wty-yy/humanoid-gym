import os, mujoco
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.envs.base.legged_robot import get_euler_xyz_tensor

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
import torch.nn as nn
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain
from collections import deque
from humanoid.utils.forward_kinematics import ForwardKinematics


class Kuavo42LeggedLejuEnv(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.resample_cmd_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.fk = ForwardKinematics(mjcf_path=self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        self.fk.to(self.device)

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

        self.load_gait_model()
        self.build_period_history()
        self.contact_history = deque(maxlen=round(self.cfg.rewards.cycle_time / self.dt))
        for _ in range(self.contact_history.maxlen):
            self.contact_history.append(torch.zeros((self.num_envs, 2), device=self.device))
        self.vel_history = deque(maxlen=round(self.cfg.rewards.cycle_time / self.dt))
        for _ in range(self.vel_history.maxlen):
            self.vel_history.append(torch.zeros((self.num_envs, 3), device=self.device))

        self.compute_observations()

        scripts_path = cfg.env.scripts_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.is_ankle_pos_legal = torch.jit.load(os.path.join(scripts_path, "is_ankle_pos_legal.pt"))
        self.joint_to_motor_position = torch.jit.load(os.path.join(scripts_path, "joint_to_motor_position.pt"))
        self.get_joint_dumping_torque = torch.jit.load(os.path.join(scripts_path, "get_joint_dumping_torque.pt"))
    
    def load_gait_model(self):
        self.gait_model = nn.Sequential(
            nn.Linear(6, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 12 + 14 + 9)
        ).to(self.device)
        self.gait_model.load_state_dict(
            torch.load(self.cfg.env.gait_model_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)))
    
    def build_period_history(self):
        self.half_period_length = round(self.cfg.rewards.cycle_time / self.dt / 2)
        self.period_history = {}
        self.period_symmetric = self.cfg.rewards.period_symmetric
        for name in self.period_symmetric:
            self.period_history[name] = deque(maxlen=self.half_period_length)
            dim_num = self.period_symmetric[name]["dim_num"]
            for _ in range(self.half_period_length):
                self.period_history[name].append(torch.zeros((self.num_envs, dim_num), device=self.device))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        phase[self.commands[:,4].to(bool)] = 0
        return phase

    def _get_gait_phase(self):
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = self.ref_body_positions["leg_l6_link"][:, 2] < self.cfg.rewards.foot_height
        stance_mask[:, 1] = self.ref_body_positions["leg_r6_link"][:, 2] < self.cfg.rewards.foot_height
        stance_mask[self.commands[:, 4].to(bool)] = 1
        return stance_mask

    def get_neural_ref_dof_pos(self):
        phase = self._get_phase()
        inputs = torch.zeros((self.num_envs, 6), device=self.device)
        inputs[:, 0] = torch.sin(2 * torch.pi * phase)
        inputs[:, 1] = torch.cos(2 * torch.pi * phase)
        inputs[:, 2] = self.cfg.rewards.cycle_time
        inputs[:, 3:6] = self.commands[:, :3]
        outputs = self.gait_model(inputs)

        self.ref_dof_pos = outputs[:, :self.num_dof]
        self.ref_euler_xy = outputs[:, self.num_dof:self.num_dof + 2]
        self.ref_height = outputs[:, self.num_dof + 2]
        self.ref_lin_vel = outputs[:, self.num_dof + 3:self.num_dof + 6]
        self.ref_ang_vel = outputs[:, self.num_dof + 6:self.num_dof + 9]

        self.ref_dof_pos[self.commands[:, 4].to(bool)] = self.default_dof_pos
        self.ref_euler_xy[self.commands[:, 4].to(bool), 0] = 0
        self.ref_euler_xy[self.commands[:, 4].to(bool), 1] = 0.05
        self.ref_height[self.commands[:, 4].to(bool)] = self.cfg.init_state.pos[-1]
        self.ref_lin_vel[self.commands[:, 4].to(bool)] = 0
        self.ref_ang_vel[self.commands[:, 4].to(bool)] = 0

    def get_ref_position_rotation(self):
        qpos = torch.zeros(self.num_envs, 7 + self.num_dofs, device=self.device)
        qpos[:, :2] = self.root_states[:, :2]
        qpos[:, 2] = self.ref_height
        qpos[:, 3:7] = quat_from_euler_xyz(self.ref_euler_xy[:, 0], self.ref_euler_xy[:, 1], self.base_euler_xyz[:, 2])
        qpos[:, 7:7 + self.num_dofs] = self.ref_dof_pos
        self.ref_body_positions, self.ref_body_rotations = self.fk(qpos, with_root=True)

    def compute_ref_state(self):
        self.get_neural_ref_dof_pos()
        self.get_ref_position_rotation()
        self.ref_action = 2 * self.ref_dof_pos
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :4]), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos + self.joint_pos_bias) * self.obs_scales.dof_pos
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
            self.random_force_value[:, :5],  # 5
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
            self.com_displacement,  # 3
            self.restitution_coeffs,  # 1
            self.joint_pos_bias,  # 12
            self.joint_friction_coeffs,  # 12
            self.joint_armature_coeffs,  # 12
            self.kp_factors,  # 12
            self.kd_factors,  # 12
            self.ref_euler_xy,  # 2
            self.ref_height.reshape(-1, 1),  # 1
            self.ref_lin_vel,  # 3
            self.ref_ang_vel  # 3
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

        for name in self.period_symmetric:
            self.period_history[name].append(self.get_period_symmetric_value(name).clone())
        self.contact_history.append(self.contact_forces[:, self.feet_indices, 2])
        self.vel_history.append(torch.cat([self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]], dim=-1))
        self.mean_vel = torch.mean(torch.stack([self.vel_history[i] for i in range(self.vel_history.maxlen)], dim=1), dim=1)

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
        
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_lin_vel).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_ang_vel)

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

        self.resample_cmd_length_buf[env_ids] = 0

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos # + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _get_noise_scale_vec(self, cfg):
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
    
    def _push_robots(self):
        """ remove """
        return
    
    @property
    def is_pushing(self):
        return torch.gt(self.episode_length_buf, self.random_force_begin) & torch.lt(self.episode_length_buf, self.random_force_begin + self.random_force_length)
    
    def sample_random_push(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        clean_idx = torch.gt(self.episode_length_buf, self.random_force_begin + self.random_force_length)
        self.random_force_value[clean_idx] = 0

        push_idx = torch.gt(
            self.episode_length_buf,
            self.random_force_begin + self.random_force_length + self.cfg.domain_rand.push_interval_s / self.dt
        )
        if push_idx.sum() == 0:
            return

        self.random_force_begin[push_idx] = self.episode_length_buf[push_idx]
        low_s, high_s = self.cfg.domain_rand.push_length_s
        low, high = int(low_s / self.dt), int(high_s / self.dt)
        self.random_force_length[push_idx] = torch.randint(low, high, (int(push_idx.sum()),)).to(self.device)

        for i, max_push in enumerate(self.cfg.domain_rand.max_push):
            self.random_force_value[push_idx, i] = 2 * torch.rand(int(push_idx.sum())).to(
                self.device) * max_push - max_push

        self.random_force_value[push_idx] /= (self.random_force_length[push_idx] * self.dt).reshape(-1, 1)

        # print(self.random_force_length[push_idx], self.random_force_value[push_idx])
    
    def check_termination(self):
        super().check_termination()
        ankle_illegal = (~self.is_ankle_pos_legal(self.dof_pos[:, 4:6])) | (~self.is_ankle_pos_legal(self.dof_pos[:, 10:12]))
        self.reset_buf |= ankle_illegal

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.resample_cmd_length_buf += 1
    
        if self.cfg.domain_rand.push_robots:
            self.sample_random_push()

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # compute linear acc by LowPassFilter2ndOrder
        free_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt  # acc without filter
        free_acc[:, 2] -= self.sim_params.gravity.z
        self.base_lin_acc[:] = self.lin_acc_filter.update(quat_rotate_inverse(self.base_quat, free_acc))

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
    
    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.domain_rand.randomize_kp:
            self.kp_factors = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1],
                (self.num_envs, self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_kp:
            self.kp_factors = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1],
                (self.num_envs, self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_kd:
            self.kd_factors = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1],
                (self.num_envs, self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_euler_xy_zero_pos:
            self.euler_xy_zero_pos = torch_rand_float(
                self.cfg.domain_rand.euler_xy_zero_pos_range[0],
                self.cfg.domain_rand.euler_xy_zero_pos_range[1],
                (self.num_envs, 2), device=self.device
            )
        if self.cfg.domain_rand.randomize_joint_pos_bias:
            self.joint_pos_bias = torch_rand_float(
                self.cfg.domain_rand.joint_pos_bias_range[0],
                self.cfg.domain_rand.joint_pos_bias_range[1],
                (self.num_envs, self.num_dof), device=self.device
            )
        self.random_force_value = torch.zeros(
            self.num_envs, 6, dtype=torch.float,
            device=self.device, requires_grad=False
        )
        self.random_force_length = torch.zeros(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False
        )
        self.random_force_begin = torch.zeros(
            self.num_envs, dtype=torch.long,
            device=self.device, requires_grad=False
        )
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.cfg.domain_rand.randomize_kp:
            self.kp_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0],
                self.cfg.domain_rand.kp_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_joint_pos_bias:
            self.joint_pos_bias = torch_rand_float(
                self.cfg.domain_rand.joint_pos_bias_range[0],
                self.cfg.domain_rand.joint_pos_bias_range[1],
                (self.num_envs, self.num_dof), device=self.device
            )
        if self.cfg.domain_rand.randomize_kd:
            self.kd_factors[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0],
                self.cfg.domain_rand.kd_range[1],
                (len(env_ids), self.num_actions), device=self.device
            )
        if self.cfg.domain_rand.randomize_euler_xy_zero_pos:
            self.euler_xy_zero_pos[env_ids] = torch_rand_float(
                self.cfg.domain_rand.euler_xy_zero_pos_range[0],
                self.cfg.domain_rand.euler_xy_zero_pos_range[1],
                (len(env_ids), 2), device=self.device
            )
        self.random_force_value[env_ids] = 0
        self.random_force_length[env_ids] = 0
        self.random_force_begin[env_ids] = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        # LeggedRobot.step
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        # add
        force_tensor = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        force_tensor[:, 0] = self.random_force_value[:, :3]
        torque_tensor = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        torque_tensor[:, 0] = self.random_force_value[:, 3:6]

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            # add
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(force_tensor),
                gymtorch.unwrap_tensor(torque_tensor)
            )

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        # add
        self.obs_buf[torch.isnan(self.obs_buf)] = 0

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

            # add
            self.privileged_obs_buf[torch.isnan(self.privileged_obs_buf)] = 0

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)

        if self.cfg.domain_rand.randomize_com_displacement:
            rng = self.cfg.domain_rand.com_displacement_range
            disp = np.random.uniform(rng[0], rng[1], 3)
            props[0].com += gymapi.Vec3(disp[0], disp[1],disp[2])
            self.com_displacement[env_id] = torch.from_numpy(disp).to(torch.float32).to(self.device)

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * props[i].mass
        return props

    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factors
        d_gains = self.d_gains * self.kd_factors

        p_torque = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        motor_pos = self.joint_to_motor_position(self.dof_pos)
        d_torque = - self.get_joint_dumping_torque(self.dof_pos, motor_pos, d_gains, self.dof_vel)
        torques = p_torque + d_torque

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_actions), device=self.device)
            torques *= motor_strength_factors

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def get_frictionloss_from_mujoco(self):
        xml_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        model = mujoco.MjModel.from_xml_path(xml_path)
        return torch.from_numpy(model.dof_frictionloss[6:]).to(self.device).to(torch.float32).abs()
    
    def get_torque_limit_from_mujoco(self):
        xml_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        model = mujoco.MjModel.from_xml_path(xml_path)
        return torch.from_numpy(model.actuator_ctrlrange[:, 0]).to(self.device).to(torch.float32).abs()

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            # prepare friction randomization
            if self.cfg.domain_rand.randomize_joint_friction:
                joint_friction_range = self.cfg.domain_rand.joint_friction_range
                self.joint_friction_coeffs = torch_rand_float(
                    joint_friction_range[0], joint_friction_range[1],
                    (self.num_envs, self.num_actions),
                    device=self.device
                ).squeeze(1)
            else:
                self.joint_friction_coeffs = torch.ones(
                    self.num_envs, self.num_actions, dtype=torch.float,
                    device=self.device, requires_grad=False
                )

            if self.cfg.domain_rand.randomize_joint_armature:
                joint_armature_range = self.cfg.domain_rand.joint_armature_range
                self.joint_armature_coeffs = torch_rand_float(
                    joint_armature_range[0], joint_armature_range[1],
                    (self.num_envs, self.num_actions),
                    device=self.device
                ).squeeze(1)
            else:
                self.joint_armature_coeffs = torch.ones(
                    self.num_envs, self.num_actions, dtype=torch.float,
                    device=self.device, requires_grad=False
                )

            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = self.get_torque_limit_from_mujoco() * self.cfg.safety.torque_limit
            self.dof_vel_limits = torch.tensor(self.cfg.asset.velocity_limit).to(self.device).to(torch.float32) * self.cfg.safety.vel_limit

            self.dof_armature = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_armature[i] = props["armature"][i].item()

            self.dof_friction = self.get_frictionloss_from_mujoco()

        props["effort"][:] = self.torque_limits.cpu().numpy() / self.cfg.safety.torque_limit
        props["velocity"][:] = self.dof_vel_limits.cpu().numpy() /  self.cfg.safety.vel_limit

        for i in range(len(props)):
            props["friction"][i] = self.dof_friction[i].item() * self.joint_friction_coeffs[env_id, i].item()
            props["armature"][i] = self.dof_armature[i].item() * self.joint_armature_coeffs[env_id, i].item()
            props["damping"][i] = 0
            props["stiffness"][i] = 0

        return props
    
    def _create_envs(self):
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device, requires_grad=False)
        super()._create_envs()
    
    def get_period_symmetric_value(self, name):
        dim_num = self.period_symmetric[name]["dim_num"]
        if hasattr(self, name):
            return getattr(self, name)[:, :dim_num].clone()
        else:
            raise ValueError(f"Period symmetric name {name} not recognised")
    
    # =========================== Rewards ========================== #
    def _reward_joint_pos(self):
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target

        sigma = self.cfg.rewards.joint_pos_sigma * torch.ones(self.num_envs, device=self.device)
        for idx in self.cfg.rewards.roll_joint_idxs:  # don't consider rolling joints when moving
            diff[~self.commands[:, 4].bool(), idx] *= 0
        diff[~self.commands[:, 4].bool(), 2] *= 0.5  # thigh pitch joint
        diff[~self.commands[:, 4].bool(), 8] *= 0.5  # thigh pitch joint
        rew = torch.exp(-sigma * torch.norm(diff, dim=1))
        rew[self.is_pushing] = 1  # don't consider reward when pushing
        ratio = torch.clip(self.resample_cmd_length_buf * self.dt / self.cfg.rewards.cycle_time, 0, 1)
        rew = rew * ratio + (1 - ratio)
        return rew
    
    def _reward_half_period(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        max_reward = 0
        for name in self.period_symmetric:
            target = self.period_history[name][0].clone()
            target[self.commands[:, 4].to(bool)] = self.get_period_symmetric_value(name)[self.commands[:, 4].to(bool)].clone()
            target[:, :12] = torch.roll(target[:, :12], shifts=6, dims=1)
            target[:, self.cfg.rewards.unpitch_joint_idxs] *= -1

            diff = self.get_period_symmetric_value(name) - target
            diff[:, self.cfg.rewards.unpitch_joint_idxs] *= 5
            sigma = self.period_symmetric[name]["sigma"] * torch.ones(self.num_envs, device=self.device)
            sigma[~(self.commands[:, 1:] == 0).all(dim=1)] = 0
            rew = torch.exp(-sigma * torch.norm(diff, dim=1))
            reward += rew * self.period_symmetric[name]["scale"]
            max_reward += self.period_symmetric[name]["scale"]
        reward[self.is_pushing] = max_reward
        return reward

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        rew = (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2
        return rew

    def _reward_knee_distance(self):
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2

    def _reward_feet_contact_forces(self):
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        rew = (contact_force.sum(-1) - self.cfg.rewards.max_contact_force).clip(0, 400)
        rew[self.episode_length_buf < 20] = 0
        return rew

    def _reward_foot_pos(self):
        stance_mask = self.contact_forces[:, self.feet_indices, 2] > 100.
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1) - self.cfg.rewards.foot_height
        measured_heights[torch.isnan(measured_heights)] = 0

        left_pos_diff = self.rigid_state[:, 6, :3] - self.ref_body_positions["leg_l6_link"]
        left_pos_diff[:, 2] -= measured_heights
        right_pos_diff = self.rigid_state[:, 12, :3] - self.ref_body_positions["leg_r6_link"]
        right_pos_diff[:, 2] -= measured_heights
        rew = torch.exp(-torch.norm(torch.cat([left_pos_diff, right_pos_diff], dim=-1), dim=1) * 20)
        rew[self.is_pushing] = 1
        return rew
    
    def _reward_tracking_x_lin_vel(self):
        x_vel_error = torch.abs(self.commands[:, 0] - self.mean_vel[:, 0])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.x_tracking_sigmas:
            rew += torch.exp(-sigma * x_vel_error) / len(self.cfg.rewards.x_tracking_sigmas)
        ref_instant_vel = self.commands[:, 0]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_lin_vel[:, 0]))
        return rew / 2

    def _reward_tracking_y_lin_vel(self):
        y_vel_error = torch.abs(self.commands[:, 1] - self.mean_vel[:, 1])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.y_tracking_sigmas:
            rew += torch.exp(-sigma * y_vel_error) / len(self.cfg.rewards.y_tracking_sigmas)
        ref_instant_vel = self.commands[:, 1]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_lin_vel[:, 1]))
        return rew / 2
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.abs(self.commands[:, 2] - self.mean_vel[:, 2])
        rew = torch.zeros(self.num_envs, device=self.device)
        for sigma in self.cfg.rewards.yaw_tracking_sigmas:
            rew += torch.exp(-sigma * ang_vel_error) / len(self.cfg.rewards.yaw_tracking_sigmas)
        ref_instant_vel = self.commands[:, 2]
        rew += torch.exp(-10 * torch.square(ref_instant_vel - self.base_ang_vel[:, 2]))
        return rew / 2

    def _reward_feet_clearance(self):
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    
        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
    
        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()
    
        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_base_height(self):
        stance_mask = self.contact_forces[:, self.feet_indices, 2] > 100.
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1) - self.cfg.rewards.foot_height
        measured_heights[torch.isnan(measured_heights)] = 0
        base_height = self.root_states[:, 2] - measured_heights
        rew = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 20)
        return rew

    def _reward_base_acc(self):
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_feet_contact_same(self):
        contacts = torch.stack([self.contact_history[i] for i in range(len(self.contact_history))], dim=1)
        mean_force = contacts.mean(dim=1)
        diff1 = (mean_force[:, 0] - mean_force[:, 1])
        diff2 = (contacts[:, -1] - contacts[:, -self.contact_history.maxlen // 2]).sum(-1)
        rew1 = torch.exp(- 0.01 * diff1.abs())
        rew2 = torch.exp(- 0.01 * diff2.abs())
        return (rew1 + rew2) / 2
    
    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 0., -1.)
        return torch.mean(reward, dim=1)

    def _reward_torques(self):
        weight = torch.tensor(self.cfg.rewards.torque_weights, device=self.device)
        rew = torch.sum(torch.square(self.torques * weight), dim=1)
        rew[self.is_pushing] /= 3
        return rew
    
    def _reward_dof_vel(self):
        weight = torch.tensor(self.cfg.rewards.dof_vel_weights, device=self.device)
        rew = torch.sum(torch.square(self.dof_vel * weight), dim=1)
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3