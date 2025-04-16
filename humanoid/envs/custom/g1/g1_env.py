import os
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch, mujoco
from humanoid.envs import Kuavo42LeggedEnv, Kuavo42LeggedFineObsEnv

from humanoid.utils.terrain import  HumanoidTerrain
from humanoid.envs.base.legged_robot import get_euler_xyz_tensor


class G1Env(Kuavo42LeggedEnv):
    '''
    Same as Kuavo42LeggedEnv
    '''
    ...

class G1ObsEnv(Kuavo42LeggedFineObsEnv):
    ...

class G1ObsDomainEnv(G1ObsEnv):
    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos + self.joint_pos_bias) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
            self.com_displacement,  # 3
            self.restitution_coeffs,  # 1
            self.joint_pos_bias,  # 12
            # self.joint_friction_coeffs,  # 12
            # self.joint_armature_coeffs,  # 12
            self.kp_factors,  # 12
            self.kd_factors,  # 12
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity * self.obs_scales.quat,  # 3
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
    
        # if self.cfg.domain_rand.push_robots:
        #     self.sample_random_push()

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
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.kp_factors
        d_gains = self.d_gains * self.kd_factors

        p_torque = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
        d_torque = - d_gains * self.dof_vel
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

            self.dof_armature = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_armature[i] = props["armature"][i].item()
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit

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