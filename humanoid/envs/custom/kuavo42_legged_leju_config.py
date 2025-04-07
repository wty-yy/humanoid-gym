import numpy as np
from itertools import product
from humanoid.envs import Kuavo42LeggedFineCfg, Kuavo42LeggedFineCfgPPO
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Kuavo42LeggedLejuCfg(LeggedRobotCfg):
    class env(Kuavo42LeggedFineCfg.env):
        frame_stack = 100
        num_single_obs = 6 + 12 * 3 + 3 * 2 + 2
        num_observations = int(frame_stack * num_single_obs)
        c_frame_stack = 3
        single_num_privileged_obs = 162
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False  # speed up training by using reference actions

        gait_model_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/gait_sk120.pth'
        scripts_path = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/scripts'
    
    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0
    
    class asset(Kuavo42LeggedFineCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s42_fine/xml/biped_s42_only_lower_body.xml'

        name = "kuavo"
        foot_names = ["leg_r6_link", "leg_l6_link"]
        knee_names = ["leg_r4_link", "leg_l4_link"]

        terminate_after_contacts_on = [
            'base_link',
            'leg_r1_link', 'leg_l1_link',
            'leg_r2_link', 'leg_l2_link',
            'leg_r3_link', 'leg_l3_link',
            'leg_r4_link', 'leg_l4_link',
            'leg_r5_link', 'leg_l5_link',
        ]
        penalize_contacts_on =[
            'base_link',
            'leg_r1_link', 'leg_l1_link',
            'leg_r2_link', 'leg_l2_link',
            'leg_r3_link', 'leg_l3_link',
            'leg_r4_link', 'leg_l4_link',
            'leg_r5_link', 'leg_l5_link',
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

        velocity_limit = [14, 14, 23, 14, 10, 10] * 2  # + [10] * 14

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = True
        # mujoco use the max friction of two contact body, so here is actually the "min friction"
        static_friction = 0
        dynamic_friction = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 5  # number of terrain rows (levels)
        num_cols = 5  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.4, 0., 0.2, 0.2, 0, 0]
        # terrain_proportions = [0.0, 0., 0.0, .0, 1.0, 0, 0]
        restitution = 0.

        measured_points_x = [-0.3, -0.15, 0., 0.15, 0.3]
        measured_points_y = [-0.15, 0., 0.15]

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            lin_acc = 0.5  # add
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.
    
    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stance
        num_commands = 5
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        rel_standing_envs = 0.3
        rel_marking_envs = 0.1
        rel_straight_envs = 0.4
        min_lin_vel = 0.
        min_ang_vel = 0.

        class ranges:
            lin_vel_x = [-0.4, 1.0]   # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4] # min max [rad/s]
            heading = [0, 0]
    
    class noise:
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            lin_acc = 0.5
            quat = 0.03
            height_measurements = 0.1

    class control(LeggedRobotCfg.control):
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

        # PD Drive parameters:
        stiffness, damping = {}, {}

        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [60.0, 60.0, 60.0, 60.0, 30.0, 15.0]):
                stiffness[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            stiffness[f'zarm_{lr}{idx}_joint'] = 15.0

        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [10.0, 6.0, 12.0, 12.0, 22.0, 22.0]):
                damping[f'leg_{lr}{idx}_joint'] = value
        for lr, idx in product(['l', 'r'], range(1, 8)):
            damping[f'zarm_{lr}{idx}_joint'] = 3

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_base_mass = True  # 整机质量随机偏移
        added_mass_range = [-5., 5.]

        randomize_com_displacement = True  # 质心随机偏移
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True  # link质量随机比例缩放
        link_mass_range = [0.8, 1.2]

        randomize_friction = True  # 随机摩擦
        friction_range = [0.2, 1.0]

        randomize_restitution = True  # 随机弹性系数
        restitution_range = [0., 0.5]

        randomize_motor_strength = True  # 电机（力矩强度）
        motor_strength_range = [0.8, 1.2]

        randomize_joint_friction = True  # 关节摩擦
        joint_friction_range = [0.5, 1.5]

        randomize_joint_armature = True  # 转动惯量
        joint_armature_range = [0.5, 1.5]

        push_robots = False  # 连续推力
        push_interval_s = 6.
        push_length_s = [0.05, 0.5]
        max_push = [15, 15, 15, 5, 5, 5]

        randomize_kp = True  # kp值
        kp_range = [0.8, 1.2]

        randomize_kd = True  # kd值
        kd_range = [0.8, 1.2]

        randomize_joint_pos_bias = True
        joint_pos_bias_range = [-0.05, 0.05]

        randomize_euler_xy_zero_pos = False
        euler_xy_zero_pos_range = [-0.03, 0.03]

        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02
    
    class rewards(Kuavo42LeggedFineCfg.rewards):
        cycle_time = 1.2                # sec
        base_height_target = 0.85
        min_dist = 0.25
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joints_delta = [-0.25, 0.5, -0.25]  # leg, knee, foot
        ref_joint_idxs = [2, 3, 4, 8, 9, 10]
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 600  # Forces above this value are penalized
        
        roll_joint_idxs = [0, 5, 6, 11]  # all rolling joints
        yaw_joint_idxs = [1, 7]  # all yawing joints
        pitch_joint_idxs = [2, 3, 4, 8, 9, 10]  # all pitch joints
        unpitch_joint_idxs = roll_joint_idxs + yaw_joint_idxs

        period_symmetric = {  # compute symmetric items in half period
            "dof_pos": {"dim_num": 12, "sigma": 4., "scale": 1.},
        }

        x_tracking_sigmas = [6, 60]
        y_tracking_sigmas = [6, 60]
        yaw_tracking_sigmas = [6, 60]

        foot_height = 0.07
        torque_weights = [1, 1, 1, 1, 2, 3] * 2
        dof_vel_weights = [3, 3, 1, 1, 1, 3] * 2

        class scales:
            joint_pos = 10
            half_period = 2.
            foot_slip = -2.
            foot_pos = 5.
            feet_contact_forces = -0.05
            tracking_x_lin_vel = 3
            tracking_y_lin_vel = 3
            tracking_ang_vel = 3
            vel_mismatch_exp = 2
            low_speed = 0.2
            orientation = 1.
            base_height = 0.5
            base_acc = 0.5  # original 0.2

            feet_contact_same = 1
            feet_contact_number = 20

            # reg
            action_smoothness = -0.01  # original -0.002
            torques = -5e-5  # original -1e-5
            dof_vel = -5e-3  # original -5e-4
            dof_acc = -1e-7  # original -1e-7
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.85]
        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.46, 0.84, -0.44, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value

class Kuavo42LeggedLejuCfgPPO(Kuavo42LeggedFineCfgPPO):
    class runner(Kuavo42LeggedFineCfgPPO.runner):
        policy_class_name = 'LongShortActorCritic'
        experiment_name = 'Kuavo42_legged_leju_ppo'
        wandb_project = 'HumanoidGym-Kuavo'

    class algorithm(Kuavo42LeggedFineCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4