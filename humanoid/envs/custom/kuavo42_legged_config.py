import numpy as np
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Kuavo42LeggedCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/mjcf/biped_s42_fixed_arm.xml'

        name = 'Kuavo42'
        foot_name = knee_name = None
        foot_names = ['leg_l6_link', 'leg_r6_link']
        knee_names = ['leg_l4_link', 'leg_r4_link']

        terminate_after_contacts_on = []
        terminate_base_height = 0.3
        penalize_contacts_on = [
            "base_link"
        ]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0., 0., 0.88]

        default_joint_angles = {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [0.0, 0.0, -0.27, 0.52, -0.3, 0.0]):
                default_joint_angles[f'leg_{lr}{idx}_joint'] = value

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness, damping = {}, {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [60.0, 60.0, 60.0, 60.0, 15.0, 15.0]):
                stiffness[f'leg_{lr}{idx}_joint'] = value
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [34.0, 6.0, 12.0, 12.0, 22.0, 22.0]):
                damping[f'leg_{lr}{idx}_joint'] = value

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        prob_high_lin_y_and_yaw = 0.05  # prob for both high line y and yaw velocity

        class ranges:
            lin_vel_x = [-0.4, 1.0]   # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.85
        min_dist = 0.25
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joints_delta = [-0.25, 0.5, -0.25]  # leg, knee, foot
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 550  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 3.2
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.0
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.6
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = 0.0
            termination = -200.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class Kuavo42LeggedCfgPPO(LeggedRobotCfgPPO):
    seed = 42
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    # class algorithm(LeggedRobotCfgPPO.algorithm):
    #     entropy_coef = 0.001
    #     learning_rate = 1e-5
    #     num_learning_epochs = 2
    #     gamma = 0.994
    #     lam = 0.9
    #     num_mini_batches = 4

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Kuavo42_legged_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        wandb = True

class Kuavo42LeggedHighPDCfg(Kuavo42LeggedCfg):
    class control(Kuavo42LeggedCfg.control):
        # PD Drive parameters:
        stiffness, damping = {}, {}
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [120.0, 120.0, 120.0, 120.0, 30.0, 30.0]):
                stiffness[f'leg_{lr}{idx}_joint'] = value
        for lr in ['l', 'r']:
            for idx, value in zip(range(1, 7), [10.0, 6.0, 12.0, 12.0, 22.0, 22.0]):
                damping[f'leg_{lr}{idx}_joint'] = value

class Kuavo42LeggedHighPDCfgPPO(Kuavo42LeggedCfgPPO):
    class runner(Kuavo42LeggedCfgPPO.runner):
        experiment_name = 'Kuavo42_legged_high_pd_ppo'

class Kuavo42LeggedSingleObsCfg(Kuavo42LeggedCfg):
    class env(Kuavo42LeggedCfg.env):
        frame_stack = 1
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)

class Kuavo42LeggedSingleObsCfgPPO(Kuavo42LeggedCfgPPO):
    class runner(Kuavo42LeggedCfgPPO.runner):
        experiment_name = 'Kuavo42_legged_single_obs_ppo'