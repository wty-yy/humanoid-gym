import numpy as np
from humanoid.envs import Kuavo42LeggedFineCfg, Kuavo42LeggedFineCfgPPO


class Kuavo42LeggedLejuCfg(Kuavo42LeggedFineCfg):
    class env(Kuavo42LeggedFineCfg.env):
        frame_stack = 100
        num_single_obs = 6 + 12 * 3 + 3 * 2 + 2
        num_observations = int(frame_stack * num_single_obs)
        c_frame_stack = 3
        single_num_privileged_obs = 126
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    
    class asset(Kuavo42LeggedFineCfg.asset):
        velocity_limit = [14, 14, 23, 14, 10, 10] * 2  # + [10] * 14

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
    
    class commands(Kuavo42LeggedFineCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stance
        num_commands = 5
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        rel_standing_envs = 0.1
        rel_marking_envs = 0.1
        rel_straight_envs = 0.4

        class ranges:
            lin_vel_x = [-0.4, 1.0]   # min max [m/s]
            lin_vel_y = [-0.2, 0.2]   # min max [m/s]
            ang_vel_yaw = [-0.4, 0.4] # min max [rad/s]
            heading = [-3.14, 3.14]
    
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

    class control(Kuavo42LeggedFineCfg.control):
        action_scale = 0.25

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

        push_robots = True  # 连续推力
        push_interval_s = 6.
        push_length_s = [0.05, 0.5]
        max_push = [15, 15, 15, 5, 5, 5]

        randomize_kp = True  # kp值
        kp_range = [0.8, 1.2]

        randomize_kd = True  # kd值
        kd_range = [0.8, 1.2]

        randomize_joint_pos_bias = False
        joint_pos_bias_range = [-0.05, 0.05]

        randomize_euler_xy_zero_pos = False
        euler_xy_zero_pos_range = [-0.03, 0.03]

        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02
    
    class rewards(Kuavo42LeggedFineCfg.rewards):
        cycle_time = 1.2                # sec

class Kuavo42LeggedLejuCfgPPO(Kuavo42LeggedFineCfgPPO):
    class runner(Kuavo42LeggedFineCfgPPO.runner):
        policy_class_name = 'LongShortActorCritic'
        experiment_name = 'Kuavo42_legged_leju_ppo'
