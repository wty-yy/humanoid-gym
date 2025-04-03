import numpy as np
from humanoid.envs import Kuavo42LeggedFineCfg, Kuavo42LeggedFineCfgPPO


class Kuavo42LeggedLejuCfg(Kuavo42LeggedFineCfg):
    class env(Kuavo42LeggedFineCfg.env):
        frame_stack = 100
        num_single_obs = 6 + 12 * 3 + 3 * 2 + 2
        num_observations = int(frame_stack * num_single_obs)
        c_frame_stack = 3
        single_num_privileged_obs = 74
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

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

class Kuavo42LeggedLejuCfgPPO(Kuavo42LeggedFineCfgPPO):
    class runner(Kuavo42LeggedFineCfgPPO.runner):
        policy_class_name = 'LongShortActorCritic'
        experiment_name = 'Kuavo42_legged_leju_ppo'
