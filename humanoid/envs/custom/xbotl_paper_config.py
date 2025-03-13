from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class XBotLPaperCfg(XBotLCfg):
  class env(XBotLCfg.env):
    # change the observation dim
    c_frame_stack = 3
    single_num_privileged_obs = 61
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

  class commands(XBotLCfg.commands):
    # Keep True and try False
    heading_command = True  # if true: compute ang vel command from heading error

  class rewards:
    base_height_target = 0.89
    cycle_time = 0.64  # left and right foot down cycle (first right, second left)
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    # Polynomial foot trajectory parameters, from low to heigh order
    foot_traj_poly_coefs = [0.0, 0.1, 5.0, -18.8, 12.0, 9.6]

    class scales:
      # cmd tracking
      lin_velocity_tracking = 1.0
      ang_velocity_tracking = 1.0
      orientation_tracking = 1.0
      base_height_tracking = 1.0
      # period
      periodic_force = 1e-3
      periodic_velocity = 1.0
      # foot tracking
      foot_height_tracking = 1.0
      foot_vel_tracking = 1.0
      # others
      default_joint = 0.2
      energy_cost = -1e-4
      action_smoothness = -1e-2
      feet_movements = -1e-2
      large_contact = -1e-2

class XBotLPaperCfgPPO(XBotLCfgPPO):
  class runner(XBotLCfgPPO.runner):
    # logging
    experiment_name = 'XBot_paper_ppo'