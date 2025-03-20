from humanoid.envs.custom.kuavo42_legged_config import Kuavo42LeggedCfg, Kuavo42LeggedCfgPPO


class Kuavo42LeggedSimpleRewardCfg(Kuavo42LeggedCfg):

  class env(Kuavo42LeggedCfg.env):
    frame_stack = 15
    c_frame_stack = 3
    num_single_obs = 45
    num_observations = int(frame_stack * num_single_obs)
    single_num_privileged_obs = 57
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

  class rewards:
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = False
    base_height_target = 0.75

    class scales:
      lin_velocity_tracking = 1.0
      ang_velocity_tracking = 1.5
      orientation = -2
      energy = -2.5e-7
      base_height_tracking = 0.5

class Kuavo42LeggedSimpleRewardCfgPPO(Kuavo42LeggedCfgPPO):
  class runner(Kuavo42LeggedCfgPPO.runner):
    # logging
    experiment_name = 'Kuavo42_legged_simple_reward_ppo'