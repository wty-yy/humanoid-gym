import os
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import Kuavo42LeggedEnv, Kuavo42LeggedFineObsEnv

from humanoid.utils.terrain import  HumanoidTerrain


class G1Env(Kuavo42LeggedEnv):
    '''
    Same as Kuavo42LeggedEnv
    '''
    ...

class G1ObsEnv(Kuavo42LeggedFineObsEnv):
    ...