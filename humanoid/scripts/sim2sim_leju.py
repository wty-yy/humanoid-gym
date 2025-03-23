"""
python humanoid/scripts/sim2sim_leju.py --load-onnx models/Isaaclab/v2_20250319_lowpd.onnx --version isaacsim
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_ppo_v8_model_3001.onnx \
  --cycle-time 1.2 --version legged_gym
"""

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import Kuavo42Leggeds2sCfg, Kuavo42LeggedCfg
from humanoid.scripts.sim2sim import quaternion_to_euler_array
import torch


class cmd:
  vx = 0.0
  vy = 0.0
  dyaw = 0.0

def get_obs(data):
  '''Extracts an observation from the mujoco data structure
  Returns:
    [joint_position, joint_velocity, quaternion, 
    base_velocoity, base_ang_vel, projected_gravity]
  '''
  q = data.qpos.astype(np.double)
  dq = data.qvel.astype(np.double)
  quat = data.sensor('BodyQuat').data[[1, 2, 3, 0]].astype(np.double)
  r = R.from_quat(quat)
  v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
  omega = data.sensor('BodyGyro').data.astype(np.double)  # base angular-velocity
  # omega = r.apply(omega, inverse=True).astype(np.double)
  # v = data.sensor('BodyVel').data.astype(np.double)  # base velocity
  # v = r.apply(v, inverse=True).astype(np.double)
  gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
  return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
  '''Calculates torques from position commands
  '''
  return (target_q - q) * kp + (target_dq - dq) * kd

def convert_joint_idx(vec: np.ndarray, to_lab: bool):
  x = vec.copy()
  if to_lab:
    x[cfg.convert.joint_cvt_idx] = x.copy()
  else:
    x = x[cfg.convert.joint_cvt_idx]
  return x

def run_mujoco(policy, cfg: Kuavo42LeggedCfg, version):
  """
  Run the Mujoco simulation using the provided policy and configuration.

  Args:
    policy: The policy used for controlling the simulation.
    cfg: The configuration object containing simulation settings.
    version: Support [isaacsim, legged_gym]

  Returns:
    None
  """
  model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
  model.opt.timestep = cfg.sim_config.dt
  data = mujoco.MjData(model)
  mujoco.mj_step(model, data)
  viewer = mujoco_viewer.MujocoViewer(model, data)

  target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
  action = np.zeros((cfg.env.num_actions), dtype=np.double)

  count_lowlevel = 0

  if version == 'legged_gym':
    obs_stack = np.zeros([cfg.env.frame_stack, cfg.env.num_single_obs], np.float32)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)][1:]  # first one is None
    default_joint_pos = np.array([cfg.init_state.default_joint_angles[name] for name in joint_names])
    data.qpos[-cfg.env.num_actions:] = default_joint_pos
    mujoco.mj_step(model, data)
    viewer.render()


  for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

    # Obtain an observation
    q, dq, quat, v, omega, gvec = get_obs(data)
    # print(f"{quat=}")
    q = q[-cfg.env.num_actions:]  # q:shape=(19,), get foot action:shape=(12,)
    dq = dq[-cfg.env.num_actions:]

    # 1000hz -> 100hz
    if count_lowlevel % cfg.sim_config.decimation == 0:

      if version == 'isaacsim':
        obs = np.concatenate([
          v, omega, gvec, [cmd.vx, cmd.vy, cmd.dyaw], 
          convert_joint_idx(q, True),
          convert_joint_idx(dq, True),
          convert_joint_idx(action, True)
        ], dtype=np.float32).reshape(1, -1)
      elif version == 'legged_gym':
        phase = count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time
        print(cfg.rewards.cycle_time)
        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        tmp = np.concatenate([
          [
            math.sin(2 * math.pi * phase),
            math.cos(2 * math.pi * phase),
            cmd.vx * cfg.normalization.obs_scales.lin_vel,
            cmd.vy * cfg.normalization.obs_scales.lin_vel,
            cmd.dyaw * cfg.normalization.obs_scales.ang_vel,
          ],
          q * cfg.normalization.obs_scales.dof_pos,
          dq * cfg.normalization.obs_scales.dof_vel,
          action,
          omega * cfg.normalization.obs_scales.ang_vel,
          eu_ang * cfg.normalization.obs_scales.quat
        ], dtype=np.float32).reshape(1, -1)
        obs_stack = np.concatenate([obs_stack[1:], tmp], axis=0)
        obs = obs_stack.reshape(1, -1)

      # print(f"{v=},\n{omega=},\n{gvec=},\ncmd={[cmd.vx, cmd.vy, cmd.dyaw]},")
      # print(f"q={convert_joint_idx(q, True)},")
      # print(f"dq={convert_joint_idx(dq, True)},")
      # print(f"action={convert_joint_idx(action, True)}")

      obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

      action[:] = policy(obs)[0]
      if version == 'isaacsim':
        action = convert_joint_idx(action, False)
      action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

      target_q = action * cfg.control.action_scale


    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    # Generate PD control
    tau = pd_control(target_q, q, cfg.robot_config.kps,
            target_dq, dq, cfg.robot_config.kds)  # Calc torques
    tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
    # tau = np.zeros_like(tau)
    data.ctrl = tau

    mujoco.mj_step(model, data)
    viewer.render()
    count_lowlevel += 1

  viewer.close()


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Deployment script.')
  parser.add_argument('--load-onnx', type=str, required=True,
    help='Run to load from.')
  parser.add_argument('--version', type=str, default='isaacsim',
    help="Support model version: [isaacsim, legged_gym]")
  parser.add_argument('--cycle-time', type=float, default=0.64,
    help="Cover cfg.rewards.cycle_time")
  args = parser.parse_args()

  if args.version == 'isaacsim':
    cfg_class = Kuavo42Leggeds2sCfg
  elif args.version == 'legged_gym':
    cfg_class = Kuavo42LeggedCfg
  else:
    raise ValueError(f"Don't know version={args.version}")

  class Sim2simCfg(cfg_class):

    class sim_config:
      mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/mjcf/biped_s42_fixed_arm.xml'
      sim_duration = 60.0
      # sim_duration = 0.2
      dt = 0.001
      decimation = 10

    class robot_config:
      kps = np.array([60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0], dtype=np.double)
      kds = np.array([34.0, 6.0, 12.0, 12.0, 22.0, 22.0, 34.0, 6.0, 12.0, 12.0, 22.0, 22.0], dtype=np.double)
      tau_limit = 200. * np.ones(12, dtype=np.double)

  import onnxruntime as ort
  session = ort.InferenceSession(args.load_onnx)
  input_name = session.get_inputs()[0].name
  policy = lambda x: session.run(None, {input_name: x})[0]
  cfg = Sim2simCfg()
  cfg.rewards.cycle_time = args.cycle_time
  run_mujoco(policy, cfg, args.version)
