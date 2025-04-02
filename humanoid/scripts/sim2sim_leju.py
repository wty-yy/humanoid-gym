"""
python humanoid/scripts/sim2sim_leju.py --load-onnx models/Isaaclab/v2_20250319_lowpd.onnx --version isaacsim
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_ppo_v8_model_3001.onnx \
  --cycle-time 1.2 --version legged_gym
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_ppo_v8.3_model_10001.onnx \
  --cycle-time 0.64 --version legged_gym
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_single_obs_ppo_v1_model_3001.onnx \
  --cycle-time 0.64 --version legged_gym_single_obs --joystick 1
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_fine_ppo_v1_model_3001.onnx \
  --cycle-time 1.2 --version legged_gym_fine --joystick 1
python humanoid/scripts/sim2sim_leju.py --load-onnx models/g1/g1_ppo_v1_model_3001.onnx \
  --cycle-time 0.64 --version legged_gym_single_obs_g1
python humanoid/scripts/sim2sim_leju.py --load-onnx /home/yy/Coding/robotics/humanoid-gym/logs/g1_obs_ppo/exported/policies/g1_obs_ppo_v1_model_3001.onnx \
  --cycle-time 0.64 --version legged_gym_fine_obs_v2


For DEBUG
python humanoid/scripts/sim2sim_leju.py --load-onnx models/kuavo42_legged/Kuavo42_legged_fine_ppo_v1.1_model_3001.onnx \
  --cycle-time 0.64 --version legged_gym_fine --save-obs 1 --sim-duration 10
"""

import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import (
  Kuavo42Leggeds2sCfg, Kuavo42LeggedCfg, Kuavo42LeggedSingleObsCfg,
  Kuavo42LeggedFineCfg, Kuavo42LeggedFineObsCfg,
  G1RoughCfg, G1ObsCfg
)
from humanoid.scripts.sim2sim import quaternion_to_euler_array
import cv2, os, time
from pathlib import Path
from datetime import datetime

from humanoid.utils.joystick import JoystickTwistCommand
import imageio

class cmd:
  vx = 0.3
  vy = 0.0
  dyaw = 0.0

def get_obs(data):
  '''Extracts an observation from the mujoco data structure
  Returns:
    [joint_position, joint_velocity, quaternion, 
    base_velocoity, base_ang_vel, projected_gravity]
  '''
  q = data.qpos.astype(np.double)[7:]
  dq = data.qvel.astype(np.double)[6:]
  # quat = data.sensor('BodyQuat').data[[1, 2, 3, 0]].astype(np.double)
  quat = data.qpos[3:7][[1,2,3,0]]
  r = R.from_quat(quat)
  v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
  # omega = data.sensor('BodyGyro').data.astype(np.double)  # base angular-velocity
  omega = data.qvel[3:6]
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
  viewer = mujoco_viewer.MujocoViewer(model, data, width=1280, height=720)

  target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
  action = np.zeros((cfg.env.num_actions), dtype=np.double)

  count_lowlevel = 0

  if args.save_obs:
    obs_memory = []

  if 'legged_gym' in version:
    obs_stack = np.zeros([cfg.env.frame_stack, cfg.env.num_single_obs], np.float32)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)][1:]  # first one is None
    default_joint_pos = np.array([cfg.init_state.default_joint_angles[name] for name in joint_names])
    data.qpos[-cfg.env.num_actions:] = default_joint_pos
    mujoco.mj_step(model, data)
    viewer.render()


  try:
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

      # Obtain an observation
      q, dq, quat, v, omega, gvec = get_obs(data)

      # 1000hz -> 100hz
      if count_lowlevel % cfg.sim_config.decimation == 0:

        if args.joystick:
          cmd.vx = joystick_twist_cmd.x_vel_cmd
          cmd.vy = joystick_twist_cmd.y_vel_cmd
          cmd.dyaw = joystick_twist_cmd.yaw_vel_cmd

        if version == 'isaacsim':
          obs = np.concatenate([
            v, omega, gvec, [cmd.vx, cmd.vy, cmd.dyaw], 
            convert_joint_idx(q, True),
            convert_joint_idx(dq, True),
            convert_joint_idx(action, True)
          ], dtype=np.float32).reshape(1, -1)
        elif 'legged_gym' in version:
          # if hasattr(cfg.rewards, "low_speed_stance"):
          #   cond = np.array(cfg.rewards.low_speed_stance)
          #   phase_mask = ~np.all(np.abs(np.array([cmd.vx, cmd.vy, cmd.dyaw])) < cond)
          phase = count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time
          eu_ang = quaternion_to_euler_array(quat)
          eu_ang[eu_ang > math.pi] -= 2 * math.pi
          if 'obs_v2' in version:
            tmp = np.concatenate([
              [
                math.sin(2 * math.pi * phase),
                math.cos(2 * math.pi * phase),
                cmd.vx * cfg.normalization.obs_scales.lin_vel,
                cmd.vy * cfg.normalization.obs_scales.lin_vel,
                cmd.dyaw * cfg.normalization.obs_scales.ang_vel,
              ],
              (q - default_joint_pos) * cfg.normalization.obs_scales.dof_pos,
              dq * cfg.normalization.obs_scales.dof_vel,
              action,
              omega * cfg.normalization.obs_scales.ang_vel,
              gvec * cfg.normalization.obs_scales.quat
            ], dtype=np.float32).reshape(1, -1)
          else:
            tmp = np.concatenate([
              [
                math.sin(2 * math.pi * phase),
                math.cos(2 * math.pi * phase),
                cmd.vx * cfg.normalization.obs_scales.lin_vel,
                cmd.vy * cfg.normalization.obs_scales.lin_vel,
                cmd.dyaw * cfg.normalization.obs_scales.ang_vel,
              ],
              (q - default_joint_pos) * cfg.normalization.obs_scales.dof_pos,
              dq * cfg.normalization.obs_scales.dof_vel,
              action,
              omega * cfg.normalization.obs_scales.ang_vel,
              eu_ang * cfg.normalization.obs_scales.quat
            ], dtype=np.float32).reshape(1, -1)
          obs_stack = np.concatenate([obs_stack[1:], tmp], axis=0)
          obs = obs_stack.reshape(1, -1)

          obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
          if args.save_obs:
            obs_memory.append(obs)
          # np.save(f'./logs/debug_obs/tmp_obs_{count_lowlevel}.npy', obs)
          # if count_lowlevel > 200: exit()

        # print(f"{v=},\n{omega=},\n{gvec=},\ncmd={[cmd.vx, cmd.vy, cmd.dyaw]},")
        # print(f"q={convert_joint_idx(q, True)},")
        # print(f"dq={convert_joint_idx(dq, True)},")
        # print(f"action={convert_joint_idx(action, True)}")

        action[:] = policy(obs)[0]
        if version == 'isaacsim':
          action = convert_joint_idx(action, False)
        action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        # np.save(f'./logs/debug_action/tmp_action_{count_lowlevel}.npy', action)

        target_q = action * cfg.control.action_scale
        if 'legged_gym' in version:
          target_q = target_q + default_joint_pos


      target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
      # Generate PD control
      tau = pd_control(target_q, q, cfg.robot_config.kps,
              target_dq, dq, cfg.robot_config.kds)  # Calc torques
      tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
      # tau = np.zeros_like(tau)
      # np.save(f'./logs/debug_tau/tmp_tau_{count_lowlevel}.npy', tau)
      data.ctrl = tau

      mujoco.mj_step(model, data)
      viewer.render()
      count_lowlevel += 1
  
  except Exception as e:
    print(e)
  # except Exception as e:
  #   print('Stop by:', e)
  
  if args.save_obs:
    path_save_obs_dir = Path(LEGGED_GYM_ROOT_DIR) / f"logs/save_obs/{Path(args.load_onnx).stem}/"
    path_save_obs_dir.mkdir(exist_ok=True, parents=True)
    obs_memory = np.array(obs_memory, dtype=np.float32)
    print(f"[INFO]: Save obs memory at {path_save_obs_dir}")
    np.save(path_save_obs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_mujoco.npy", obs_memory)

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
  parser.add_argument("--joystick", type=lambda x: x in ['1', 'True', 'true'], default=True)
  parser.add_argument("--benchmark", type=lambda x: x in ['1', 'True', 'true'], default=False)
  parser.add_argument("--save-obs", type=lambda x: x in ['1', 'True', 'true'], default=False)
  parser.add_argument("--sim-duration", type=float, default=60)
  args = parser.parse_args()

  model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/kuavo_s42/mjcf/biped_s42_fixed_arm.xml'
  if args.version == 'isaacsim':
    cfg_class = Kuavo42Leggeds2sCfg
  elif args.version == 'legged_gym':
    cfg_class = Kuavo42LeggedCfg
  elif args.version == 'legged_gym_single_obs':
    cfg_class = Kuavo42LeggedSingleObsCfg
  elif args.version in ['legged_gym_single_obs_g1', 'g1_obs_v2']:
    if args.version == 'legged_gym_single_obs_g1':
      cfg_class = G1RoughCfg
    elif args.version == 'g1_obs_v2':
      cfg_class = G1ObsCfg
    model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/scene.xml'
  elif args.version == 'legged_gym_fine':
    cfg_class = Kuavo42LeggedFineCfg
    model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s42_fine/xml/biped_s42_only_lower_body_scene.xml'
  elif args.version == 'legged_gym_fine_obs_v2':
    cfg_class = Kuavo42LeggedFineObsCfg
    model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/biped_s42_fine/xml/biped_s42_only_lower_body_scene.xml'
  else:
    raise ValueError(f"Don't know version={args.version}")

  class Sim2simCfg(cfg_class):

    class sim_config:
      mujoco_model_path = model_path
      sim_duration = args.sim_duration
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
  if 'g1' in args.version:
    cfg.robot_config.kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.double)
    cfg.robot_config.kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.double)
  if args.joystick:
    joystick_twist_cmd = JoystickTwistCommand(cfg)
  run_mujoco(policy, cfg, args.version)
