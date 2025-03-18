# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR
import copy

# import isaacgym
from humanoid.envs import *
from humanoid.utils import get_args, export_policy_as_jit, task_registry, Logger
from humanoid.utils.helpers import get_load_path
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime
import onnxruntime as ort


def get_symm_obs(obs_batch):
    # is_standing = obs_batch.reshape(batch_size, 15, 92)[:, -1, 5].reshape(batch_size)

    batch_size = obs_batch.size(0)
    obs_hist = obs_batch.clone().reshape(batch_size, 15, 92)

    def symm_func(obs_hist, idx, start):
        obs_hist[:, idx, start:start + 12] = torch.roll(obs_hist[:, i, start:start + 12],
                                                        shifts=6,
                                                        dims=1)
        obs_hist[:, idx, [start, start + 1, start + 5, start + 6, start + 7, start + 11]] *= -1
        obs_hist[:, idx, start + 12:start + 26] = torch.roll(obs_hist[:, i, start + 12:start + 26],
                                                             shifts=7,
                                                             dims=1)
        obs_hist[:, idx, [
            start + 13, start + 14, start + 16, start + 18, start + 20, start + 21, start +
            23, start + 25
        ]] *= -1

    for i in range(15):
        obs_hist[:, i, [0, 1, 3, 4]] *= -1
        symm_func(obs_hist, i, 6)
        symm_func(obs_hist, i, 32)
        symm_func(obs_hist, i, 58)
        obs_hist[:, i, [84, 86, 87, 90]] *= -1

    return obs_hist.reshape(batch_size, -1)


def get_symm_action(action_batch):
    res = action_batch.clone()
    start = 0
    res[:, start:start + 12] = torch.roll(res[:, start:start + 12], shifts=6, dims=1)
    res[:, [start, start + 1, start + 5, start + 6, start + 7, start + 11]] *= -1
    res[:, start + 12:start + 26] = torch.roll(res[:, start + 12:start + 26], shifts=7, dims=1)
    res[:, [
        start + 13, start + 14, start + 16, start + 18, start + 20, start + 21, start + 23, start +
        25
    ]] *= -1
    return res


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env,
                                                          name=args.task,
                                                          args=args,
                                                          train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    session = ort.InferenceSession(
        os.path.join(LEGGED_GYM_ROOT_DIR, '../', 'models/v2_2025_3_18.onnx'))
    input_name = session.get_inputs()[0].name

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        model_path = get_load_path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                                train_cfg.runner.experiment_name),
                                   load_run=train_cfg.runner.load_run,
                                   checkpoint=train_cfg.runner.checkpoint)
        onnx_path = model_path.replace('.pt', '.onnx')
        inputs = torch.rand([1, obs.shape[-1]]).to(env.device)
        torch.onnx.export(copy.deepcopy(ppo_runner.alg.actor_critic.actor), inputs, onnx_path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(h1, env.envs[0], body_handle,
                                      gymapi.Transform(camera_offset, camera_rotation),
                                      gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos',
                                      train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir,
                           datetime.now().strftime('%b%d_%H-%M-%S') + args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    for i in tqdm(range(stop_state_log)):

        actions = torch.tensor(session.run(None, {input_name: obs.detach().cpu().numpy()})[0]).to(
            env.device)
        # actions = policy(obs.detach()) # * 0.

        # batch_size = obs.size(0)
        # sin_cos = obs.reshape(batch_size, 15, 92)[:, -1, :2]
        # phase = torch.atan2(sin_cos[:, 0], sin_cos[:, 1]) / (2 * torch.pi)
        # right_idx = (phase > -0.2) & (phase < 0.3)
        #
        # actions[right_idx] = get_symm_action(policy(get_symm_obs(obs)))[right_idx]
        # # print(right_idx)

        if FIX_COMMAND:
            env.commands[:, 0] = 0.6
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.6
            env.commands[:, 4] = 0.
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states({
            'dof_pos_target':
            actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
            'dof_pos':
            env.dof_pos[robot_index, joint_index].item(),
            'dof_vel':
            env.dof_vel[robot_index, joint_index].item(),
            'dof_torque':
            env.torques[robot_index, joint_index].item(),
            'command_x':
            env.commands[robot_index, 0].item(),
            'command_y':
            env.commands[robot_index, 1].item(),
            'command_yaw':
            env.commands[robot_index, 2].item(),
            'base_vel_x':
            env.base_lin_vel[robot_index, 0].item(),
            'base_vel_y':
            env.base_lin_vel[robot_index, 1].item(),
            'base_vel_z':
            env.base_lin_vel[robot_index, 2].item(),
            'base_vel_yaw':
            env.base_ang_vel[robot_index, 2].item(),
            'contact_forces_z':
            env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
        })
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()
    logger.plot_states()

    if RENDER:
        video.release()


if __name__ == '__main__':
    EXPORT_POLICY = False
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
