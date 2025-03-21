"""
Load last pt model params:
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run_name v4
Load onnx model:
python humanoid/scripts/play.py --load-onnx models/Isaaclab/v2_20250319_lowpd.onnx --task=kuavo42_legged_s2s_ppo --run_name v1
Load torch.jit model:
python humanoid/scripts/play.py --load-jit models/XBot_ppo/jit_policy_example.pt --task=humanoid_ppo --run_name v1
Use joystick to control:
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run_name v8 --fix-command 0
"""

import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from humanoid.utils.logger_legged_info import Logger as LoggerLeggedInfo
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import pygame
from threading import Thread


x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
joystick_use = True
joystick_opened = False

if joystick_use:
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")
    # joystick thread exit flag
    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            # get joystick input
            pygame.event.get()
            # update robot command
            x_vel_cmd = -joystick.get_axis(1) * 1
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 1
            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

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
    env_cfg.domain_rand.push_robots = False 
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
    if args.load_onnx is not None:
        import onnxruntime as ort
        session = ort.InferenceSession(args.load_onnx)
        input_name = session.get_inputs()[0].name
        policy = lambda x: torch.tensor(session.run(
            None, {input_name: x.detach().cpu().numpy()}
        )[0]).to(env.device)
    elif args.load_jit is not None:
        policy = torch.jit.load(args.load_jit, map_location=env.device)
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY and args.load_onnx is None and args.load_jit is None:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt, train_cfg.runner.experiment_name, args.run_name, 1)
    logger_legged_info = LoggerLeggedInfo(env.dt, train_cfg.runner.experiment_name, args.run_name, 1)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1200 # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name, args.run_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        print(dir)
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))

    # for i in tqdm(range(stop_state_log)):
    try:
        while 1:

            actions = policy(obs.detach()).detach() # * 0.
            
            if args.fix_command:
                env.commands[:, 0] = 0.
                env.commands[:, 1] = 0.
                env.commands[:, 2] = 0.5
                env.commands[:, 3] = 0.
            else:
                env.commands[:, 0] = x_vel_cmd
                env.commands[:, 1] = y_vel_cmd
                env.commands[:, 2] = 0.
                env.commands[:, 3] = yaw_vel_cmd
            
            print(f"[DEBUG]: command={env.commands}")
            
                
                

            obs, critic_obs, rews, dones, infos = env.step(actions)

            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
                img = np.reshape(img, (1080, 1920, 4))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img[..., :3])

            logger_legged_info.log_states({
                'dof_pos_target': actions[robot_index, :].cpu().numpy() * env.cfg.control.action_scale + env.default_dof_pos[0].cpu().numpy(),
                'dof_pos': env.dof_pos[robot_index, :].cpu().numpy(),
                'dof_pos_ref': env.ref_dof_pos[robot_index, :].detach().cpu().numpy(),
                'dof_torque': env.torques[robot_index, :].cpu().numpy(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'feet_height': env.feet_height[robot_index].cpu().numpy()
            })
            # ====================== Log states ======================
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
    except KeyboardInterrupt:
        pass
    logger_legged_info.plot()

    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    args = get_args(extra_parameters=[
        {
            "name": "--load-onnx",
            "type": str,
            "default": None,
            "help": "Path of onnx model actor model",
        },
        {
            "name": "--load-jit",
            "type": str,
            "default": None,
            "help": "Path of torch.jit model actor model",
        },
        {
            "name": "--fix-command",
            "type": lambda x: x in ['1', 'true', 'True'],
            "default": True,
            "help": "Use fix command or joystick command input",
        }
    ])
    play(args)
