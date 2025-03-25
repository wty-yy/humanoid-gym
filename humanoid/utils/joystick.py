# -*- coding: utf-8 -*-
'''
@File    : joystick.py
@Time    : 2025/03/25 16:44:21
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Joystick for twist command
@Ref     : https://github.com/AgibotTech/agibot_x1_train
'''
import pygame
import numpy as np
from threading import Thread
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

class JoystickTwistCommand:
  def __init__(self, env_cfg: LeggedRobotCfg):
    self.env_cfg = env_cfg
    self.x_vel_cmd = self.y_vel_cmd = self.yaw_vel_cmd = 0.0
    self.joystick_opened = False

    pygame.init()
    try:
      # get joystick
      self.joystick = pygame.joystick.Joystick(0)
      self.joystick.init()
      self.joystick_opened = True
    except Exception as e:
        print(f"Can't open joystick: {e}")
    # joystick thread exit flag
    self.exit_flag = False

    if self.joystick_opened:
      joystick_thread = Thread(target=self.handle_joystick_input, daemon=True)
      joystick_thread.start()
  
  @staticmethod
  def scale_range(x, range):
    sz = range[1] - range[0]
    if range[0] >= 0:
      return range[0] if x < 0 else x * sz + range[0]
    elif range[1] <= 0:
      return range[1] if x > 0 else x * sz + range[1]
    return x * (-range[0] if x < 0 else range[1])

  def handle_joystick_input(self):
    while not self.exit_flag:
      # get joystick input
      pygame.event.get()
      # update robot command
      self.x_vel_cmd = self.scale_range(
        -self.joystick.get_axis(1),
        self.env_cfg.commands.ranges.lin_vel_x
      )
      self.y_vel_cmd = self.scale_range(
        -self.joystick.get_axis(0),
        self.env_cfg.commands.ranges.lin_vel_y
      )
      self.yaw_vel_cmd = self.scale_range(
        -self.joystick.get_axis(3),
        self.env_cfg.commands.ranges.ang_vel_yaw
      )
      # print("[DEBUG] twist cmd =", self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd)
      pygame.time.delay(100)
  
  def get_twist_cmd(self):
    return self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd

if __name__ == '__main__':
  env_cfg = LeggedRobotCfg()
  env_cfg.commands.ranges.lin_vel_x = [-0.2, 0.8]
  env_cfg.commands.ranges.lin_vel_y = [-0.4, -0.1]
  env_cfg.commands.ranges.ang_vel_yaw = [0.2, 0.5]
  joystick_twist_cmd = JoystickTwistCommand(env_cfg)
  while not joystick_twist_cmd.exit_flag and joystick_twist_cmd.joystick_opened:
    print(joystick_twist_cmd.get_twist_cmd())
