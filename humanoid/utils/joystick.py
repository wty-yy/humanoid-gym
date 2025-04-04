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

JOYSTICK_INDEX_MAPPING = {
  # axis
  'lin_vel_x': 1,  # left stick up and down
  'lin_vel_y': 0,  # left stick left and right
  'ang_vel_yaw': 3,  # right stick left and right
  # button
  'stand_switch': 0  # right down button, 'A' for Xbox
}

class JoystickCommand:
  def __init__(self, env_cfg: LeggedRobotCfg):
    self.env_cfg = env_cfg
    joystick_thread = Thread(target=self.handle_joystick_input, daemon=True)
    joystick_thread.start()
    self.exit_flag = False
  
  def init_joystick(self):
    """ init pygame must be in thread handle function """
    self.x_vel_cmd = self.y_vel_cmd = self.yaw_vel_cmd = 0.0
    self.stand_cmd = 1  # default stand
    self.last_stand_btn_state = 0
    pygame.init()
    try:
      # get joystick
      self.joystick = pygame.joystick.Joystick(0)
      self.joystick.init()
    except Exception as e:
        print(f"Can't open joystick: {e}")
    # joystick thread exit flag
  
  @staticmethod
  def scale_range(x, range):
    sz = range[1] - range[0]
    if range[0] >= 0:
      return range[0] if x < 0 else x * sz + range[0]
    elif range[1] <= 0:
      return range[1] if x > 0 else x * sz + range[1]
    return x * (-range[0] if x < 0 else range[1])

  def handle_joystick_input(self):
    self.init_joystick()

    while not self.exit_flag:
      # get joystick input
      pygame.event.get()
      # update robot command
      map = JOYSTICK_INDEX_MAPPING
      self.x_vel_cmd = self.scale_range(
        -self.joystick.get_axis(map['lin_vel_x']),
        self.env_cfg.commands.ranges.lin_vel_x
      )
      self.y_vel_cmd = self.scale_range(
        -self.joystick.get_axis(map['lin_vel_y']),
        self.env_cfg.commands.ranges.lin_vel_y
      )
      self.yaw_vel_cmd = self.scale_range(
        -self.joystick.get_axis(map['ang_vel_yaw']),
        self.env_cfg.commands.ranges.ang_vel_yaw
      )
      self.stand_btn_state = self.joystick.get_button(map['stand_switch'])
      if self.last_stand_btn_state and self.stand_btn_state == 0:
        self.stand_cmd ^= 1
      self.last_stand_btn_state = self.stand_btn_state
      # self.debug()
      # print("[DEBUG] twist cmd =", self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd, self.stand_cmd)
      pygame.time.delay(100)
  
  def debug(self):
    axes = [self.joystick.get_axis(a) for a in range(self.joystick.get_numaxes())]
    print(f"Joystick Axes: {axes}")
    buttons = [self.joystick.get_button(b) for b in range(self.joystick.get_numbuttons())]
    print(f"Joystick Buttons: {buttons}")
    hats = [self.joystick.get_hat(h) for h in range(self.joystick.get_numhats())]
    print(f"Joystick Hats: {hats}")
  
  def get_twist_cmd(self):
    return self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd
  
  def get_twist_and_stand_cmd(self):
    if self.stand_cmd:
      self.x_vel_cmd = self.y_vel_cmd = self.yaw_vel_cmd = 0.0
    return self.x_vel_cmd, self.y_vel_cmd, self.yaw_vel_cmd, self.stand_cmd
  
  def close(self):
    self.exit_flag = True

if __name__ == '__main__':
  env_cfg = LeggedRobotCfg()
  env_cfg.commands.ranges.lin_vel_x = [-0.2, 0.8]
  env_cfg.commands.ranges.lin_vel_y = [-0.4, -0.1]
  env_cfg.commands.ranges.ang_vel_yaw = [0.2, 0.5]
  joystick_cmd = JoystickCommand(env_cfg)
  try:
    while not joystick_cmd.exit_flag:
      print(joystick_cmd.get_twist_cmd())
  except KeyboardInterrupt:
    joystick_cmd.close()
