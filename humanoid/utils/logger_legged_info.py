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
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from humanoid import LEGGED_GYM_ROOT_DIR

class Logger:
    def __init__(self, dt, experiment_name, run_name, plot_period):
        figure_dir = Path(LEGGED_GYM_ROOT_DIR) / 'figures' / experiment_name / run_name
        figure_dir.mkdir(exist_ok=True, parents=True)
        self.path_figure = figure_dir / (time.strftime('%b%d_%H-%M-%S') + run_name + "legged_info" + '.png')
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_period = plot_period
        self.plot_process = None
        # self.start_plot_process(self)

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self.plot)
        self.plot_process.start()

    def plot(self):
        nb_rows = 5
        nb_cols = 4
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(25, 15))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log= self.state_log
        # left leg
        for i, name in enumerate(['leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint']):
            ax = axs[1, i]
            ax.plot(time, np.stack(log['dof_pos'], axis=0)[:, i+2], label='measured')
            ax.plot(time, np.stack(log['dof_pos_target'], axis=0)[:, i+2], label='target')
            ax.plot(time, np.stack(log['dof_pos_ref'], axis=0)[:, i+2], label='ref')
            ax.set(xlabel='time [s]', ylabel='Position [rad]', title=f'{name} Position')
            ax.legend()
        for i, name in enumerate(['leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint']):
            ax = axs[2, i]
            ax.plot(time, np.stack(log['dof_torque'], axis=0)[:, i+2])
            ax.set(xlabel='time [s]', ylabel='Torque [N/m]', title=f'{name} Torque')
        for i, name in enumerate(['leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint']):
            ax = axs[3, i]
            ax.plot(time, np.stack(log['dof_pos'], axis=0)[:, i+8], label='measured')
            ax.plot(time, np.stack(log['dof_pos_target'], axis=0)[:, i+8], label='target')
            ax.plot(time, np.stack(log['dof_pos_ref'], axis=0)[:, i+8], label='ref')
            ax.set(xlabel='time [s]', ylabel='Position [rad]', title=f'{name} Position')
            ax.legend()
        for i, name in enumerate(['leg_r3_joint', 'leg_r4_joint', 'leg_r5_joint']):
            ax = axs[4, i]
            ax.plot(time, np.stack(log['dof_torque'], axis=0)[:, i+8])
            ax.set(xlabel='time [s]', ylabel='Torque [N/m]', title=f'{name} Torque')
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot feet height
        a = axs[0, 3]
        a.plot(time, np.stack(log["feet_height"])[:, 0], label='left')
        a.plot(time, np.stack(log["feet_height"])[:, 1], label='right')
        a.set(xlabel='time [s]', ylabel='height [m]', title='Feet height')
        a.legend()
        fig.tight_layout()
        plt.savefig(self.path_figure, dpi=100)
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()