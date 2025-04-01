# -*- coding: utf-8 -*-
'''
@File    : compare_sim2sim_obs_dist.py
@Time    : 2025/03/31 15:28:01
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : 计算两个仿真器下obs的每个维度的JS散度并绘图
'''

if __name__ == '__main__':
  pass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from humanoid import LEGGED_GYM_ROOT_DIR
from pathlib import Path
import time
from scipy.stats import entropy
from tqdm import tqdm

path_figures_dir = Path(LEGGED_GYM_ROOT_DIR) / "figures/compare_obs"
path_figures_dir.mkdir(exist_ok=True, parents=True)

paths_npy = [
  "/home/yy/Coding/robotics/humanoid-gym/logs/save_obs/Kuavo42_legged_fine_ppo_v1.1_model_3001/20250331_143857_mujoco_xlin=0.3.npy",
  "/home/yy/Coding/robotics/humanoid-gym/logs/save_obs/Kuavo42_legged_fine_ppo_v1.1_model_3001/20250331_153846_isaacgym_xlin=0.3.npy"
]
names = [
  "mujoco", "isaacgym"
]
joints = [f'leg_{i}_{j+1}' for i in ['l', 'r'] for j in range(6)]
obs_dim_names = [
  'sin', 'cos', 'vel_x', 'vel_y', 'vel_yaw',
  *[f'q_{x}' for x in joints],
  *[f'dq_{x}' for x in joints],
  *[f'a_{x}' for x in joints],
  'ang_vel_x', 'ang_vel_y', 'ang_vel_yaw',
  'euler_x', 'euler_y', 'euler_z'
]

def jensen_shannon_divergence(p, q):
  p += 1e-10  # 避免log(0)
  q += 1e-10
  p /= p.sum()  # 归一化
  q /= q.sum()
  m = 0.5 * (p + q)
  return 0.5 * (entropy(p, m) + entropy(q, m))

class CompareTool:
  def __init__(self, paths_npy=paths_npy, names=names):
    self.obs, self.obs_shape, self.names = [], None, names
    for p in paths_npy:
      obs = np.load(p)
      obs = obs.reshape(obs.shape[0], -1).T
      if self.obs_shape is None:
        self.obs_shape = obs.shape
      assert self.obs_shape == obs.shape, f"path={p} {obs.shape=} diff with first {self.obs_shape=}"
      self.obs.append(obs)  # (obs_dim, N)
    self.obs = np.array(self.obs, dtype=np.float32)
  
  def compare(self, ncol=8, pixel=5):
    nrow = int(np.ceil(self.obs_shape[0]/ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(nrow*pixel, ncol*pixel//2))
    for i in tqdm(range(self.obs_shape[0])):
      data1, data2 = self.obs[:, i]
      bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 50)
      hist1, _ = np.histogram(data1, bins=bins, density=True)
      hist2, _ = np.histogram(data2, bins=bins, density=True)
      js_div = jensen_shannon_divergence(hist1, hist2)

      ax = axs[i//ncol, i%ncol]
      ax.hist(data1, bins=bins, alpha=0.5, density=True, label=self.names[0])
      ax.hist(data2, bins=bins, alpha=0.5, density=True, label=self.names[1])
      ax.set_title(f"{obs_dim_names[i]} js={js_div:.4f}")
      ax.legend()
    fig.tight_layout()
    fig.savefig(path_figures_dir / time.strftime("%Y%m%d_%M%H%S.png"), dpi=100)
    plt.show()

if __name__ == '__main__':
  tool = CompareTool()
  tool.compare()
