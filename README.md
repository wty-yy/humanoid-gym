# Installation
1. 创建python3.8 `conda create -n humanoid-gym python=3.8`
2. 参考[pytorch官网](https://pytorch.org/get-started/locally/)安装pytorch-cuda124 `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
3. 安装isaacgym preview 4
  - 下载并解压, 移动到固定位置: https://developer.nvidia.com/isaac-gym
  - 安装`cd isaacgym/python && pip install -e .`
  - 测试用例`cd examples && python 1080_balls_of_solitude.py`
4. 安装humanoid-gym:
  - Clone本仓库，移动到固定位置
  - 安装`cd humanoid-gym && pip install -e .`

# 更新日志
## 2025.3.13.
完成`xbotl_paper_ppo`的全部奖励设计，开始训练，将periodic_force系数下降`1.0->1e-3`

## 2025.3.14.
训练效果为抬腿但是不断抖动，并且进行原地旋转，原地旋转因为`ang_velocity_tracking`过低只有`0.02`，并且通过水平摆腿偷吃`periodic_velocity`奖励

v2: 修改`periodic_velocity`系数为0，已经能够正常走路，但是没有抬脚，发现抬脚高度的mask写反了

v3: 修复抬脚traj的mask错误问题，优化`play.py`没有可视化图表的问题，当前终止验证后会将图表保存在`./figures/{experiment_name}`下

v4:
1. 修复stance_mask错误左右脚顺序的问题;
2. 修复`foot_height_tracking`奖励过大的问题将`w: 5 -> 50`, 加入仅考虑`0.1 < half_phase < 0.4`下的奖励;
3. 修复对`ref_feet_height`和`ref_feet_vel`错误clip的问题(half_phase<0.1就clip了)
4. 将phi指数误差中的平方删除

## 2025.3.15.

v5:
1. 无法学到很高的抬脚奖励，将phi指数误差中的平方加上
2. 功率计算错误，应该是力矩乘以角速度

加入一个新环境`xbotl_simple_reward`
v1: 仅包含基础的4个奖励，测试训练，训练结果高度控制非常低，原地踏步，无法保持站姿

## 2025.3.16.
xbotl_simple_reward_v2: 加入1个保持base高度的奖励，高度保持为0.75，高度计算方法为base到最低脚的距离

## 2025.3.18.
### kuavo42_legged 
加入kuavo42_legged环境: 参考[kuavo-rl-opensource](https://gitee.com/leju-robot/kuavo-rl-opensource/)中的`kuavo-robot-train`配置参数，将kuavo42模型加入，仅训练腿部12个关节，奖励参考humanoid-gym

- `biped_s42_fixed_arm.xml`将14的arm关节换成fixed类型，`biped_s42.xml`中修改内容如下：
  1. 修改`zarm`的`joint`类型
  ```xml
  <joint name="zarm_l1_joint" pos="0 0 0" axis="0 1 0" range="-2.0933 0.5233" actuatorfrcrange="-60 60" damping='0.2'/>
  <!-- 替换为 -->
  <joint name="zarm_l1_joint" pos="0 0 0" type="fixed"/>
  ```
  2. 删除`<actuator>`中`zarm`相关的`motor`
  ```xml
  <!-- 删除 -->
  <motor gear="1" joint="zarm_l1_joint" name="zarm_l1_motor" ctrllimited="true" ctrlrange='-5 5'/>
  ```
  3. 删除`<sensor>`中`zarm`相关的`jointpos`和`jointvel`传感器
  ```xml
  <!-- 删除 -->
  <jointpos name="zarm_l1_pos" joint="zarm_l1_joint"/>
  <jointvel name="zarm_l1_vel" joint="zarm_l1_joint"/>
  ```

参考[`kuavo_s40_config.py`](https://gitee.com/leju-robot/kuavo-rl-opensource/blob/master/kuavo-robot-train/humanoid/envs/custom/kuavo_s40_config.py)和[`kuavo_s42_config.py`](https://gitee.com/leju-robot/kuavo-rl-opensource/blob/master/kuavo-robot-train/humanoid/envs/custom/kuavo_s42_config.py)将配置转入到`kuavo42_legged_config.py`，修改内容如下：
- `Kuavo42LeggedCfg`修改内容如下：
  1. `terrain`没有使用`trimesh`而是`plane`
  2. 所有的`control`和`init_state`都不包含手臂
  3. `domain_rand`使用`humanoid-gym`默认的5个
  4. `commands`中`num_commands`保持4个，没有保持站立的第五维状态
  5. `reward`项和系数保持和`humanoid-gym`一致，计算奖励的常熟使用`kuavo_s42_config.py`例如`base_height_target, min_dist, max_dist, ...`
- `Kuavo42LeggedCfgPPO`修改内容如下：
  1. 仅修改`experiment_name = 'Kuavo42_legged_ppo'`

参考[`kuavo_s40_env.py`](https://gitee.com/leju-robot/kuavo-rl-opensource/blob/master/kuavo-robot-train/humanoid/envs/custom/kuavo_s40_env.py)和[`kuavo_s42_env.py`](https://gitee.com/leju-robot/kuavo-rl-opensource/blob/master/kuavo-robot-train/humanoid/envs/custom/kuavo_s42_env.py)将配置转入到`kuavo42_legged_env.py`，内容和`humanoid_env.py`保持一致，仅修改类名

修改`legged_robot.py`的`_create_env`函数，支持`foot_names`和`knee_names`作为列表传入（在`foot_name = knee_name = None`时使用）

### 修改pd系数
训练kuavo42_legged 1.5h完全无法站住，因此修改PD系数，加入新环境`kuavo42_legged_high_pd_ppo`，有更高的x2后的pd系数

## 2025.3.19.
### 20250319 v2 low pd
1. 将`kuavo42_legged_s2s`配置修改为低pd系数值，加入20250319 isaacsim训练的`models/Isaaclab/v2_20250319_lowpd.onnx`
2. 将`terrain.static_friction`和`terrain.dynamic_friction`从`0.6->1.0`，和isaacsim对应
3. isaaclab2isaacgym迁移基本成功，但单脚跳仍严重，步态奖励还需继续改进
### kuavo42_legged_v2
ghw在isaacsim中可以使用低pd系数训练出可运动的机器人，因此继续调低pd系数下是否可以训练出
1. 发现可能是`collision`奖励系数过低且`dt`过低导致终止`rewards.scales.collision: -1 -> -2000`
2. 将`terrain.static_friction`和`terrain.dynamic_friction`从`0.6->1.0`
3. 将`asset.terminate_after_contacts_on`, `asset.penalize_contacts_on`加入除去脚的所有的关节
