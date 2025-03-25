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

# Run
## Train
```bash
python humanoid/scripts/train.py --task=kuavo42_legged_ppo --run-name v8.3 --headless
# Add iterations (default 3001)
python humanoid/scripts/train.py --task=kuavo42_legged_ppo --run-name v8.3 --max-iterations 10001 --headless
# kuavo42 legged single obs 
python humanoid/scripts/train.py --task=kuavo42_legged_single_obs_ppo --run-name v1.1 --max-iterations 3001 --headless
```
## Play
```bash
# Load last pt model params:
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run_name v4
# Load onnx model:
python humanoid/scripts/play.py --load-onnx models/Isaaclab/v2_20250319_lowpd.onnx --task=kuavo42_legged_s2s_ppo --run-name v1
# Load torch.jit model:
python humanoid/scripts/play.py --load-jit models/XBot_ppo/jit_policy_example.pt --task=humanoid_ppo --run-name v1
# kuavo42 legged ppo
# Use joystick to control:
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run-name v8 --fix-command 0 --cycle-time 1.2
# v8.1
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run-name v8.1 --fix-command 0 --cycle-time 0.9
# v8.2
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --run-name v8.2 --fix-command 1 --cycle-time 0.64
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --load-onnx models/kuavo42_legged/Kuavo42_legged_ppo_v8.2_model_3001.onnx --fix-command 1 --cycle-time 0.64 --run-name v8.2
# v8.3
python humanoid/scripts/play.py --task=kuavo42_legged_ppo --load-onnx models/kuavo42_legged/Kuavo42_legged_ppo_v8.3_model_10001.onnx --fix-command 0 --cycle-time 0.64 --run-name v8.3
# kuavo42 legged ppo single obs
# v1
python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1 --fix-command 0 --cycle-time 0.64 --load-onnx models/kuavo42_legged/Kuavo42_legged_single_obs_ppo_v1_model_3001.onnx
```

# Fixed Bugs
1. `humanoid/utils/helpers.py`中`update_cfg_from_args`中的`args.seed`应该对`env_cfg.seed`进行更新而非`train_cfg.seed`，否则无法在`set_seed`中正确使用新种子

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
### kuavo42_legged_s2s v2 low pd
1. 将`kuavo42_legged_s2s`配置修改为低pd系数值，加入20250319 isaacsim训练的`models/Isaaclab/v2_20250319_lowpd.onnx`
2. 将`terrain.static_friction`和`terrain.dynamic_friction`从`0.6->1.0`，和isaacsim对应
3. isaaclab2isaacgym迁移基本成功，但单脚跳仍严重，步态奖励还需继续改进
### kuavo42_legged v2
ghw在isaacsim中可以使用低pd系数训练出可运动的机器人，因此继续调低pd系数下是否可以训练出
1. 发现可能是`collision`奖励系数过低且`dt`过低导致终止`rewards.scales.collision: -1 -> -2000`
2. 将`terrain.static_friction`和`terrain.dynamic_friction`从`0.6->1.0`
3. 将`asset.terminate_after_contacts_on`, `asset.penalize_contacts_on`加入除去脚的所有的关节


## 2025.3.20.
1. 加入`kuavo42_legged_simple_reward_ppo`环境，使用4个简易奖励函数在kuavo42上进行训练，训练出来的模型能稳定站住，但学不会抬脚

### kuavo42_legged_v3
1. 修复模型的抬脚奖励错误，左右脚是平移得到的，而非xbotl中的对称，并且默认的关节不是0，所以现在考虑从default关节出发，加上delta的角度得到目标关节角度

### kuavo42_legged_v4
1. 修复v3中的抬脚奖励错误，错误将默认关节索引给错，`torch.where`的条件结果写反

## 2025.3.21.
1. 测试sim2sim.py代码到mujoco中，但是原版的humanoidgym都无法成功迁移，检查问题
2. 将play_onnx的读取onnx功能合并到play代码中，logger支持torque绘制
3. 发现biped_s42_fixed_arm.xml中基础base高度过低的问题（机器人会从地里面卡出来）`z: 0.82->0.9`，sim2sim_leju.py代码能够将isaacsim的onnx模型迁移过去

### kuavo42_legged_v5
1. 向logger_legged_info.py中添加torque打印信息，发现踝关节的力矩相比官方模型给出的非常小，尝试
  - 增大`target_joints_delta: [-0.2, 0.3, -0.1] -> [-0.3, 0.6, -0.3]`
  - 启动`use_ref_actions = True`进行引导
  - 增大`action_scale: 0.25 -> 0.5`
  - 增大`cycle_time: 0.97 -> 1.20`
  - 减小目标抬脚高度`target_feet_height: 0.12 -> 0.06`

### kuavo42_legged_v6
v5训练会导致迅速收腿中止，发现可能是因为错误的中止条件
- 修改中止条件为base高度小于0.3，删除其他部位受到力中止的判断
- 删除collision奖励，`-2000 -> 0`
- 加入termination奖励`-200`，当高度意外中止时奖励为`-2`
- `use_ref_actions = False`关闭引导

### kuavo42_legged_v7
依旧没有抬腿，joint_pos奖励非常小，接近-0.1，尝试增大系数
- 增大`joint_pos: 1.6 -> 6.4`

### kuavo42_legged_v8
尝试修改ppo配置（参考isaacsim）
- `learning_rate: 1e-5 -> 1e-3`
- `num_learning_epochs: 2 -> 5`
- `gamma: 0.994 -> 0.99`
- `lambda: 0.9 -> 0.95`

v8成功✌训练出抬脚，但是由于joint_pos奖励过大，对速度追踪不是很正确，而且抬脚周期过长

## 2025.3.22.
1. 在`logger_legged_info`中加入`feet_height`的图像绘制
### kuavo42_legged_v9
- 减小`joint_pos: 6.4 -> 1.6`
- 减小抬脚周期`cycle_time: 1.2 -> 0.64`
- 提高`tracking_lin_vel: 1.2 -> 1.6`
v9训练结果不好，没有学到抬脚，可能是奖励给的太小导致
### kuavo42_legged_v8.1
- 增大`joint_pos: 1.6 -> 3.2`
- 增大抬脚周期`cycle_time: 0.64 -> 0.9`
### kuavo42_legged_v8.2
- 减小抬脚周期`cycle_time: 0.9 -> 0.64`

## 2025.3.23.
1. v8.2的效果已经很不错了，稍微抬脚高度有点高
2. 对`play.py`的手柄输出指令范围进行稍微限制
### kuavo42_legged_v8.3
- 提高`base_height_target: 0.795 -> 0.85`到正确高度上
- 稍微降低`target_joints_delta: [-0.3, 0.6, 0.3] -> [-0.25, 0.5, -0.25]`
- 尝试进行一次长时间训练`max_iterations: 3001 -> 10001`，从CLI中添加

## 2025.3.24.
1. v8.3训练效果也不错，长时间训练已经收敛没出现崩溃的问题，除了抬腿会在空中滞空比较长的一段时间
2. 稍微调整play时的初始相机角度，调整
### kuavo42_legged_single_obs_v1
加入`kuavo42_legged_single_obs`环境，仅修改`frame_stack: 15 -> 1`

## 2025.3.25.
1. 将训练完成的`kuavo42_legged_singe_obs`模型存入`models/kuavo42_legged`
### Fixed Bugs
1. `humanoid/utils/helpers.py`中`update_cfg_from_args`中的`args.seed`应该对`env_cfg.seed`进行更新而非`train_cfg.seed`，否则无法在`set_seed`中正确使用新种子
### kuavo42_legged_single_obs_v1.1
single obs v1就是容易在y,yaw同时较大时踩到脚导致不稳定摔倒
1. 加入`prob_high_lin_y_and_yaw=0.05`随机化概率，若触发，随机将`lin_y`和`yaw`同时拉满到最大或最小值
