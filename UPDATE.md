# 2025.3.13.
完成`xbotl_paper_ppo`的全部奖励设计，开始训练，将periodic_force系数下降`1.0->1e-3`

# 2025.3.14.
训练效果为抬腿但是不断抖动，并且进行原地旋转，原地旋转因为`ang_velocity_tracking`过低只有`0.02`，并且通过水平摆腿偷吃`periodic_velocity`奖励

v2: 修改`periodic_velocity`系数为0，已经能够正常走路，但是没有抬脚，发现抬脚高度的mask写反了

v3: 修复抬脚traj的mask错误问题，优化`play.py`没有可视化图表的问题，当前终止验证后会将图表保存在`./figures/{experiment_name}`下

v4:
1. 修复stance_mask错误左右脚顺序的问题;
2. 修复`foot_height_tracking`奖励过大的问题将`w: 5 -> 50`, 加入仅考虑`0.1 < half_phase < 0.4`下的奖励;
3. 修复对`ref_feet_height`和`ref_feet_vel`错误clip的问题(half_phase<0.1就clip了)
4. 将phi指数误差中的平方删除

