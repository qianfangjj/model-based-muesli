# Muesli
基于github上的 [Muzero代码](https://github.com/JimOhman/model-based-rl) ，实现了Muesli。

## 主要修改文件
* config.py 增加muesli相关的超参配置项
* actor.py 训练时actor端直接用policy输出，而不是通过mcts搜索
* evaluate.py 测试时直接使用policy的输出，而不是通过mcts搜索
* game.py 传输的样本中增加policy_logits项，learner端计算loss需要
* replay_buffer.py  传输样本修改
* learners.py 实现muesli的各项loss

## 实验
安装及结果复现，参考原Muzero页面描述: https://github.com/JimOhman/model-based-rl

已使用该版本的Muesli，跑过实验场景: Breakout-ramNoFrameskip-v4
* 环境: DevCloud GPU
* ```python train.py --environment Breakout-ramNoFrameskip-v4 --architecture FCNetwork --num_actors 7 --fixed_temperatures 1.0 0.8 0.7 0.5 0.3 0.2 0.1 --td_steps 10 --window_size 200000 --batch_size 512 --obs_range 0 255 --norm_obs --sticky_actions 4 --noop_reset --episode_life --fire_reset --clip_rewards --group_tag my_group_tag --run_tag my_run_tag --use_gpu_for learner```
* 结果: 见周报


