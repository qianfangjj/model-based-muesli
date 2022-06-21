# Muesli
基于github上的 [Muzero代码](https://github.com/JimOhman/model-based-rl) ，实现了Muesli。

## 主要修改及涉及文件
* action选择方式修改: 
  * muesli在训练和测试时，直接使用policy输出的action，而不是通过mcts搜索
  * 涉及文件:
    * actor.py 
    ```bash
    97       # self.play_game(game)
    98       self.play_game_use_policy(game)
    ```
    * evaluate.py
    ```bash
    602   with torch.inference_mode():
    603     # game = evaluator.play_game(environment)
    604     game = evaluator.play_game_muesli(environment)
    ```
* 数据流修改:
  * policy_logits来源: child_visits -> policy net output
  * 增加oberservation: s_t,....,s_t+K
  * 涉及文件:
    * game.py
    * replay_buffer.py
* loss相关修改:
  * 增加target network
  * 增加policy gradient loss，cmpo loss
  * 文件:
    * learners.py
    ```bash
     135         batch = ray.get(ready_batches[0])
     136         # self.update_weights(batch)
     137         # muesli 训练入口
     138         self.update_weights_muesli(batch)
     139         self.training_step += 1
     140         # soft update target network params
     141         for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
     142           target_param.data.copy_(
     143             target_param.data * (1.0 - self.config.alpha_target) + param.data * self.config.alpha_target
     144           )
    ```
* 配置增加:
  * config.py 增加muesli相关的超参配置项
  ```bash
  227   ### Muesli
  228   muesli = parser.add_argument_group('muesli')
  229   muesli.add_argument('--threshold_c', type=float, default=1.0)
  230   muesli.add_argument('--simulation_depth', type=int, default=1)
  231   muesli.add_argument('--clip_param', type=float, default=0.1)
  232   muesli.add_argument('--num_sampled_actions', type=int, default=16)
  233   muesli.add_argument('--alpha_target', type=float, default=0.1)
  ```


## 实验
安装及结果复现，参考原Muzero页面描述: https://github.com/JimOhman/model-based-rl

已使用该版本的Muesli，跑过实验场景: Breakout-ramNoFrameskip-v4
* 环境: DevCloud GPU
* ```python train.py --environment Breakout-ramNoFrameskip-v4 --architecture FCNetwork --num_actors 7 --fixed_temperatures 1.0 0.8 0.7 0.5 0.3 0.2 0.1 --td_steps 10 --window_size 200000 --batch_size 512 --obs_range 0 255 --norm_obs --sticky_actions 4 --noop_reset --episode_life --fire_reset --clip_rewards --group_tag my_group_tag --run_tag my_run_tag --use_gpu_for learner```
* 结果: 见周报


