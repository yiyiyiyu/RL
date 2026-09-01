# 参考
1. [入门](https://imzhanghao.com/2022/02/10/reinforcement-learning/)
2. [动手学强化学习](https://hrl.boyuai.com/)
3. [openAI](https://spinningup.readthedocs.io/zh-cn/latest/user/introduction.html)
4. http://incompleteideas.net/book/RLbook2020.pdf

# 资料
1. UCL David Silver RL Course: https://www.davidsilver.uk/teaching/ 
2. Berkeley Sergey Levine Deep RL Course: http://rail.eecs.berkeley.edu/deeprlcourse/ 
3.  OpenAI DRL Camp: https://sites.google.com/view/deep-rl-bootcamp/lectures 
4.  RL China Camp: http://rlchina.org/
# OpenAI Spinning up 
## 资料
1. [官方文档](https://spinningup.readthedocs.io/zh-cn/latest/user/algorithms.html)
2. [安装方法](https://zhuanlan.zhihu.com/p/472290066)
## 框架

## 使用方法

# 1. 入门
## 1.1 概念：
强化学习（Reinforcement learning，RL）讨论的问题是一个**智能体(agent)** 怎么在一个复杂不确定的 **环境(environment)** 里面去极大化它能获得的奖励。通过感知所处环境的 **状态(state)** 对 **动作(action)** 的 **反应(reward)**， 来指导更好的动作，从而获得最大的 **收益(return)**，这被称为在交互中学习，这样的学习方法就被称作强化学习。

！Reinforcement learning is learning what to do—how to **map situations to actions**——so as to maximize a numerical reward signal. 

![概念图](https://oss.imzhanghao.com/img/202202061348504.png)

- `感知`
智能体在某种程度上感知环境的状态，从而知道自己所处的现状。
- `决策`
智能体根据当前的状态计算出达到目标需要采取的动作的过程叫作决策
- `奖励`
环境根据状态和智能体采取的动作，产生一个标量信号作为奖励反馈。这个标量信号衡量智能体**这一轮**动作的好坏。最大化累积奖励期望是智能体提升策略的目标，也是衡量智能体策略好坏的关键指标。

从以上分析可以看出，面向决策任务的强化学习和面向预测任务的有监督学习在形式上是有不少区别的。首先，决策任务往往涉及多轮交互，即序贯决策；而预测任务总是单轮的独立任务。如果决策也是单轮的，那么它可以转化为“判别最优动作”的预测任务。其次，因为决策任务是多轮的，智能体就需要在每轮做决策时考虑未来环境相应的改变，所以当前轮带来最大奖励反馈的动作，在长期来看并不一定是最优的。
## 1.2 相关术语
1. 状态和观察(states and observations)
 **状态**$s$是一个关于这个世界状态的完整描述。这个世界除了状态以外没有别的信息。
**观察**$o$是对于一个状态的部分描述，可能会漏掉一些信息。
~
在深度强化学习中，一般用  [实数向量、矩阵或者更高阶的张量（tensor）](https://en.wikipedia.org/wiki/Real_coordinate_space)  表示状态和观察。
比如说，视觉上的  **观察**  可以用RGB矩阵的方式表示其像素值；机器人的  **状态**  可以通过关节角度和速度来表示。
如果智能体观察到环境的全部状态，通常说环境是被  **全面观察**  的。如果智能体只能观察到一部分，称之为  **部分观察**。
2. 动作空间(action spaces)
不同的环境有不同的动作。所有有效动作的集合称之为  **动作空间**。有些环境，比如说 Atari 游戏和围棋，属于  **离散动作空间**，这种情况下智能体只能采取有限的动作。其他的一些环境，比如智能体在物理世界中控制机器人，属于  **连续动作空间**。在连续动作空间中，动作是实数向量。
3. 策略(policies)
**策略**是智能体用于决定下一步执行什么行动的规则。
可以是确定性的，一般表示为：$\mu$
$$
a_t=\mu(s_t)
$$
也可以是随机的，一般表示为  $\pi$:
$$
a_t \sim \pi(\cdot | s_t)
$$
因为策略本质上就是智能体的大脑，所以很多时候“策略”和“智能体”这两个名词经常互换，例如我们会说：“策略的目的是最大化奖励”。
在深度强化学习中，我们处理的是参数化的策略，这些策略的输出，依赖于一系列计算函数，而这些函数又依赖于参数（例如神经网络的权重和误差），所以我们可以通过一些优化算法改变智能体的的行为。
经常把这些策略的参数写作  $\theta$  或者  $\phi$  ，然后把它写在策略的下标上来强调两者的联系。
$$
a_t = \mu_{\theta}(s_t) \\
a_t \sim \pi_{\theta}(\cdot | s_t).
$$

4. 行动轨迹(trajectories)
运动轨迹 $\tau$指的是状态和行动的序列。
$$
\tau=(s_0,a_0,s_1,a_1,...)
$$
第一个状态  $s_0$，是从  **开始状态分布**  中随机采样的，有时候表示为  $\rho_0$  :
$$s_0 \sim \rho_0(\cdot)$$
转态转换（从某一状态时间$t$ ,  $s_t$ 到另一状态时间  $t+1$  ,  $s_{t+1}$  会发生什么），是由环境的自然法则确定的，并且只依赖于最近的行动 $a_t$。它们可以是确定性的：
$$s_{t+1} = f(s_t, a_t)$$
而可以是随机的：
$$s_{t+1} \sim P(\cdot|s_t, a_t)$$
智能体的行为由策略确定。
~ 行动轨迹常常也被称作  **回合(episodes)**  或者  **rollouts**。
5. 奖励和回报
6. 不同的回报公式(formulations of return)
7.  强化学习优化问题(the RL optimization problem)
8.  值函数(value functions)

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM3NDI4MjQzXX0=
-->