import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # tqdm是显示循环进度条的库
from env  import  CliffWalkingEnv
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)


#单步Sarsa运行程序
# from sarsa import Sarsa
# np.random.seed(0)

# epsilon = 0.1
# alpha = 0.1
# gamma = 0.9
# agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
# num_episodes = 5000  # 智能体在环境中运行的序列的数量

# return_list = []  # 记录每一条序列的回报
# for i in range(100):  # 显示10个进度条
#     # tqdm的进度条功能
#     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
#             episode_return = 0
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False
#             while not done:
#                 next_state, reward, done = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_return += reward  # 这里回报的计算不进行折扣因子衰减
#                 agent.update(state, action, reward, next_state, next_action)
#                 state = next_state
#                 action = next_action
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
#                 pbar.set_postfix({
#                     'episode':
#                     '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('Sarsa on {}'.format('Cliff Walking'))
# plt.show()

 # 5步Sarsa运行程序
# from n_step_Sarsa import nstep_Sarsa
# np.random.seed(0)
# n_step = 5  # 5步Sarsa算法
# alpha = 0.1
# epsilon = 0.1
# gamma = 0.9
# agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
# num_episodes = 500  # 智能体在环境中运行的序列的数量

# return_list = []  # 记录每一条序列的回报
# for i in range(10):  # 显示10个进度条
#     #tqdm的进度条功能
#     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#         for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
#             episode_return = 0
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False
#             while not done:
#                 next_state, reward, done = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_return += reward  # 这里回报的计算不进行折扣因子衰减
#                 agent.update(state, action, reward, next_state, next_action,
#                              done)
#                 state = next_state
#                 action = next_action
#             return_list.append(episode_return)
#             if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
#                 pbar.set_postfix({
#                     'episode':
#                     '%d' % (num_episodes / 10 * i + i_episode + 1),
#                     'return':
#                     '%.3f' % np.mean(return_list[-10:])
#                 })
#             pbar.update(1)

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
# plt.show()
# Q_learning算法执行程序
from Q_learing import QLearning
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))
plt.show()


#运行打印程序
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    action_meaning = ['^', 'v', '<', '>']
#     print('Sarsa算法最终收敛得到的策略为：')
#     print('5步Sarsa算法最终收敛得到的策略为：')
  
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
