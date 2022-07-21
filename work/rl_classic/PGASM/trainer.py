import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
from shutil import copy
from matplotlib import pyplot as plt


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    copy(src_dir, tar_dir)
    env = 'envs/Grid_envs.py'
    copy(env, tar_dir)
    policies = 'agents/policies.py'
    copy(policies, tar_dir)
    models = 'agents/models.py'
    copy(models, tar_dir)
    main = 'main.py'
    copy(main, tar_dir)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


# 初始化 创建新文件夹，存放相关文件
def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


# 初始化日志，日志格式
def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


# 计步器类 记录模型训练的步数
class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return self.cur_step % self.log_step == 0

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


# 模型训练类
class Trainer:
    def __init__(self, env, model, global_counter, summary_writer, output_path=None, model_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        assert self.env.T % self.n_step == 0
        self.data = []
        self.episode_rewards = [0]
        self.output_path = output_path
        self.model_path = model_path
        self.env.train_mode = True

    # 训练可视化 增加训练的奖励数据
    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            self.summary_writer.add_scalar('train_reward', reward, global_step=global_step)
        else:
            self.summary_writer.add_scalar('test_reward', reward, global_step=global_step)

    # 根据当前的状态，获取模型的策略和动作
    def _get_policy(self, ob, done, mode='train'):
        if self.agent.startswith('ma2c'):
            self.ps = self.env.get_fingerprint()
            policy = self.model.forward(ob, done, self.ps)
        else:
            policy = self.model.forward(ob, done)
        action = []
        for pi in policy:
            if mode == 'train':
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(pi))
        return policy, np.array(action)

    # 根据当前的动作，获取相对应的值
    def _get_value(self, ob, done, action):
        if self.agent.startswith('ma2c'):
            value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
        else:
            # 获取当前动作的邻近动作
            self.naction = self.env.get_neighbor_action(action)  # action=[2,3,2,2]; nactions =[[3], [2,2], [3,2], [2]]
            if not self.naction:
                self.naction = np.nan
            value = self.model.forward(ob, done, self.naction, 'v')
        return value

    # 日志打印，模型结果动态保存，训练可视化
    def _log_episode(self, global_step, mean_reward, std_reward):
        log = {'agent': self.agent,
               'step': global_step,
               'test_id': -1,
               'avg_reward': mean_reward,
               'std_reward': std_reward}
        # 模型结果动态保存
        with open(self.output_path + 'train_reward_log.csv', 'a') as f:
            f.write('{},{},{},{},{}\n'.format(self.agent, global_step, -1, mean_reward, std_reward))
        self.data.append(log)
        # 训练可视化
        self._add_summary(mean_reward, global_step)
        self.summary_writer.flush()

    # 探索
    def explore(self, prev_ob, prev_done):
        # run a batch of steps
        ob = prev_ob
        done = prev_done
        # 默认20步为一次训练 n_step
        for _ in range(self.n_step):
            # 根据当前的状态，获取模型的策略和动作
            policy, action = self._get_policy(ob, done)
            # 决策后，根据当前的动作，获取相对应的值
            value = self._get_value(ob, done, action)
            # 更新当前环境中的策略
            self.env.update_fingerprint(policy)
            # 根据当前动作，获取环境返回的 下一时刻的状态，当前动作的奖励，是否满足结束条件，当前动作的全部奖励
            next_ob, reward, done, global_reward = self.env.step(action)
            # 计算总的奖励
            self.episode_rewards[-1] += global_reward
            self.global_counter.next()
            self.cur_step += 1
            # 收集经验，保存此次训练的状态，动作，奖励。作为模型训练样本
            if self.agent.startswith('ma2c'):
                self.model.add_transition(ob, self.ps, action, reward, value, done)
            else:
                self.model.add_transition(ob, self.naction, action, reward, value, done)
            if done:
                break
            ob = next_ob
        if done:
            R = np.zeros(self.model.n_agent)
        else:
            _, action = self._get_policy(ob, done)
            R = self._get_value(ob, done, action)
        return ob, done, R

    def perform(self, test_ind):
        # do a test
        ob = self.env.reset(test_ind=test_ind)
        rewards = []
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        while True:
            if self.agent == 'greedy':
                action = self.model.forward(ob)
            else:
                policy, action = self._get_policy(ob, done, mode='test')
                self.env.update_fingerprint(policy)
            next_ob, reward, done, global_reward = self.env.step(action, mode='train')  # with disturbance
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run(self):
        # 初始化训练结果保存文件
        with open(self.output_path + 'train_reward_log.csv', 'w') as f:
            f.write('agent,step,test_id,avg_reward,std_reward\n')
        # 判断训练是否到达停止条件
        while not self.global_counter.should_stop():
            # 环境初始化，重置
            ob = self.env.reset()
            # 请注意，这样做是重置 LSTM 状态的预先决定！
            done = True
            # 模型重置
            self.model.reset()
            self.cur_step = 0
            while True:
                # 通过当前状态，通过算法模型获取下一次的状态以及奖励
                ob, done, R = self.explore(ob, done)
                dt = self.env.T - self.cur_step
                global_step = self.global_counter.cur_step
                # 模型训练
                self.model.backward(R, dt, self.summary_writer, global_step)
                # 结束
                if done:
                    if self.global_counter.should_log():
                        # logging
                        mean_reward = round(np.mean(self.episode_rewards[-101:-1]) / self.env.T, 4)
                        logging.info('''Training: global step %d, episode step %d,train r: %.4f, done: %r''' %
                                     (global_step, len(self.episode_rewards), mean_reward, done))
                        np.save('{}'.format('states'), np.array(self.env.states))
                        # 当第一次达到正常电压时，保存每个时期的步数计数器
                        np.save(self.output_path + '{}'.format('step_count'), self.env.step_list)
                        np.save(self.output_path + '{}'.format('episode_rewards'), self.episode_rewards)
                        # 保存模型
                        # self.model.save(self.model_path, global_step)
                    self.env.terminate()
                    self.episode_rewards.append(0)
                    break
            # 计算平均奖励
            rewards = np.array(self.episode_rewards[-101:-1]) / self.env.T
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            # 保存日志
            self._log_episode(global_step, mean_reward, std_reward)
        # 训练数据转化为DataFrame
        df = pd.DataFrame(self.data)
        # 训练平均奖励结果，画图
        plt.figure()
        plt.plot(df['avg_reward'], 'b-.', label='avg_reward')
        plt.title('PGSIM avg_reward')
        plt.legend(), plt.savefig(
            self.output_path + '/avg_reward={:.4f}.png'.format(self.data[-1]['avg_reward'])), plt.show()
        # 保存训练数据
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def run_offline(self):
        self.env.cur_episode = 0
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Evaluator(Tester):
    def __init__(self, env, model, output_path):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path

    def run(self):
        self.env.cur_episode = 0
        time.sleep(1)
        rewards = 0
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind)
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            rewards += reward
            time.sleep(0.5)
            self.env.output_data(self.output_path, test_ind)
        print('Testing finished, avg reward %.2f' % (rewards / self.test_num))
