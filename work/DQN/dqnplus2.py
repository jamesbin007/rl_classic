# 1.4，图片输入
# 图片输入效果不是很好，因此大部分都是使用的非图片输入。

import collections

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# env = gym.make('CartPole-v0').unwrapped
from tqdm import tqdm

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return transitions

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, h, w, outputs):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN:
    def __init__(self, gamma, epsilon):
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.init_screen = self.get_screen()
        _, _, self.screen_height, self.screen_width = self.init_screen.shape
        self.n_actions = self.env.action_space.n
        self.policy_net = Qnet(self.screen_height, self.screen_width, self.n_actions).to(device)
        self.target_net = Qnet(self.screen_height, self.screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略

    def take_action(self, state):
        global steps_done
        sample = random.random()
        steps_done += 1
        if sample > self.epsilon:  # epsilon-贪婪策略采取动作
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def update(self, transitions):
        batch = Transition(*zip(*transitions))
        # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # 计算Q(s_t, a)-模型计算 Q(s_t)，然后选择所采取行动的列。
        # 这些是根据策略网络对每个批处理状态所采取的操作。
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # 计算下一个状态的V(s_{t+1})。
        # 非最终状态下一个状态的预期操作值是基于“旧”目标网络计算的；选择max(1)[0]的最佳奖励。
        # 这是基于掩码合并的，这样当状态为最终状态时，我们将获得预期状态值或0。
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 计算期望Q值
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # 计算Huber损失
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
        screen = screen[:, :, slice_range]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        return resize(screen).unsqueeze(0).to(device)


gamma = 0.999
epsilon = 0.9
steps_done = 0
num_episodes = 1000
minimal_size = 100
batch_size = 32
replay_buffer = ReplayBuffer(10000)
agent = DQN(gamma, epsilon)
return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            done = False
            episode_return = 0
            agent.env.reset()
            last_screen = agent.get_screen()
            current_screen = agent.get_screen()
            state = current_screen - last_screen
            while not done:
                action = agent.take_action(state)
                _, reward, done, _ = agent.env.step(action.item())
                episode_return += reward
                reward = torch.tensor([reward], device=device)
                last_screen = current_screen
                current_screen = agent.get_screen()
                next_state = current_screen - last_screen

                replay_buffer.add(state, action, next_state, reward)
                state = next_state
                if replay_buffer.size() > minimal_size:
                    transition_dict = replay_buffer.sample(batch_size)
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
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
plt.title('DQN on {}'.format("CartPole-v0"))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format("CartPole-v0"))
plt.show()