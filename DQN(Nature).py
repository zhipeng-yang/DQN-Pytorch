# DQN(Nature)

import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import pylab as plt


class MODEL(torch.nn.Module):
    def __init__(self, env):
        super(MODEL, self).__init__()  # 调用父类构造函数
        self.state_dim = env.observation_space.shape[0]  # 状态个数
        self.action_dim = env.action_space.n  # 动作个数
        self.fc1 = torch.nn.Linear(self.state_dim, 20)  # 建立第一层网络 : 随机生成20*4的权重，以及1*20的偏置，Y = XA^T + b
        self.fc1.weight.data.normal_(0, 0.6)  # 设置第一层网络参数，使得第一层网络的权重服从正态分布：均值为0，标准差为0.6
        self.fc2 = torch.nn.Linear(20, self.action_dim)  # 建立第二层网络：随机生成2*20的权重，以及1*2的偏置，Y = XA^T + b

    def create_Q_network(self, x):  # 创建 Q 网络
        x = F.relu(self.fc1(x))  # 调用 torch 的 relu 函数
        Q_value = self.fc2(x)  # 输出 Q_value
        return Q_value

    def forward(self, x, action_input):
        Q_value = self.create_Q_network(x)
        Q_action = torch.mul(Q_value, action_input).sum(
            dim=1)  # 计算执行动作action_input得到的回报。torch.mul:矩阵点乘; torch.sum: dim = 1按行求和，dim = 0按列求和
        return Q_action


# 设置参数
GAMMA = 0.9  # 折现因子
INITIAL_EPSILON = 0.5  # 初始的epsilon
FINAL_EPSILON = 0.01  # 最终的epsilon
Update_Target_Freq = 10  # 目标网络参数更新频率


class DQN:
    def __init__(self, env):
        self.replay_total = 0  # 定义回放次数
        self.Replay_Size = 10000  # 定义经验池大小
        self.Batch_Size = 128  # 定义mini_batch大小
        self.replay_buffer = np.zeros(self.Replay_Size, dtype=object)  # 初始化经验池,用来存储所有转换关系的数据
        self.data_pointer = 0  # 初始化数据指针
        self.target_Q_net = MODEL(env)  # 定义目标网络
        self.current_Q_net = MODEL(env)  # 定义当前网络
        self.time_step = 0  # 定义时间步数
        self.epsilon = INITIAL_EPSILON  # 定义初始epsilon
        self.optimizer = torch.optim.Adam(params=self.current_Q_net.parameters(), lr=0.0001)  # 使用Adam优化器

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.current_Q_net.action_dim)  # 对 action 进行 one_hot 编码，若选择某个动作，对应位置为1.
        one_hot_action[action] = 1
        self.store_transition(state, one_hot_action, reward, next_state, done)
        self.replay_total += 1  # 完成一次存储，回放次数加1
        if self.replay_total > self.Batch_Size:  # 判断回放总次数是否大于BATCH_SIZE，大于就开始训练
            self.train_Q_network()

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))  # np.hstack: 按水平方向堆叠数组构成一个新的数组
        self.replay_buffer[self.data_pointer] = transition  # 更新,存储数据
        self.data_pointer += 1
        if self.data_pointer >= self.Replay_Size:  # 若指针大于等于经验池的容量,重置为 0
            self.data_pointer = 0

    def train_Q_network(self, k=0):  # 定义训练
        minibatch = np.empty((self.Batch_Size, self.replay_buffer[0].size))
        self.time_step += 1
        # 1. 从经验池采样
        a = random.sample(range(0, self.replay_total-1), self.Batch_Size)  # 在 0, self.Replay_Size-1 之间随机生成Batch_Size个数
        for i in range(self.Batch_Size):  # 遍历 BATCH_SIZE
            v = a[k]
            k += 1
            minibatch[i, :] = self.replay_buffer[v]
        state_batch = torch.tensor(minibatch[:, 0:4], dtype=torch.float32)  # 取出state_batch，minibatch中所有行的前4列
        action_batch = torch.tensor(minibatch[:, 4:6], dtype=torch.float32)  # 取出action_batch，minibatch中所有行的第5，6列
        reward_batch = [data[6] for data in minibatch]  # 取出reward_batch，minibatch中每一行的第7列
        next_state_batch = torch.tensor(minibatch[:, 7:11],
                                        dtype=torch.float32)  # 取出next_state_batch，minibatch中所有行的第8，9，10，11列

        # 2. 计算 y
        y_batch = []  # 定义y_batch为一个数组
        Q_value_batch = self.target_Q_net.create_Q_network(next_state_batch)  # 调用 create_Q_network，使用target_Q_net计算Q值
        max_target_Q_value_batch = torch.max(Q_value_batch, dim=1)[0]  # 返回每一行中的最大值
        for i in range(0, self.Batch_Size):
            done = minibatch[i][11]  # 取出 minibatch 中每一行的第12个数据，即取出是否到达终止的标识
            if done:
                y_batch.append(reward_batch[i])  # 若到达终止条件，y_batch=reward_batch
            else:  # 若未到达终止条件
                max_target_Q_value = max_target_Q_value_batch[i]  # 取出在目标网络中每个状态执行动作获得的最大Q值
                y_batch.append(reward_batch[i] + GAMMA * max_target_Q_value)  # 计算Y, reward_batch + GAMMA *目标网络中动作的最大Q值
        y = self.current_Q_net(torch.FloatTensor(state_batch),
                               torch.FloatTensor(action_batch))  # 调用当前网络计算在state_batch下执行action_batch得到的回报
        # torch.FloatTensor ：转换数据类型为32位浮点型
        y_batch = torch.FloatTensor(y_batch)
        cost = self.loss(y_batch, y)  # 调用loss函数，计算损失函数
        self.optimizer.zero_grad()  # 初始化，把梯度置零，把loss关于weight的导数变成0.
        cost.backward()  # 计算梯度
        self.optimizer.step()  # 根据梯度更新参数

    def loss(self, y_output, y_true):  # 定义损失函数
        value = y_output - y_true
        return torch.mean(value * value)

    def e_greedy_action(self, state):  # 定义epsilon_greedy算法
        Q_value = self.current_Q_net.create_Q_network(torch.FloatTensor(state))  # 跟据输入状态调用当前网络计算 Q_value
        if random.random() <= self.epsilon:  # 使用random函数随机生成一个0-1的数，若小于epsilon，更新epsilon并返回随机动作
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.current_Q_net.action_dim - 1)
        else:  # 否则更新 epsilon， 并返回 Q_value 最大时对应的动作
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return torch.argmax(Q_value).item()  # 返回Q_value中最大值的索引值

    def action(self, state):  # 返回目标网络中Q_value最大值的索引值
        return torch.argmax(self.target_Q_net.create_Q_network(torch.FloatTensor(state))).item()

    def update_target_params(self, episode):  # 更新目标网络参数
        if episode % Update_Target_Freq == 0:
            torch.save(self.current_Q_net.state_dict(), 'dqn_net_params.pkl')  # 保存当前网络参数到本地
            self.target_Q_net.load_state_dict(torch.load('dqn_net_params.pkl'))  # 上穿当前网络参数并赋给目标网络


# ---------------------------------------------------------
def main():
    # 初始化参数，智能体环境
    ENV_NAME = 'CartPole-v0'
    EPISODE = 3000  # 迭代周期数
    STEP = 300  # 每个周期迭代时间步
    TEST = 10  # 测试次数
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    Ave_reward = []
    Episode = []

    for episode in range(EPISODE):
        # 初始化环境
        state = env.reset()
        for step in range(STEP):
            action = agent.e_greedy_action(state)  # 调用epsilon_greedy算法选择动作
            next_state, reward, done, _ = env.step(action)  # 执行当前动作获得所有转换数据
            # 定义回报
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)  # 调用perceive函数存储所有转换数据
            state = next_state  # 更新状态
            if done:
                break
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # 调用action函数,获得目标网络中Q值最大的动作
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        agent.update_target_params(episode)
        Episode.append(episode)
        Ave_reward.append(ave_reward)
    # 绘制平均奖励图像
    plt.plot(Episode, Ave_reward)
    plt.title('Nature DQN')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()


if __name__ == '__main__':
    main()
