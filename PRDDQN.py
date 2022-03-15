# Prioritized Replay DQN
import gym
import torch
import torch.nn.functional as F
import numpy as np
import random
import pylab as plt


class SumTree(object):  # 定义SumTree
    data_pointer = 0  # 初始化数据指针

    def __init__(self, capacity):
        self.capacity = capacity  # 定义容量
        self.tree = np.zeros(2 * capacity - 1)  # 初始化树结构,用来存储优先级
        self.data = np.zeros(capacity, dtype=object)  # 初始化数据结构,用来存储所有转换关系的数据

    def add(self, p, data):  # 更新数据
        tree_idx = self.data_pointer + self.capacity - 1  # 初始化树节点的索引
        self.data[self.data_pointer] = data  # 更新,存储数据
        self.update(tree_idx, p)  # 调用 update 函数,更新优先级 P
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 若指针大于等于SumTree的容量,重置为 0
            self.data_pointer = 0

    def update(self, tree_idx, p):  # 更新优先级
        change = p - self.tree[tree_idx]  # 计算变化量
        self.tree[tree_idx] = p  # 存储优先级
        while tree_idx != 0:  # 若树节点的索引不为 0 ,更新父节点
            tree_idx = (tree_idx - 1) // 2  # 寻找父节点
            self.tree[tree_idx] += change  # 更新父节点

    def get_leaf(self, v):  # 采样
        parent_idx = 0  # 初始化父亲节点索引
        while True:
            cl_idx = 2 * parent_idx + 1  # 计算左右孩子的索引
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # 若左孩子索引大于树的长度,那么父亲节点索引就是叶节点索引
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:  # 若 V 小于等于左孩子,那么就把左孩子的索引赋给父亲节点索引
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]  # 否则用 V - 左孩子, 并把右孩子的索引赋给父亲节点索引
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1  # 计算数据索引
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]  # 返回叶子节点索引,对应存储的优先级\转换关系的数据

    @property  # 装饰器,把一个方法变成属性调用的
    def total_p(self):
        return self.tree[0]  # 返回 total_p


class Memory(object):
    epsilon = 0.01  # 设置阈值,防止出现 0 的优先级
    alpha = 0.6  # [0~1] 决定我们要使用多少 Importance Sampling weight 的影响, 如果 alpha = 0, 我们就没使用到任何 Importance Sampling.
    beta = 0.4  # importance sampling 系数, 从 0.4 到 1
    beta_increment_per_sampling = 0.001  # 重要性每次增长0.001
    abs_err_upper = 1.  # 预设 P 的最大值

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):  # 存储转换关系的数据到 SumTree
        max_p = np.max(self.tree.tree[-self.tree.capacity:])  # 取出 SumTree中倒数capacity个值中的最大的一个,也就是最大的叶子节点值
        if max_p == 0:  # 若 max_p为0,则将 abs_err_upper 赋给 max_p
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # 调用add函数更新 SumTree 的数据

    def sample(self, n):  # 采样 n 个样本
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment 优先级分段
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # 更新 beta, importance sampling 系数,最大为1

        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # min_prob: 用叶子节点中最小的优先级值除以总的优先级值 (方便后边计算重要性采样权重)
        if min_prob == 0:  # 若 min_prob 为 0,给它赋值为 0.0001 (后边计算,这个出现在分母,不能为0)
            min_prob = 0.00001
        for i in range(n):  # 遍历 n
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)  # 在 a,b 之间随机生成一个数
            idx, p, data = self.tree.get_leaf(v)  # 调用 get_leaf 函数,返回叶子节点索引,对应存储的优先级\转换关系的数据
            prob = p / self.tree.total_p  # 计算优先级概率
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)  # 计算重要性采样权重
            b_idx[i], b_memory[i, :] = idx, data  # 存储索引，所有转换关系数据
        return b_idx, b_memory, ISWeights  # 返回索引，数据，权重

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # 防止出现 0
        clipped_errors = np.minimum(abs_errors.detach().numpy(), self.abs_err_upper)  # 比较两个数组并返回一个包含元素最小值的新数组
        ps = np.power(clipped_errors, self.alpha)  # p**alpha
        for ti, p in zip(tree_idx, ps):  # ti,p在 zip(tree_idx, ps) 中取，zip()是一个打包函数，将 tree_idx, ps 打包。
            self.tree.update(ti, p)  # 调用update函数，更新优先级


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
REPLAY_SIZE = 10000  # 经验池大小
BATCH_SIZE = 128  # Minimatch 大小
Update_Target_Freq = 10  # 目标网络参数更新频率


class DQN:
    def __init__(self, env):
        self.replay_total = 0  # 定义回放次数
        self.target_Q_net = MODEL(env)  # 定义目标网络
        self.current_Q_net = MODEL(env)  # 定义当前网络
        self.memory = Memory(capacity=REPLAY_SIZE)  # 定义经验池大小
        self.time_step = 0  # 定义时间步数
        self.epsilon = INITIAL_EPSILON  # 定义初始epsilon
        self.optimizer = torch.optim.Adam(params=self.current_Q_net.parameters(), lr=0.0001)  # 使用Adam优化器

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))  # np.hstack: 按水平方向堆叠数组构成一个新的数组
        self.memory.store(transition)  # 调用store函数存储转换关系数据

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.current_Q_net.action_dim)  # 对 action 进行 one_hot 编码，若选择某个动作，对应位置为1.
        one_hot_action[action] = 1
        self.store_transition(state, one_hot_action, reward, next_state, done)
        self.replay_total += 1  # 完成一次存储，回放次数加1
        if self.replay_total > BATCH_SIZE:  # 判断回放总次数是否大于BATCH_SIZE，大于就开始训练
            self.train_Q_network()

    def train_Q_network(self):  # 定义训练
        self.time_step += 1
        # 1. 从经验池采样
        tree_idx, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)  # 调用sample函数采样
        state_batch = torch.tensor(minibatch[:, 0:4], dtype=torch.float32)  # 取出state_batch，minibatch中所有行的前4列
        action_batch = torch.tensor(minibatch[:, 4:6], dtype=torch.float32)  # 取出action_batch，minibatch中所有行的第5，6列
        reward_batch = [data[6] for data in minibatch]  # 取出reward_batch，minibatch中每一行的第7列
        next_state_batch = torch.tensor(minibatch[:, 7:11], dtype=torch.float32)  # 取出next_state_batch，minibatch中所有行的第8，9，10，11列

        # 2. 计算 y
        y_batch = []  # 定义y_batch为一个数组
        current_a = self.current_Q_net.create_Q_network(next_state_batch)  # 调用 create_Q_network，使用current_Q_net计算Q值
        max_current_action_batch = torch.argmax(current_a, dim=1)  # argmax 函数：dim = 1,返回每一行中最大值的索引；dim = 0 ,返回每一列中最大值的索引
        Q_value_batch = self.target_Q_net.create_Q_network(next_state_batch)  # 调用 create_Q_network，使用target_Q_net计算Q值

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][11]  # 取出 minibatch 中每一行的第十二个数据，即取出是否到达终止的标识
            if done:
                y_batch.append(reward_batch[i])  # 若到达终止条件，y_batch=reward_batch
            else:  # 若未到达终止条件
                max_current_action = max_current_action_batch[i]  # 取出在当前网络中Q值最大时对应动作的索引
                y_batch.append(reward_batch[i] + GAMMA * Q_value_batch[
                    i, max_current_action])  # 计算Y, reward_batch +GAMMA *目标网络中对应当前网络中Q值最大时对应动作的Q值

        y = self.current_Q_net(torch.FloatTensor(state_batch),
                               torch.FloatTensor(action_batch))  # 调用当前网络计算在state_batch下执行action_batch得到的回报
        # torch.FloatTensor ：转换数据类型为32位浮点型
        y_batch = torch.FloatTensor(y_batch)
        cost = self.loss(y_batch, y, torch.tensor(ISWeights))  # 调用loss函数，计算损失函数
        self.optimizer.zero_grad()  # 初始化，把梯度置零，把loss关于weight的导数变成0.
        cost.backward()  # 计算梯度
        self.optimizer.step()  # 根据梯度更新参数
        abs_errors = torch.abs(y_batch - y)  # 计算y_batch - y的绝对值
        self.memory.batch_update(tree_idx, abs_errors)  # 调用batch_update函数，更新树

    def loss(self, y_output, y_true, ISWeights):  # 定义损失函数
        value = y_output - y_true
        return torch.mean(value * value * ISWeights)

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
            torch.save(self.current_Q_net.state_dict(), 'prdqn_net_params.pkl')  # 保存当前网络参数到本地
            self.target_Q_net.load_state_dict(torch.load('prdqn_net_params.pkl'))  # 上穿当前网络参数并赋给目标网络


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
            agent.perceive(state, action, reward, next_state, done)   # 调用perceive函数存储所有转换数据
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
    plt.title('Prioritized Replay DQN')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()


if __name__ == '__main__':
    main()
