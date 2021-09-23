import gym
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
import copy
import pandas as pd
import time
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()
eps = np.finfo(np.float32).eps.item()

class DuelingNet(nn.Module) :

	def __init__(self, n_state, n_hidden, n_output) :
		super().__init__()
		self.l1 = nn.Linear(n_state, n_hidden, bias=True)
		self.l2 = nn.ReLU()
		self.l3 = nn.Linear(n_hidden, n_output, bias=True)
		self.l4 = nn.Linear(n_hidden, 1, bias=True)
	
	def forward(self, X) :
		V1 = self.l1(X)
		V2 = self.l2(V1)
		V3 = self.l3(V2)
		V4 = self.l4(V2).expand(V3.size())
		V5 = V3+V4-V3.mean().expand(V3.size())
		return V5

class Replay :

	def __init__(self, capacity) :
		self.memory = pd.DataFrame(index=range(capacity), 
			columns=['observation', 'action', 
			'reward', 'next_observation', 'done'])
		self.capacity = capacity
		self.count = 0
		self.top = 0
	
	def store(self, *args) :
		self.memory.loc[self.count] = args
		self.count = (self.count+1) % self.capacity
		self.top = min(self.top+1, self.capacity)
	
	def sample(self, size) :
		index = np.random.choice(self.top, size=size)
		ret = (np.stack(self.memory.loc[index, field]) for field in self.memory.columns)
		return ret

class Agent() :

	def __init__(self, n_state, n_hidden, n_output, lr=5e-4, gamma=0.9, epsilon=0.01, device='cpu') :
		self.n_state = n_state
		self.n_output = n_output
		self.device = device
		self.net1 = DuelingNet(n_state, n_hidden, n_output).to(self.device)
		self.net2 = DuelingNet(n_state, n_hidden, n_output).to(self.device)
		self.optimizer1 = torch.optim.Adam(self.net1.parameters(), lr=lr)
		self.gamma = gamma
		self.epsilon = epsilon
		self.loss_func = nn.MSELoss()
		self.replay = Replay(1024)
	
	def action(self, state, israndom) :
		cur = torch.Tensor(state).unsqueeze(dim=0).to(self.device)
		if israndom and np.random.random() < self.epsilon :
			return np.random.randint(0, self.n_output)
		output = self.net1.forward(cur).cpu().detach().numpy()[0]
		return np.argmax(output)
	
	def train(self, state, action, reward, next_state, done, batch_size) :

		self.replay.store(state, action, reward, next_state, int(done))

		batch = list(self.replay.sample(batch_size))

		state = torch.FloatTensor(batch[0]).to(self.device)
		action = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
		reward = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
		next_state = torch.FloatTensor(batch[3]).to(self.device)
		done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

		# 获得下一状态的估值函数的最大值位置
		a = self.net1.forward(next_state).argmax(dim=1).view(-1,1)
		# 获取目标量
		u = reward + self.gamma * self.net2.forward(next_state).gather(1, a) * done
		v = self.net1.forward(state).gather(1, action)
		loss = self.loss_func(v, u)
		self.optimizer1.zero_grad()
		loss.backward()
		self.optimizer1.step()
	
	def save_model(self, episode) :
		torch.save(self.net1.state_dict(), './saves/net1')
		torch.save(self.net2.state_dict(), './saves/net2')
		print('episode {} saved'.format(episode))
	
	def load_model(self) :
		self.net1.load_state_dict(torch.load('./saves/net1'))
		self.net2.load_state_dict(torch.load('./saves/net2'))


def test(env, model, max_episode) :

	for t in range(max_episode) :
		state = env.reset()
		test_reward = 0
		while True :
			time.sleep(0.01)
			env.render()
			action = model.action(state, israndom=False)
			next_state, reward_real, done, info = env.step(action)
			test_reward += reward_real
			state = next_state
			if done :
				break
	avg_test_reward = test_reward / 10
	print('test reward : {}'.format(avg_test_reward))

if __name__ == '__main__' :

	env_name  = 'MountainCar-v0'
	env = gym.make(env_name)

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = Agent(2, 64, 3, device=device)
	model.load_model()

	test(env, model, 10)
	
