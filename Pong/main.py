import os
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import pdb

from torch.optim import optimizer

inputDim = (160, 160, 3)
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='PyTorch Policy-Gradient with baseline at openai-gym pong')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
					help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='G',
					help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=1, metavar='G',
					help='Every how many episodes to do a param update')
parser.add_argument('--save_round', type=int, default=1, metavar='G',
					help='Every how many episodes to save model')
parser.add_argument('--seed', type=int, default=123, metavar='N',
					help='random seed (default: 123)')
parser.add_argument('--test', action='store_true',
	   				help='whether to test the trained model or keep training')
parser.add_argument('--render', action='store_true',
	   				help='whether to show the window')
args = parser.parse_args()

class myNet(nn.Module) :

	def __init__(self, actionN) -> None:
		
		super().__init__()
		self.extractor = nn.Sequential(
			nn.Conv2d(3, 6, kernel_size=(4,4), stride=(1,1)),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(6, 24, kernel_size=(16,16), stride=(2,2)),
			nn.Conv2d(24, 96, kernel_size=(16,16), stride=(2,2)),
			nn.Conv2d(96, 192, kernel_size=(9,9), stride=(1,1))
		)
		self.actionHead = nn.Sequential(
			nn.ReLU(),
			nn.Linear(192, 192),
			nn.ReLU(),
			nn.Linear(192, actionN),
			nn.Softmax(dim=1)
		)
		self.valueHead = nn.Sequential(
			nn.ReLU(),
			nn.Linear(192, 192),
			nn.ReLU(),
			nn.Linear(192, 1)
		)
		self.actionN = actionN

		self.logProb = [[]]
		self.value = [[]]
		self.reward = [[]]

	def forward(self, X) :

		batchSize = X.shape[0]
		X = X.permute(2, 0, 1).unsqueeze(0)
		patterns = self.extractor(X).view(1, -1)
		actions =  self.actionHead(patterns)
		values = self.valueHead(patterns)

		return actions[0], values[0][0]

	def selectAction(self, X) :
		
		X = torch.Tensor(X).to(device)
		probs, value = self(X)
		m = Categorical(probs)
		action = m.sample()
		self.logProb[-1].append(m.log_prob(action))
		self.value[-1].append(value)

		# print(probs.tolist())

		return action
	
	def updateParam(self, optimizer) :

		R = 0
		stateReward = []	# the real reward
		stateValue = []		# the estimated value of a state (baseline)
		actionLogProb = []
		policyLoss = []
		valueLoss = []

		for rewardPerEps in self.reward[::-1] :
			R = 0
			for r in rewardPerEps[::-1] :
				R = r + args.gamma*R
				stateReward.append(R)
		stateReward = torch.tensor(list(reversed(stateReward))).to(device)
		stateReward = (stateReward - stateReward.mean()) / (1e-6 + stateReward.var())

		for valuePerEps in self.value :
			stateValue += valuePerEps
		
		for logProbPerEps in self.logProb :
			actionLogProb += logProbPerEps
		
		for reward, value, logProb in zip(stateReward, stateValue, actionLogProb) :

			advantage = reward - value
			policyLoss.append(-advantage * logProb)
			valueLoss.append(F.smooth_l1_loss(value, reward))
		
		optimizer.zero_grad()
		loss = torch.stack(policyLoss).sum() + torch.stack(valueLoss).sum()
		loss.to(device)

		loss.backward()
		optimizer.step()

		del self.value[:]
		del self.logProb[:]
		del self.reward[:]
		del stateReward
		del loss


def train(env, net : myNet, optimizer) :

	lastState = np.zeros(inputDim)
	currentReward = 0
	runningReward = None

	for episode in count(1) :

		net.reward.append([])
		net.logProb.append([])
		net.value.append([])
		state = env.reset()
		
		for tick in count(1) :
			if args.render : env.render()
			state = state[35:195]
			action = net.selectAction(state - lastState)	# try state - lastState 
			lastState = state
			state, reward, done, _ = env.step(action+2)

			if not args.test : net.reward[-1].append(reward)

			currentReward += reward

			if done :
				runningReward = currentReward if runningReward is None else (runningReward*0.95 + currentReward*0.05)
				print('Episode {} , current reward : {}'.format(episode, runningReward))
				currentReward = 0
				break

		if episode % args.batch_size == 0 and not args.test :

			net.updateParam(optimizer)
			net.reward = [[]]
			net.logProb = [[]]
			net.value = [[]]
			torch.save(net.state_dict(), 'latest.model'.format(episode))
			print('Episode {} , model saved'.format(episode))
		
if __name__ == '__main__' :

	env = gym.make('Pong-v0')
	env.seed(args.seed)
	torch.manual_seed(args.seed)

	net = myNet(3).to(device)
	optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

	if args.test :
		net.load_state_dict(torch.load('latest.model', map_location=torch.device(device)))

	train(env, net, optimizer)