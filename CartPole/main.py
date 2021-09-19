import gym
import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
import copy
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

class agent(nn.Module) :

	def __init__(self, dim_in, dim_calc, dim_out) :
		super().__init__()
		self.layer_1 = nn.Linear(dim_in, dim_calc)
		self.layer_2 = nn.ReLU()
		self.layer_3 = nn.Linear(dim_calc, dim_calc)
		self.layer_4 = nn.ReLU()
		self.layer_5 = nn.Linear(dim_calc, dim_out)
		self.layer_6 = nn.ReLU()
		self.layer_7 = nn.Softmax(dim=1)

		self.stack_log_prob = []
		self.stack_R = []
	
	def forward(self, X) :

		l1_tensor = self.layer_1(X)
		l2_tensor = self.layer_2(l1_tensor)
		l3_tensor = self.layer_3(l2_tensor)
		l4_tensor = self.layer_4(l3_tensor)
		l5_tensor = self.layer_5(l4_tensor)
		l6_tensor = self.layer_6(l5_tensor)
		l7_tensor = self.layer_7(l6_tensor)
		
		return l7_tensor
	
def get_action(model, state, device) :

	state = torch.Tensor(state).to(device).unsqueeze(0)
	option_prob = model(state)
	m = Categorical(option_prob)
	action = m.sample()
	model.stack_log_prob.append(m.log_prob(action))	#log_prob will be used to calculate CE

	return action.item()

def show_model(model, env, show_round, max_t, device) :

	model.to(device)

	with torch.no_grad():
		# model.eval()
		for i in range(show_round) :
			state = env.reset()
			for t in range(max_t) :
				env.render()
				action = 0 # get_action(model, state, device)
				state, reward, done, _ = env.step(action)
				if done :
					break
		# model.train()
	model.stack_log_prob.clear()
	model.stack_R.clear()

def test_model(model, env, test_round, max_t, device) :

	model.to(device)

	total_reward = 0

	with torch.no_grad():
		model.eval()
		for i in range(test_round) :
			state = env.reset()
			round_R = 0
			for t in range(max_t) :
				action = get_action(model, state, device)
				state, reward, done, _ = env.step(action)
				round_R += reward
				if done :
					break
			total_reward += round_R
		model.train()
	
	avg_reward = total_reward/test_round

	print('Average reward: {:.2f}'.format(avg_reward))

	return avg_reward

def update_model(model, optimizer) :
	R = 0
	loss = []
	returns = []
	for r in model.stack_R[::-1] :
		R = r + args.gamma*R
		returns.append(R)
	returns = torch.tensor(list(reversed(returns)))
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for log_prob, answer in zip(model.stack_log_prob, returns) :
		loss.append(-log_prob * answer)
	
	optimizer.zero_grad()
	loss = torch.cat(loss).sum()
	loss.backward()
	optimizer.step()

	model.stack_log_prob.clear()
	model.stack_R.clear()


def train_model(model, env, max_iter=10000, max_t=1000, device='cpu') :

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4 ,weight_decay=1e-4)

	model.train()
	model = model.to(device)
	print('Training on {}'.format(device))

	running_reward = 10
	best_reward = 0

	for iteration in range(max_iter) :
		model.stack_log_prob.clear()
		model.stack_R.clear()
		state = env.reset()
		cur_reward = 0
		for t in range(max_t) :
			# run the model
			if args.render and iteration % args.log_interval == 0 :
				env.render()
				pass
			action = get_action(model, state, device)
			state, reward, done, _ = env.step(action)
			model.stack_R.append(reward)
			cur_reward += reward
			if done :
				break
		update_model(model, optimizer)
		running_reward = 0.05 * cur_reward + (1 - 0.05) * running_reward
		if iteration % args.log_interval == 0 :
			print('Round {} Reward: {}'.format(iteration, running_reward))
			# show_model(model, env, 100, max_t=max_t, device=device)
		if running_reward > best_reward :
			torch.save(model.state_dict(), "trained.model", _use_new_zipfile_serialization=False)
			best_reward = running_reward

if __name__ == '__main__' :

	env_name  = 'CartPole-v1' # 'FrozenLake-v1'
	env = gym.make(env_name)

	env.seed(args.seed)
	torch.manual_seed(args.seed)

	model = agent(4, 128, 2)
	device = "cuda" if torch.cuda.is_available() else "cpu"

	test_model(model, env, test_round=100, max_t=1000, device=device)

	train_model(model, env, max_iter=3000, max_t=1000, device=device)

	test_model(model, env, test_round=100, max_t=1000, device=device)