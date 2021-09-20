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
		model.eval()
		for i in range(show_round) :
			state = env.reset()
			for t in range(max_t) :
				env.render()
				action = get_action(model, state, device)
				state, reward, done, _ = env.step(action)
				if done :
					print(t)
					break
		model.train()
	model.stack_log_prob.clear()
	model.stack_R.clear()

if __name__ == '__main__' :

	gym.register(
		id="CartPole-v1",
		entry_point="gym.envs.classic_control:CartPoleEnv",
		max_episode_steps=np.Inf,
		reward_threshold=np.Inf,
	)


	env_name  = 'CartPole-v1' # 'FrozenLake-v1'
	env = gym.make(env_name)

	env.seed(args.seed)
	torch.manual_seed(args.seed)

	device =  "cpu"

	model = agent(4, 128, 2)
	model.load_state_dict(torch.load('trained.model', map_location=torch.device(device)))

	show_model(model, env, show_round=1, max_t=10000, device=device)