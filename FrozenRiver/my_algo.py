import numpy as np
import gym
from numpy.ma.core import fabs

def calc_value(env, gamma, value, policy) :

	new_value = np.zeros(env.env.nS)

	for s in range(env.env.nS) :
		a = policy[s]
		for p,s2,r,d in env.env.P[s][a] :
			new_value[s] += p*(r + gamma*value[s2])

	return new_value

def calc_policy(env, gamma, value) :

	policy = np.zeros(env.env.nS)

	for s in range(env.env.nS) :
		action_exp = np.zeros(env.env.nA)
		for a in range(env.env.nA) :
			action_exp[a] = sum([p*(r + gamma*value[s2]) for p,s2,r,_ in env.env.P[s][a]])

		policy[s] = np.argmax(action_exp)

	return policy

def policy_iteration(env, gamma, max_iter = 4000) :

	policy = np.random.choice(env.env.nA, env.env.nS)
	value = np.zeros(env.env.nS)
	eps = 1e-8

	for i in range(max_iter) :

		new_value = calc_value(env, gamma, value, policy)

		if np.max(np.fabs(new_value-value)) < eps and i>10:
			print('Accuracy enough, stop iteration at round {}'.format(i))
			break

		value = new_value
		policy = calc_policy(env, gamma, value)
			
	return policy

def evaluate_policy(env, policy, gamma, eval_round, render = False) :

	reward_list = np.zeros(eval_round)

	for i in range(eval_round) :
		total_reward = 0
		obs = env.reset()
		step_idx = 0

		while True:
			if render:
				env.render()
			obs, reward, done, _ = env.step(int(policy[obs]))
			total_reward += (gamma ** step_idx * reward)
			step_idx += 1
			if done:
				break
		reward_list[i] = total_reward
		render = False
	
	return reward_list

if __name__ == '__main__':

	np.set_printoptions(linewidth=np.inf)

	env_name  = 'FrozenLake8x8-v1' # 'FrozenLake-v1'
	env = gym.make(env_name)

	gamma = 1.0
	optimal_policy = policy_iteration(env, gamma)
	scores = evaluate_policy(env, optimal_policy, gamma, eval_round=1000, render=False)

	print('Average scores = ', np.mean(scores))
