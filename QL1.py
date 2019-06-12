import gym
import numpy as np

env =gym.make("MountainCar-v0")
#env.reset()

learning_rate=0.1
discount=0.95
episodes=25000

show_every=2000

DOS=[20]*len(env.observation_space.high)
doswin=(env.observation_space.high-env.observation_space.low)/DOS


epsilon=0.3
start_epsilon_decay=1
end_epsilon_decay=episodes//2
epsilon_decay_value=epsilon/(end_epsilon_decay- start_epsilon_decay)

q_table=np.random.uniform(low=-2,high=0, size=(DOS+[env.action_space.n]))

def get_discrete_state(state):
	discrete_state=(state - env.observation_space.low)/doswin
	return tuple(discrete_state.astype(np.int))


for episode in range(episodes):

	if episode % show_every==0:
		print(episode)
		render=True
	else:
	 	render=False


	discrete_state=get_discrete_state(env.reset())
	done=False
	while not done:

		if np.random.random()> epsilon:
			action=np.argmax(q_table[discrete_state])
		else:
			action=np.random.randint(0, env.action_space.n)


		newstate,reward,done,_=env.step(action)
		new_discrete_state=get_discrete_state(newstate)
		if render:
			env.render()
		print(reward,newstate)
		if not done:
			max_future_q=np.max(q_table[new_discrete_state])
			current_q=q_table[discrete_state+(action,)]
			new_q=(1-learning_rate)*current_q+learning_rate*(reward+discount*max_future_q)
			q_table[discrete_state+(action,)]=new_q
		elif newstate[0]==env.goal_position:
			q_table[discrete_state+(action,)]=0
		discrete_state=new_discrete_state

	if end_epsilon_decay>= episode >=start_epsilon_decay:
		epsilon-=epsilon_decay_value

env.close()
