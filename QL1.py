import gym

env =gym.make("MountainCar-v0")
state=env.reset()

done=False
while not done:
	action=2
	newstate,reward,done,_=env.step(action)
	#env.render()
	print(reward,newstate)
