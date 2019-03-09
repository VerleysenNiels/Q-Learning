#ToDo: Restructuring
import gym
import cv2
import numpy as np
from DQN import DQN
from Double_DQN import DoubleDQN
from Dueling_Double_DQN import DuelingDoubleDQN
from Dueling_DQN import DuelingDQN
import matplotlib.pyplot as plt
import csv

episode_rewards = []
mean_rewards = []

plt.ion()

# Plotting
def plot_rewards():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards)
    plt.pause(0.001)


#INIT
env = gym.make('CartPole-v1')
env.reset()
EPISODES = 500
current_action = 0
next_state, _, _, _ = env.step(current_action)
next_state = np.expand_dims(next_state, axis=0)

file = open("Rewards.csv", "w")
writer = csv.writer(file, delimiter=',')

#Deep Q-Network
network = DQN(2)

#Double DQN
#network = DoubleDQN(2)

#Dueling DQN
#network = DuelingDQN(2)

#Dueling Double DQN
#network = DuelingDoubleDQN(2)

#Playing
for t in range(0, EPISODES):
    done = 0
    episode_rewards.append(0)
    while not done:
        env.render()
        state = next_state
        #Select next action
        current_action = network.act(state)
        print(current_action)

        next_state, reward, done, _ = env.step(current_action)
        next_state = np.expand_dims(next_state, axis=0)

        episode_rewards[t] += reward

        if done:
            reward = -0.1

        #Push transition to memory
        network.memory.push(state, current_action, next_state, reward, done)

        #Replay
        if network.memory.__len__() > 32:
            network.replay(32)
    env.reset()
    if t % 10 == 0:
        plot_rewards()

    if t % 100 == 0:
        network.save()

    # Write reward to csv file
    file.write(str(episode_rewards[t]) + '\n')

file.close()
plot_rewards()
cv2.waitKey()
plt.ioff()
plt.show()