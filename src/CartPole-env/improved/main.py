#ToDo: Restructuring
import gym
import cv2
import numpy as np
from DQN_v2 import DQN
from Double_DQN_v2 import DoubleDQN
from Dueling_Double_DQN_v2 import DuelingDoubleDQN
from Dueling_DQN_v2 import DuelingDQN
import matplotlib.pyplot as plt
import csv
import random

episode_rewards = []
mean_rewards = []
ACTIONS = ['LEFT', 'RIGHT']

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
TRAINING_EPISODES = 400
EVALUATION_EPISODES = 200
current_action = 0
state = None
next_state = None
done = 0

file = open("Rewards.csv", "w")
writer = csv.writer(file, delimiter=',')

#Deep Q-Network
#network = DQN(2)

#Double DQN
#network = DoubleDQN(2)

#Dueling DQN
#network = DuelingDQN(2)

#Dueling Double DQN
network = DuelingDoubleDQN(2)

#--------------------------------------------- PLAY -----------------------------------------------------
for t in range(0, TRAINING_EPISODES + EVALUATION_EPISODES):
    if t < TRAINING_EPISODES:
        print("-------------------TRAINING-------------------")
        # Random NO-OP start (there is no NO-OP, so do a random amount of random actions instead)
        print("-----Random start:")
        for r in range(0, random.randint(0, 3)):
            current_action = random.randint(0, 1)
            next_state, reward, done, _ = env.step(current_action)
            next_state = np.expand_dims(next_state, axis=0)
            print(ACTIONS[current_action])
    else:
        print("-------------------EVALUATING-------------------")
        current_action = random.randint(0, 1)
        next_state, reward, done, _ = env.step(current_action)
        next_state = np.expand_dims(next_state, axis=0)

    print("-----Start:")
    episode_rewards.append(0)
    while not done:
        env.render()
        state = next_state

        # Keep learning from mistakes, but don't use training-experiences
        if t == TRAINING_EPISODES:
            network.memory.memory.clear()
            network.memory.position = 0

        #Select next action
        if t < TRAINING_EPISODES:
            current_action = network.act_stochastic(state)
        else:
            current_action = network.act(state)

        print(ACTIONS[current_action])

        next_state, reward, done, _ = env.step(current_action)
        next_state = np.expand_dims(next_state, axis=0)

        episode_rewards[t] += reward

        #if done:
        #    reward = -0.1

        #Push transition to memory
        network.memory.push(state, current_action, next_state, reward, done)

        #Replay
        #Ability to stop training when evaluating
        if network.memory.__len__() > 32:
            network.replay(32)

    env.reset()
    #if t % 10 == 0:
    #    plot_rewards()

    if t % 100 == 0:
        network.save()

    # Write reward to csv file
    file.write(str(episode_rewards[t]) + '\n')

file.close()
plot_rewards()
cv2.waitKey()
plt.ioff()
plt.show()