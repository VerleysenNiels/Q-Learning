import gym
import cv2
import numpy as np
from DQN_v2 import DQN
from Double_DQN_v2 import DoubleDQN
from Dueling_Double_DQN_v2 import DuelingDoubleDQN
from Dueling_DQN_v2 import DuelingDQN
import matplotlib.pyplot as plt
import random

import keras.activations

ACTIONS = ['LEFT', 'RIGHT']

plt.ion()

# Plotting
def plot_rewards(episode_rewards):
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
TRAINING_EPISODES = 0
EVALUATION_EPISODES = 500
current_action = 0
state = None
next_state = None
done = 0
NR_MAX_SCORE = 3

for net in range(1, 2):

    if net == 0:
        #Deep Q-Network
        network = DQN(2)
        file = open("Rewards_DQN.csv", "w")
    elif net == 1:
        #Double DQN
        network = DoubleDQN(2)
        file = open("Rewards_DDQN.csv", "w")
    elif net == 2:
        #Dueling DQN
        network = DuelingDQN(2)
        file = open("Rewards_DUEL_DQN.csv", "w")
    else:
        #Dueling Double DQN
        network = DuelingDoubleDQN(2)
        file = open("Rewards_DUEL_DDQN.csv", "w")

    max_score_counter = 0
    episode_rewards = []

    #--------------------------------------------- PLAY -----------------------------------------------------
    for t in range(0, TRAINING_EPISODES + EVALUATION_EPISODES):
        if t < TRAINING_EPISODES:
            print("-------------------TRAINING episode: " + str(t) + "-------------------")
            # Random NO-OP start (there is no NO-OP, so do a random amount of random actions instead)
            print("-----Random start:")
            for r in range(1, random.randint(1, 5)):
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

            #Select next action
            if t < TRAINING_EPISODES:
                current_action = network.act_stochastic(state)
            else:
                #current_action = network.act(state)
                current_action = network.act_eps_greedy(state) # Act method from paper

            print(ACTIONS[current_action])

            next_state, reward, done, _ = env.step(current_action)
            next_state = np.expand_dims(next_state, axis=0)

            episode_rewards[t] += reward

            bool_end = env._max_episode_steps <= env._elapsed_steps + 1
            if done and not bool_end:
                reward = 0  # crashed

            #Push transition to memory
            network.memory.push(state, current_action, next_state, reward, done)

            #Replay
            #Ability to stop training when evaluating
            if network.memory.__len__() > 32 and max_score_counter < NR_MAX_SCORE:
                network.replay(32)

        env.reset()
        if t % 10 == 0:
            plot_rewards(episode_rewards)

        if t % 100 == 0:
            network.save()

        if episode_rewards[t] >= 499:
            max_score_counter += 1
        else:
            max_score_counter = 0

        # Write reward to csv file
        file.write(str(episode_rewards[t]) + '\n')

    file.close()
plt.ioff()
plt.show()