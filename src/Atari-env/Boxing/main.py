import gym
import cv2
import csv
import numpy as np
from DQN_Boxing import DQN
from Double_DQN_Boxing import DoubleDQN
import matplotlib.pyplot as plt
import random
import csv

def preprocess_images(images):
    # Resize last two frames and take pixel-wise maximum of both images
    images[0] = cv2.resize(images[0], (84, 84), interpolation=cv2.INTER_LINEAR)
    images[1] = cv2.resize(images[1], (84, 84), interpolation=cv2.INTER_LINEAR)

    # Transform frames to grayscale
    images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    # Change each pixel to the maximum value of both corresponding pixels
    i = np.maximum(images[0], images[1])

    # Uncomment these two lines to show the preprocessed image
    #cv2.imshow('preprocessed', i)
    #cv2.waitKey(0)

    # Keras needs 4 dimensions, first dimension gives number of the data
    # Add extra axis in front and fill in with 1
    i = np.reshape(i, (84, 84, 1))
    i = np.expand_dims(i, axis=0)
    return i


episode_rewards = []
plt.ion()

file = open("Rewards.csv", "w")
writer = csv.writer(file, delimiter=',')


# Plotting
def plot_rewards():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards)
    plt.pause(0.001)


# INIT
env = gym.make('Boxing-v4')
env.reset()
TRAINING_EPISODES = 1000
EVALUATION_EPISODES = 1000
current_action = 0
state = None
done = 0

file = open("Rewards.csv", "w")

ACTIONS = [0, 1, 2, 3, 4, 5]
ACTION_MEANINGS = ['NO-OP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN']

# ---------------CHOOSE NETWORK-------------------
# Deep Q-Network
network = DQN(6)

# Double DQN
#network = DoubleDQN(6)

# ------------------------------------------------

# Get first 2 screens
prev_screen = env.render(mode='rgb_array')
env.step(current_action)
next_screen = env.render(mode='rgb_array')
next_state = preprocess_images([prev_screen, next_screen])


# Playing
for t in range(0, TRAINING_EPISODES + EVALUATION_EPISODES):
    done = 0
    if t < TRAINING_EPISODES:
        print("-------------------TRAINING-------------------")
        # Random NO-OP start (Test both NO-OP and RANDOM start)
        print("-----Random start:")
        for r in range(0, random.randint(1, 5)):
            current_action = random.randint(0, 5)
            _, reward, done, _ = env.step(current_action)
            next_screen = env.render(mode='rgb_array')
            next_state = preprocess_images([prev_screen, next_screen])
            print(ACTION_MEANINGS[current_action])
    else:
        print("-------------------EVALUATING-------------------")
        next_screen = env.render(mode='rgb_array')

    episode_rewards.append(0)
    while not done:
        env.render()
        # Move to next state
        state = next_state

        # Keep learning from mistakes, but don't use training-experiences
        if t == TRAINING_EPISODES:
            network.memory.memory.clear()
            network.memory.position = 0

        # Select next action
        if t < TRAINING_EPISODES:
            current_action = network.act_stochastic(state)
        else:
            current_action = network.act(state)

        print(ACTION_MEANINGS[current_action])

        # Make a decision using last 2 frames as a state
        prev_screen = next_screen
        _, reward, done, _ = env.step(ACTIONS[current_action])
        next_screen = env.render(mode='rgb_array')
        next_state = preprocess_images([prev_screen, next_screen])

        episode_rewards[t] += reward

        # Reward clipping
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        # Push transition to memory
        network.memory.push(state, current_action, next_state, reward, done)

        # Replay
        # If your hardware allows, you can increase the size of the minibatch
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