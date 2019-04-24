import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import cv2
import numpy as np
from DQN_Beamrider import DQN
from DeadBuffer import Buffer
from Double_DQN_Beamrider import DoubleDQN
from Dueling_DQN_Beamrider import DuelingDQN
from Dueling_Double_DQN_Beamrider import DuelingDoubleDQN
import matplotlib.pyplot as plt
import random
import copy

def preprocess_images(images):
    # Resize last two frames and take pixel-wise maximum of both images
    #images[0] = cv2.resize(images[0], (80, 110), interpolation=cv2.INTER_LINEAR)
    images[1] = cv2.resize(images[1], (80, 110), interpolation=cv2.INTER_LINEAR)

    # Transform frames to grayscale
    #images[0] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    images[1] = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    # Cut top border from image
    images[1] = images[1][25:105, 0:80]


    # Change each pixel to the maximum value of both corresponding pixels
    #i = cv2.addWeighted(images[1], 0.3, images[0], 0.7, 0)

    # TEST
    i = images[1]

    # Uncomment these two lines to show the preprocessed image
    #cv2.imshow('preprocessed', i)
    #cv2.waitKey(0)

    # Keras needs 4 dimensions, first dimension gives number of the data
    # Add extra axis in front and fill in with 1
    i = np.reshape(i, (80, 80, 1))
    i = np.expand_dims(i, axis=0)
    return i


plt.ion()


# Plotting
def plot_rewards(rewards):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.pause(0.001)

def plot_loss(losses):
    plt.figure(2)
    plt.clf()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.pause(0.001)

def plot_averageQ(averages):
    plt.figure(3)
    plt.clf()
    plt.title('Average Q')
    plt.xlabel('Epoch')
    plt.ylabel('Q')
    plt.plot(averages)
    plt.pause(0.001)


# INIT
env = gym.make('BeamRider-v4')
env.reset()
TRAINING_EPISODES = 0
EVALUATION_EPISODES = 1000
current_action = 0
state = None
done = 0

ACTIONS = [0, 1, 3, 4]
ACTION_MEANINGS = ['NO-OP', 'FIRE', 'RIGHT', 'LEFT']

for net in range(0, 1):

    if net == 0:
        #Deep Q-Network
        network = DQN(4)
        file = open("Rewards_DQN.csv", "w")
        lossesFile = open("Losses_DQN.csv", "w")
        averagesFile = open("Averages_DQN.csv", "w")
    elif net == 1:
        #Double DQN
        network = DoubleDQN(4)
        file = open("Rewards_DDQN.csv", "w")
        lossesFile = open("Losses_DDQN.csv", "w")
        averagesFile = open("Averages_DDQN.csv", "w")
    elif net == 2:
        #Dueling DQN
        network = DuelingDQN(4)
        file = open("Rewards_DUEL_DQN.csv", "w")
        lossesFile = open("Losses_DUEL_DQN.csv", "w")
        averagesFile = open("Averages_DUEL_DQN.csv", "w")
    else:
        #Dueling Double DQN
        network = DuelingDoubleDQN(4)
        file = open("Rewards_DUEL_DDQN.csv", "w")
        lossesFile = open("Losses_DUEL_DDQN.csv", "w")
        averagesFile = open("Averages_DUEL_DDQN.csv", "w")

    episode_rewards = []
    avg_losses = []
    losses = []
    averages = []
    lives = 3
    buffer = Buffer()

    # Playing
    for t in range(0, TRAINING_EPISODES + EVALUATION_EPISODES):
        # Get first 2 screens
        prev_screen = env.render(mode='rgb_array')
        _, reward, done, args = env.step(0)
        lives = args['ale.lives']
        next_screen = env.render(mode='rgb_array')
        next_state = preprocess_images([prev_screen, next_screen])
        if t < TRAINING_EPISODES:
            print("-------------------TRAINING-------------------")
            # Random NO-OP start (Test both NO-OP and RANDOM start)
            print("-----NO-OP start:")
            for r in range(1, random.randint(2, 35)):
                _, reward, done, args = env.step(0)
                next_screen = env.render(mode='rgb_array')
                next_state = preprocess_images([prev_screen, next_screen])
                print(ACTION_MEANINGS[current_action])
        else:
            print("-------------------EVALUATING-------------------")
            prev_screen = env.render(mode='rgb_array')
            _, reward, done, _ = env.step(0)
            next_screen = env.render(mode='rgb_array')
            next_state = preprocess_images([prev_screen, next_screen])

        episode_rewards.append(0)
        while not done:
            env.render()
            # Move to next state
            state = copy.deepcopy(next_state)

            # Keep learning from mistakes, but don't use training-experiences
            if t == TRAINING_EPISODES:
                network.memory.memory.clear()
                network.memory.position = 0
                network.epsilon = 0.75

            # Select next action
            if t < TRAINING_EPISODES or t % 25 == 0:
                #current_action = network.act_stochastic(state)
                current_action = random.randint(0, 3)
            else:
                current_action = network.act_eps_greedy(state)

            print(ACTION_MEANINGS[current_action])

            prev_screen = next_screen
            x = 0
            died = False

            _, reward, done, args = env.step(ACTIONS[current_action])
            if args['ale.lives'] < lives:
                died = True
                lives = args['ale.lives']

            next_screen = env.render(mode='rgb_array')
            next_state = preprocess_images([prev_screen, next_screen])

            episode_rewards[t] += reward

            if died:
                reward = -1

            # Reward clipping
            if reward > 0:
                reward = 1

            # Push transition to memory, Gym removes a live about 80 frames after actually dying so buffer those frames
            if died:
                trans = buffer.deadTrans()
                #cv2.imshow('preprocessed', np.reshape(np.squeeze(trans[0], axis=0), (110, 80)))
                #cv2.waitKey(0)
                network.memoryDied.push(trans[0], trans[1], trans[2], -10, trans[4])
            else:
                trans = buffer.push(state, current_action, next_state, reward, done)
                if trans is not None:
                    network.memory.push(trans[0], trans[1], trans[2], trans[3], trans[4])

            # Replay
            # If your hardware allows, you can increase the size of the minibatch
            if len(network.memory) > 32 and len(network.memoryDied) > 11:
                history = network.replay(32)
                losses.extend(history.history['loss'])
            else:
                network.epsilon = 0.99

            if len(network.average_Q) > 500:
                avg = np.average(network.average_Q)
                averagesFile.write(str(avg) + '\n')
                averages.append(avg)
                network.average_Q.clear()

        env.reset()
        if t % 1 == 0 and t > 0:
            avg = np.average(losses)
            avg_losses.append(avg)
            lossesFile.write(str(avg) + '\n')
            losses = []
            plot_rewards(episode_rewards)
            plot_loss(avg_losses)
            plot_averageQ(averages)

        if t % 10 == 0:
            network.save()

        # Write reward to csv file
        file.write(str(episode_rewards[t]) + '\n')

    file.close()
    lossesFile.close()
    averagesFile.close()

plot_rewards(episode_rewards)
cv2.waitKey()
plt.ioff()
plt.show()