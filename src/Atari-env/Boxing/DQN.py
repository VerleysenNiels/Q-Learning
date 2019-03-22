#ToDo: Restructuring + adding comments
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DQN:

    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = ReplayMemory(1000000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00025
        self.step = 0
        self.C = 5

        self.target_model = self.create_model()
        self.policy_model = self.create_model()


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print("random")
            return random.randrange(self.action_size)
        print("network")
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, next_state, reward, done in minibatch:
            # Y = r if done ; else Y = r + gamma * max (target_Q(new state, action))
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # Update policy_model
            target_f = self.policy_model.predict(state)
            target_f[0][action] = (target - target_f[0][action]) * (target - target_f[0][action])
            #print("fitting")
            self.policy_model.fit(state, target_f, epochs=1, verbose=0)
            #print("done")

        # Iterative update
        if self.step == self.C:
            self.update_epsilon()
            self.target_model = self.policy_model
            self.step = 0
        else:
            self.step += 1

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def create_model(self):
        model = Sequential()

        # CNN
        model.add(Conv2D(32, kernel_size=8, activation='relu', input_shape=(84, 84, 1), strides=4))
        model.add(Conv2D(64, kernel_size=4, activation='relu', strides=2))
        model.add(Conv2D(64, kernel_size=3, activation='relu', strides=1))
        model.add(Flatten())

        # NN
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss="mean_squared_error", optimizer=optimizers.RMSprop(lr=self.learning_rate, rho=self.gamma, epsilon=self.epsilon, decay=self.epsilon_decay, clipnorm=1))
        return model

    def save(self):
        self.target_model.save('target_model.h5')
        self.policy_model.save('policy_model.h5')

    def load(self):
        self.target_model.load_weights('target_model.h5')
        self.policy_model.load_weights('policy_model.h5')
        # print(self.policy_model.get_weights())