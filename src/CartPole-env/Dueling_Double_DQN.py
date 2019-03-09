#ToDo: Restructuring + adding comments
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import optimizers
import tensorflow as tf
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DuelingDoubleDQN:

    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = ReplayMemory(1000000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
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
            target = reward
            if not done:
                target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.policy_model.predict(next_state)[0])]
            # Update policy_model
            target_f = self.policy_model.predict(state)
            target_f[0][action] = (target - target_f[0][action]) * (target - target_f[0][action])
            self.policy_model.fit(state, target_f, epochs=1, verbose=0)

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
        input = Input(shape=(4,))
        fc1 = Dense(8, activation='relu')(input)
        fc2 = Dense(12, activation='relu')(input)

        # VALUE FUNCTION ESTIMATOR
        val1 = Dense(8, activation='relu')(fc2)
        val2 = Dense(1, activation='relu')(val1)

        # ADVANTAGE FUNCTION ESTIMATOR
        adv1 = Dense(8, activation='relu')(fc2)
        adv2 = Dense(self.action_size, activation='relu')(adv1)

        # COMBINE
        merge_layer = Lambda(lambda x: np.add(np.full_like(x[0], x[1][0, 0]), np.subtract(x[0], np.full_like(x[0], tf.reduce_mean(x[0])))), output_shape=lambda x: x[0])
        merge = merge_layer([adv2, val2])

        model = Model(inputs=input, outputs=merge)
        model.compile(loss="mean_squared_error", optimizer=optimizers.Adam(lr=self.learning_rate, clipnorm=1))
        return model

    def save(self):
        self.target_model.save('target_model.h5')
        self.policy_model.save('policy_model.h5')

    def load(self):
        self.target_model.load_weights('target_model.h5')
        self.policy_model.load_weights('policy_model.h5')
        print(self.policy_model.get_weights())