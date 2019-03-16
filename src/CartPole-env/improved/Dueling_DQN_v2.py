from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import optimizers
import tensorflow as tf
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DuelingDQN:

    def __init__(self, action_size, gamma=0.99, eps_dec=0.99, lr=0.00025):
        self.action_size = action_size
        self.memory = ReplayMemory(1000000)
        # Discount rate
        self.gamma = gamma
        # Setup epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = eps_dec
        # Learning rate
        self.learning_rate = lr
        # Iterative update
        self.step = 0
        self.C = 5

        self.target_model = self.create_model()
        self.policy_model = self.create_model()

    # Return the action with the highest estimated Q-value
    def act(self, state):
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        return np.argmax(act_values[0])  # returns action

    # Return a weighted random action where the estimated Q-values are used as weights
    # Also apply epsilon-greedy action selection
    def act_stochastic(self, state):
        if np.random.rand() <= self.epsilon:
            print("random")
            return random.randrange(self.action_size)
        print("network")
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        total = np.sum(act_values[0]) + 0.2
        # FIX: make sure that sum equals to 1
        diff = 1 - (act_values[0][0] + 0.1) / total - (act_values[0][1] + 0.1) / total
        # Return weighted random chosen action
        return np.random.choice([0, 1], p=[(x + 0.1) / total + diff / 2 for x in act_values[0]])

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
        fc1 = Dense(12, activation='relu')(input)
        fc2 = Dense(24, activation='relu')(fc1)

        #VALUE FUNCTION ESTIMATOR
        val1 = Dense(12, activation='relu')(fc2)
        val2 = Dense(1, activation='relu')(val1)

        #ADVANTAGE FUNCTION ESTIMATOR
        adv1 = Dense(12, activation='relu')(fc2)
        adv2 = Dense(self.action_size, activation='relu')(adv1)

        #COMBINE
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
