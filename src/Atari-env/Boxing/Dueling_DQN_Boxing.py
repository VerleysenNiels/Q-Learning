from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten
from keras import optimizers
import tensorflow as tf
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DuelingDQN:

    def __init__(self, action_size, gamma=0.99, eps_dec=0.99, lr=0.00025):
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
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

        self.actions = [x for x in range(0, action_size)]

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
        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            print("random")
            return random.randrange(self.action_size)
        print("network")
        # Estimate Q-values
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        # Prepare weights for weighted random choice
        total = np.sum(act_values[0])
        # FIX: make sure that sum of the weights is equal to 1
        diff = 1
        for x in range(0, self.action_size):
            diff -= act_values[0][x] / total
        # Return weighted random chosen action
        return np.random.choice(self.actions, p=[x / total + diff / self.action_size for x in act_values[0]])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, next_state, reward, done in minibatch:
            # Y = r if done ; else Y = r + gamma * max (target_Q(new state, action))
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # Update policy_model
            target_f = self.policy_model.predict(state)
            # Squared difference
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
        input = Input(shape=(84, 84, 1))

        # CNN
        cnn1 = Conv2D(32, kernel_size=8, activation='relu', strides=4)(input)
        cnn2 = Conv2D(64, kernel_size=4, activation='relu', strides=2)(cnn1)
        cnn3 = Conv2D(64, kernel_size=3, activation='relu', strides=1)(cnn2)
        cnn4 = Flatten()(cnn3)


        # VALUE FUNCTION ESTIMATOR
        val1 = Dense(512, activation='relu')(cnn4)
        val2 = Dense(1, activation='relu')(val1)

        # ADVANTAGE FUNCTION ESTIMATOR
        adv1 = Dense(512, activation='relu')(cnn4)
        adv2 = Dense(self.action_size, activation='relu')(adv1)

        # COMBINE
        merge_layer = Lambda(lambda x: np.add(np.full_like(x[0], x[1][0, 0]), np.subtract(x[0], np.full_like(x[0], tf.reduce_mean(x[0])))), output_shape=lambda x: x[0])
        merge = merge_layer([adv2, val2])

        model = Model(inputs=input, outputs=merge)
        model.compile(loss="mean_squared_error", optimizer=optimizers.RMSprop(lr=self.learning_rate, rho=self.gamma, epsilon=self.epsilon, decay=self.epsilon_decay, clipnorm=1))
        return model

    def save(self):
        self.target_model.save_weights('target_model.h5')
        self.policy_model.save_weights('policy_model.h5')

    def load(self):
        self.target_model.load_weights('target_model.h5')
        self.policy_model.load_weights('policy_model.h5')
        print(self.policy_model.get_weights())
