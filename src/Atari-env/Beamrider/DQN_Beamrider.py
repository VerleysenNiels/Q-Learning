from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers
from keras import backend as K
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DQN:

    def __init__(self, action_size, gamma=0.95, eps_dec=0.99, lr=3e-5):
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.memoryDied = ReplayMemory(250)
        # Discount rate
        self.gamma = gamma
        # Setup epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = eps_dec
        # Learning rate
        self.learning_rate = lr
        # Iterative update
        self.step = 0
        self.C = 5

        self.average_Q = []

        self.actions = [x for x in range(0, action_size)]

        self.target_model = self.create_model()
        self.policy_model = self.create_model()

    # Return the action with the highest estimated Q-value
    def act(self, state):
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        avg_Q = np.average(act_values[0])
        self.average_Q.append(avg_Q)
        return np.argmax(act_values[0])  # returns action

    # Act-method from paper
    def act_eps_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return self.act(state)

    # Return a weighted random action where the estimated Q-values are used as weights
    def act_stochastic(self, state):
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        e = np.exp(act_values[0] - np.max(act_values[0]))
        p = e / np.sum(e, axis=-1)
        return np.random.choice(self.actions, p=p)

    def map_targets(self, transition):
        target = transition[3]  # 0 now manually set at main.
        if not transition[4]:
            target = transition[3] + self.gamma * np.amax(self.target_model.predict(transition[2])[0])
        # Update policy_model
        target_f = self.policy_model.predict(transition[0], batch_size=1)
        target_f[0][transition[1]] = target
        return target_f[0]

    def map_states(self, transition):
        return np.array(transition[0][0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(int(batch_size * 0.8))
        minibatch.extend(self.memoryDied.sample(int(batch_size * 0.2)))
        history = self.policy_model.fit(np.array(list(map(self.map_states, minibatch))),
                                        np.array(list(map(self.map_targets, minibatch))), verbose=0)
        self.update_epsilon()

        # Iterative update
        if self.step == self.C:
            self.target_model.set_weights(self.policy_model.get_weights())
            self.step = 0
        else:
            self.step += 1

        return history

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def create_model(self):
        model = Sequential()

        def masked_mse(y_true, y_pred):
            mask_true = K.cast(K.not_equal(y_true, y_pred), K.floatx())
            masked_squared_error = K.square(mask_true * (y_true - y_pred))
            r = K.sum(masked_squared_error, axis=-1)  # / K.sum(mask_true, axis=-1)
            return r

        # CNN
        model.add(Conv2D(32, kernel_size=8, activation='elu', input_shape=(80, 80, 1), strides=4))
        model.add(Conv2D(64, kernel_size=4, activation='elu', strides=2))
        model.add(Conv2D(64, kernel_size=3, activation='elu', strides=1))
        model.add(Flatten())

        # NN
        model.add(Dense(512, activation='elu'))
        model.add(Dense(self.action_size))
        model.compile(loss=masked_mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def save(self):
        self.target_model.save_weights('target_model.h5')
        self.policy_model.save_weights('policy_model.h5')

    def load(self):
        self.target_model.load_weights('target_model.h5')
        self.policy_model.load_weights('policy_model.h5')
        print(self.policy_model.get_weights())
