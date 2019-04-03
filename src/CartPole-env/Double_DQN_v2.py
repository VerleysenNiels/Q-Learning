from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import keras.backend as K
import random
import numpy as np
from Replay_Memory import ReplayMemory


class DoubleDQN:

    def __init__(self, action_size, gamma=0.9, eps_dec=0.9, lr=1.0e-3):
        self.action_size = action_size
        self.memory = ReplayMemory(1000)
        # Discount rate
        self.gamma = gamma
        # Setup epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
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

    # Act-method from paper
    def act_eps_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return self.act(state)

    # Return a weighted random action where the estimated Q-values are used as weights
    # Also apply epsilon-greedy action selection
    def act_stochastic(self, state):
        if np.random.rand() <= self.epsilon:
            print("random")
            return random.randrange(self.action_size)
        print("network")
        act_values = self.policy_model.predict(state, batch_size=1)
        print(act_values)
        e = np.exp(act_values[0] - np.max(act_values[0]))
        p = e / np.sum(e, axis=-1)
        return np.random.choice(self.actions, p=p)

    def map_targets(self, transition):
        target = transition[3]  # 0 now manually set at main.
        if not transition[4]:
            target = transition[3] + self.gamma * self.target_model.predict(transition[2])[0][np.argmax(self.policy_model.predict(transition[2])[0])]
        # Update policy_model
        target_f = self.policy_model.predict(transition[0], batch_size=1)
        target_f[0][transition[1]] = target
        return target_f[0]

    def map_states(self, transition):
        return np.array(transition[0][0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        history = self.policy_model.fit(np.array(list(map(self.map_states, minibatch))), np.array(list(map(self.map_targets, minibatch))), verbose=0)
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

    # This method defines the used architecture, current architecture is the best I found so far.
    def create_model(self):
        def masked_mse(y_true, y_pred):
            mask_true = K.cast(K.not_equal(y_true, y_pred), K.floatx())
            masked_squared_error = K.square(mask_true * (y_true - y_pred))
            r = K.sum(masked_squared_error, axis=-1)  # / K.sum(mask_true, axis=-1)
            return r

        model = Sequential()
        model.add(Dense(6, activation='elu', input_shape=(4,)))
        model.add(Dense(self.action_size))#, bias_initializer=Constant(100)))
        model.compile(loss=masked_mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    # Use these methods to save and load weights
    def save(self):
        self.target_model.save_weights('target_model.h5')
        self.policy_model.save_weights('policy_model.h5')

    def load(self):
        self.target_model.load_weights('target_model.h5')
        self.policy_model.load_weights('policy_model.h5')
        print(self.policy_model.get_weights())
