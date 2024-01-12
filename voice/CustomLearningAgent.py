# CustomLearningAgent.py

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

class CustomLearningAgent:
    def __init__(self, seed, exploration_decay_steps=1000, **kwargs):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.exploration_rate = 1.0
        self.exploration_decay_steps = exploration_decay_steps
        self.learning_rate = 0.01

        self.q_table = np.zeros((state_space_size, action_space_size))
        self.memory = deque(maxlen=2000)

        self.model = self._build_model()

    def begin_episode(self, observation):
        self.exploration_rate *= (1 / self.exploration_decay_steps)
        self.exploration_rate = max(0.01, self.exploration_rate)

        if np.random.uniform(0, 1) <= self.exploration_rate:
            return random.randrange(action_space_size)
        else:
            state = self._discretize_state(observation)
            return np.argmax(self.q_table[state, :])

    def act(self, observation, reward):
        state = self._discretize_state(observation)

        if np.random.uniform(0, 1) <= self.exploration_rate:
            action = random.randrange(action_space_size)
        else:
            action = np.argmax(self.q_table[state, :])

        next_state = self._discretize_state(observation)
        self.memory.append((state, action, reward, next_state))

        self._replay()

        return action

    def _replay(self):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + discount_factor * np.max(self.q_table[next_state, :])
            self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                          self.learning_rate * target

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=state_space_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _discretize_state(self, observation):
        # Discretize the observation based on the environment.py provided earlier
        discretized_state = []

        for i, value in enumerate(observation):
            feature_range = self._get_feature_range(i)
            discretized_value = self._discretize_value(value, feature_range, self.discretization_steps)
            discretized_state.append(discretized_value)

        return tuple(discretized_state)

    def _get_feature_range(self, feature_index):
        # Define the feature ranges based on environment.py
        feature_ranges = [
            [-350, 350],  # User X - serv
            [-350, 350],  # User Y - serv
            [175, 875],   # User X - interf
            [-350, 350],  # User Y - interf
            [0, 40],      # Serving BS power
            [0, 40]       # Interfering BS power
        ]

        return feature_ranges[feature_index]

    def _discretize_value(self, value, feature_range, num_bins):
        return np.digitize(x=value, bins=np.linspace(feature_range[0], feature_range[1], num_bins + 1)[1:-1])

    def save(self, name):
        self.model.save(name + "_neural_network.h5")
        np.save(name + "_q_table.npy", self.q_table)

    def load(self, name):
        self.model = load_model(name + "_neural_network.h5")
        self.q_table = np.load(name + "_q_table.npy")