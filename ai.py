# Importação das bibliotecas

from typing import Dict, List
import numpy as np
import random
import os
from numpy.core.defchararray import array
import tensorflow as tf

# Criação da arquitetura da rede neural


def loss(actual, predict):
    predMean = tf.math.reduce_mean(predict)
    m3 = actual + predMean
    return tf.math.reduce_mean(m3 - predMean)


class Network():
    def __init__(self, input_size, nb_action):
        self.input_size = input_size
        self.nb_action = nb_action

        # 5 -> 50 -> 3 - full connection (dense)
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(
            30, activation='tanh', input_shape=(self.input_size,)))
        self.model.add(tf.keras.layers.Dense(
            self.nb_action, activation='softmax'))
        # compila a rede
        self.model.compile(optimizer='adam', loss=loss, metrics=[
                           'mean_squared_error'])
        self.model.summary()

    def forward(self, state):
        x = self.model.predict([state],)
        return x[0]

# Implementação do replay de experiência


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # D,E,F,G,H
    # 4 valores: último estado, novo estado, última ação, última recompensa
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # retorna 100 amostras diferentes
        samples = zip(*random.sample(self.memory, batch_size))
        samples = tuple(samples)
        return np.asarray(samples[0]), np.asarray(samples[1]), np.asarray(samples[2]), np.asarray(samples[3])


# Implementação de Deep Q-Learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.updateCount = 0
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(1000)
        self.last_state = np.zeros(input_size)
        self.last_action = 0
        self.last_reward = 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_action(self, state):
        # softmax(1,2,3) -> (0.04, 0.11, 0.85) -> (0, 0.02, 0.98)
        probabilities = self.softmax(self.model.forward(state) * 100)
        # escolhe a ação com base em uma probabilidade
        action = np.random.choice(range(len(probabilities)), p=probabilities)
        return action

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):

        outputs = self.model.model.predict(batch_state,)
        outputs = np.max(np.take_along_axis(outputs, np.expand_dims(
            batch_action, axis=1), axis=1), axis=1)
        next_outputs = self.model.model.predict(batch_next_state,)
        next_outputs = np.max(next_outputs, axis=1)

        target = self.gamma * next_outputs + batch_reward
        error = np.square(outputs - target)
        self.model.model.fit((batch_state,), (error,), epochs=1)

    def update(self, reward, new_state):
        self.updateCount = self.updateCount + 1
        self.memory.push((self.last_state, new_state,
                          self.last_reward, int(self.last_action)))
        action = self.select_action(new_state)
        if (self.updateCount > 20 and len(self.memory.memory) > 100):
            self.updateCount = 0
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(
                100)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Carregado com sucesso')
        else:
            print('Erro ao carregar')
