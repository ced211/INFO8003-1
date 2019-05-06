import os
import keras
import tensorflow as tf
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adamax, RMSprop
from catcher import ContinuousCatcher
from BaseAgent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, pretrained_model=None, env=ContinuousCatcher(), model_path="/models"):
        self.name = 'simple policy gradient agent'
        self.model_path = model_path
        if pretrained_model:
            self.model = load_model(pretrained_model)
        else:
            # give a lerning rate for the ADAM. Same then the gamma in the catcher???
            self.gamma = 0.0001
            # neural network model
            self.model = Sequential()
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(
                Dense(self.actions_available, activation='relu'))
            self.model.compile(loss=mse,
                               optimizer=Adam(lr=self.gamma))
            self.dmu = 0.00001
            self.dsigma = 0.00001
    def step(self, state):
        """
            return the action to take given the state.
        """
        mu,sigma = self.model.predict(state, batch_size=1)
        #sample from a gaussian
        return action

    def Derivmu(self,state):
        mu,sigma = self.model.predict(state,batch_size=1)
        deriv = (normal(mu,sigma) - normal (mu+self.dmu,sigma)) / dmu
        return dmu

    def Derivsigma(self,state):
        mu,sigma = self.model.predict(state,batch_size=1)
        deriv = (normal(mu,sigma) - normal (mu,sigma + self.dsigma)) / dmsigma
        return dmu    

    def train(self):
        rewards, state, actions = [], [], []
        train_x, train_y = [], []
        episode_number = 0
        cur_state = self.env.reset()
        last_observation = None
        action = self.step(state)
        next_state, reward, done = self.env.step(action)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        if done:
            episode_number += 1
            episode_state = np.vstack(state)
            episode_reward = np.vstack(reward)
            # Store current episode into training batch
            train_x.append(episode_state)
            train_y.append(episode_reward)

