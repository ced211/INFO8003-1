import os
import keras
import tensorflow as tf
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adamax, RMSprop
from catcher import ContinuousCatcher
from BaseAgent import BaseAgent


def discount_rewards(r, gamma=0.99):
    gamma = 0.99
    discounted_r = np.zeros(r.shape)
    discounted_r[0] = r[0]
    for t in range(1, r.size):
        discounted_r[t] = discounted_r[t-1] + r[t] * (gamma ** t)
    return discounted_r


class Agent(BaseAgent):
    def __init__(self, pretrained_model=None, actions_available=1, env=ContinuousCatcher(), model_path="/models"):
        """
        Init the agent with the necessary values
        """
        self.name = 'policy gradient agent'
        self.model_path = model_path
        # define the set of actions available. Certainement pas necessaire.
        self.actions_available = actions_available
        self.env = env
        if pretrained_model:
            self.model = load_model(pretrained_model)
        else:
            # give a lerning rate for the ADAM. Same then the gamma in the catcher???
            self.gamma = 0.0001
            # neural network model
            self.model = Sequential()
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(
                Dense(self.actions_available, activation='relu'))
            # check for the lost: what kind of lost it is!!
            self.model.compile(loss=categorical_crossentropy,
                               optimizer=Adam(lr=self.gamma))

        def step(self, state):
            """
            return the action to take given the state.
            """
            action = self.model.predict(state, batch_size=1)
            return action

        def train(self):
            # Init
            batch_size = 1
            rewards, state, actions = [], [], []
            train_x, train_y = [], []
            avg_reward = []
            reward_sum = 0
            episode_number = 0
            observation = self.env.reset()

            last_observation = None
            # Training progress
            while True:
                # Consider state difference and take action.
                action = self.step(observation)
                cur_observation, reward, done = self.env.step(action)
                if last_observation != None:
                    x = cur_observation - last_observation
                else:
                    x = cur_observation
                last_observation = cur_observation
                rewards.append(reward)
                state.append(x)
                actions.append(action)

                if done:
                    episode_number += 1
                    episode_state = np.vstack(state)
                    episode_reward = np.vstack(reward)
                    # # Discount and normalize rewards
                    discounted_ep_reward = discount_rewards(
                        episode_reward, self.env.gamma())
                    # discounted_ep_reward -= np.mean(discounted_ep_reward)
                    # discounted_ep_reward /= np.std(discounted_ep_reward)
                    # ep_dlogp *= discounted_ep_reward

                    # Store current episode into training batch
                    train_x.append(episode_state)
                    # Je sais pas quoi mettre comme output pour train le neural network???
                    # normalement reward * proba to take action Mais comment avoir proba(action) en continu ?
                    train_y.append(episode_reward)
                    # idee train le network pour output un std et mean pour un gaussian
                    if episode_number % batch_size == 0:
                        input_tr_y = self.learning_rate * \
                            np.squeeze(np.vstack(train_y))
                        self.model.train_on_batch(
                            np.vstack(train_x), input_tr_y)
                        train_x, train_y, prob_actions = [], [], []
                        # Checkpoint
                        self.model.save(self.model_path)

                    avg_reward.append(float(reward_sum))
                    # if len(avg_reward)>30: avg_reward.pop(0)
                    print('Epsidoe {:} reward {:.2f}, Last 30ep Avg. rewards {:.2f}.'.format(
                        episode_number, reward_sum, np.mean(avg_reward)))
                    print('{:.4f},{:.4f}'.format(reward_sum, np.mean(
                        avg_reward)), end='\n', file=log, flush=True)
                    reward_sum = 0
                    observation = self.env.reset()
