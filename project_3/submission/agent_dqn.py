#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent import Agent
from dqn_model import DQN

from collections import namedtuple, deque

import matplotlib.pyplot as plt
import pickle

import copy
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


EPISODES = 50000
LEARNING_RATE = 1.5e-4  # alpha
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 500     ### These hyperparameters are accoriding to the slides 
EPSILON = 1.0
EPSILON_END = 0.025
FINAL_EXPL_FRAME = 1000000
TARGET_UPDATE_FREQUENCY = 1000
SAVE_MODEL_AFTER = 5000
DECAY_EPSILON_AFTER = 10000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



########### Adopted from Pytorch DQN tutorial!# ################


#################### https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html #############

reward_buffer = deque([0.0], maxlen=100)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):


        tuples=random.sample(self.memory,batch_size)


        return tuples


    def __len__(self):
        return len(self.memory)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.environment=env
        self.count_action=self.environment.action_space.n

        input_photos=4

        self.online_Q_net=DQN(input_photos,self.count_action).to(device)
        self.target_Q_net=copy.deepcopy(self.online_Q_net).to(device)

        self.experience_buffer=ReplayMemory(BUFFER_SIZE)

        self.optimizer = optim.Adam(self.online_Q_net.parameters(), lr=LEARNING_RATE)
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.online_Q_net=DQN(input_photos,self.count_action)
            self.online_Q_net.load_state_dict(torch.load('./dqn_model.pth',map_location=device))
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        observation=np.array(observation,dtype=np.float32)/255
        observation=observation.transpose(2,0,1)

        observation=np.expand_dims(observation,0)

        observation=torch.from_numpy(observation)

        logits=self.online_Q_net(observation.to(device))

        max_action=torch.argmax(logits)

        
        ###########################
        return max_action.detach().item()
    
        

    def train(self,n_episodes=EPISODES):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        episode_reward_li=[]

        count=0

        for episode_index in range(n_episodes):

            print(f'The current episode running is: {episode_index}')

            episode_reward=0
            current_state=self.env.reset()
            done=False

            

            while not done:

                if episode_index>DECAY_EPSILON_AFTER:
                    epsilon = np.interp(count, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                
                else:
                    epsilon=EPSILON

                # epsilon=np.interp(count,[])

                action=self.make_action(current_state)
                ## Get the epsilon Greedy action from the action

                probability_array = np.ones(self.count_action) * epsilon / self.count_action  # exploration
                probability_array[action] += 1 - epsilon  # exploitation

                final_action=np.random.choice(np.arange(self.count_action),p=probability_array) ### Taking the random action from the set of actions based 
                # on the probability.

                ## Take the next step in the environment based on the action ####
                next_state,reward,done,dk,info = self.env.step(final_action)


                episode_reward+=reward
                


                ## Converting current state into appropriate tensor ###
                t=np.array(current_state,dtype=np.float32)/255
                t=t.transpose(2,0,1)
                t=np.expand_dims(t,0)
                current_state_tensor=torch.from_numpy(t)

                ## Converting next state into appropriate tensor ###
                t=np.array(next_state,dtype=np.float32)/255
                t=t.transpose(2,0,1)
                t=np.expand_dims(t,0)
                next_state_tensor=torch.from_numpy(t)

                # print(current_state_tensor.shape)
                # print(next_state_tensor.shape)
                ## Action and Reward Tensor ##

                action_tensor = torch.tensor([action], device=device)
                reward_tensor = torch.tensor([reward], device=device)

                self.experience_buffer.push(current_state_tensor, action_tensor,next_state_tensor,reward_tensor)

                current_state = next_state


                # Optimize
                self.optimize()

                if done:
                    reward_buffer.append(episode_reward)
                    break

                count+=1
            

            if episode_index % TARGET_UPDATE_FREQUENCY == 0:
                self.target_Q_net.load_state_dict(self.online_Q_net.state_dict())

            if episode_index % SAVE_MODEL_AFTER == 0:
                torch.save(self.online_Q_net.state_dict(), "dqn_model.pth")

                with open('reward_li.pkl','wb') as f:
                    pickle.dump(episode_reward_li,f)

            # print('Going outside')

            episode_reward_li.append(episode_reward)

        torch.save(self.online_Q_net.state_dict(), "dqn_model.pth")

        with open('reward_li.pkl','wb') as f:
                pickle.dump(episode_reward_li,f)

        _=plt.figure()
        plt.plot(range(len(episode_reward_li)),episode_reward_li)
        plt.title('Reward Vs Episode Number')
        plt.xlabel('Episode Number')
        plt.ylabel('Reward')

        plt.savefig('rVsEpNum.png')

        print("Done!")
    
        ###########################

    def optimize(self):

        if len(self.experience_buffer) < BATCH_SIZE:
            return




        ## From PyTorch official tutorial ##### 
        
        transitions = self.experience_buffer.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        # print(transitions)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward)

    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        temp = self.online_Q_net(state_batch) 
        state_action_values = temp[torch.arange(temp.size(0)), action_batch] ### max action is chosen by this. 
        ## Picks the best action for 32 samples


        # print(non_final_next_states.shape)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_Q_net(non_final_next_states.float()).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_Q_net.parameters():
            param.grad.data.clamp_(-1, 1)  ### Clipping the gradients between -1 and 1
        self.optimizer.step()