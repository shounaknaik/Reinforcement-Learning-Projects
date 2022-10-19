#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    max_action_idx=np.argmax(Q[state])
    probability_list=(np.ones(nA)*epsilon)/nA

    probability_list[max_action_idx]+=1-epsilon

    action_idx=np.random.choice(np.arange(len(Q[state])),p=probability_list)

    return action_idx


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #

    count_action_space=env.action_space.n

    for _ in range(n_episodes):

        epsilon=0.99*epsilon

        current_state=env.reset()
        current_action=epsilon_greedy(Q,current_state,count_action_space,epsilon)

        done=False

        while not done:

            next_state,reward,done,_,info=env.step(current_action)
            next_action=epsilon_greedy(Q,next_state,count_action_space,epsilon)

            Q[current_state][current_action]=Q[current_state][current_action]+alpha*(reward+gamma*Q[next_state][next_action]-Q[current_state][current_action])

            current_state=next_state
            current_action=next_action

    

        

    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #

    count_action_space=env.action_space.n
    

    for _ in range(n_episodes):
        epsilon = 0.99*epsilon  
        current_state = env.reset()  

        done = False

        while not done:  
            current_action = epsilon_greedy(Q, current_state, count_action_space, epsilon)  
            next_state, reward, done,_, info = env.step(current_action)  


            max_action = np.argmax(Q[next_state])
            

            Q[current_state][current_action] = Q[current_state][current_action]+alpha*(reward+gamma*Q[next_state][max_action]-Q[current_state][current_action]) 

            current_state = next_state

    ############################
    return Q
