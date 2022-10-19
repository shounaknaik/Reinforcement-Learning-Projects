#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    # action

    action = 0 if score>=20 else 1

    

    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)


    ### V is calculated across episodes

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode

    for _ in range(n_episodes):

        current_state=env.reset()
        episode=[]

        done=False

        #### Simulating episodes####

        while not done:

            action=policy(current_state)

            next_state,reward,done,_,info=env.step(action)
            # print(next_state,state,reward,done)
            episode.append((current_state,action,reward))

            current_state=next_state

        ######### Calculate value for all state, action and reward.##########
        state_values=[]
        G=0
        for (state,action, reward) in reversed(episode):
            G=gamma*G+reward
            state_values.append(G)

        state_values.reverse()


        visited=[]

        for id,(state,action,reward) in enumerate(episode):

            if state in visited:
                continue
            visited.append(state)

            returns_count[state]+=1
            returns_sum[state]+=state_values[id]

            V[state]=returns_sum[state]/returns_count[state]


    return V


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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #


    max_action_idx=np.argmax(Q[state])
    probability_list=(np.ones(nA)*epsilon)/nA

    probability_list[max_action_idx]+=1-epsilon

    action_idx=np.random.choice(np.arange(len(Q[state])),p=probability_list)

    return action_idx



def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

        # define decaying epsilon


    count_action_space=env.action_space.n



    while epsilon>0:

        for _ in range(n_episodes):

            current_state=env.reset()

            episode=[]
            
            done=False


            ##### Generate the episodes ##########
            while not done:

                action=epsilon_greedy(Q,current_state,count_action_space,epsilon)

                next_state,reward,done,_,info=env.step(action)
                episode.append((current_state,action,reward))

                current_state=next_state

            
            state_values=[]
            G=0
            for (state,action, reward) in reversed(episode):
                G=gamma*G+reward
                state_values.append(G)

            state_values.reverse()

            visited=[]
            for idx,(state,action,reward) in enumerate(episode):

                if (state,action) in visited:
                    continue

                visited.append((state,action))

                returns_count[(state,action)]+=1
                returns_sum[(state,action)]+=state_values[idx]

                Q[state][action]=returns_sum[(state,action)]/returns_count[(state,action)]


            epsilon=epsilon-(0.1/n_episodes)



    return Q