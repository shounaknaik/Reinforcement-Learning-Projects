### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    
    value_function = np.zeros(nS)
    # prev_value_function=np.zeros(nS)
    # ############################
    # # YOUR IMPLEMENTATION HERE #

    while(True):

        delta=np.zeros(nS)

        for state_idx,state in enumerate(range(nS)):
            v_temp=value_function[state_idx]
            action_reward_total=0
            for action_idx,action_probabilty in enumerate(policy[state]):
            

                transition_lists=P[state][action_idx]

                transition_reward_total=0
                for transition_li in transition_lists:

                    transition_probability,next_state,reward,terminate_flag=transition_li

                    # print(transition_probability)
                    # print(next_state)

                    # print(action_probabilty*(transition_probability*(reward+gamma*value_function[next_state])))

                    transition_reward_total+=(transition_probability*(reward+gamma*value_function[next_state]))

                action_reward_total+=action_probabilty*transition_reward_total

            value_function[state_idx]=action_reward_total

            delta[state_idx]=max(delta[state_idx],abs(v_temp-value_function[state_idx]))

            

            # break

        if np.all(delta<tol):
            break

    return value_function

    

    # print(f'New value function is: \n {value_function}')


    
    ############################


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.zeros([nS, nA])
    
	############################
	# YOUR IMPLEMENTATION HERE #

    

        # old_policy=new_policy.copy()


    for state_idx in range(nS):

        this_states_action_values=np.zeros(nA)
        for action_idx in range(nA):

            

            transition_lists=P[state_idx][action_idx]

            for transition_li in transition_lists:

                transition_probability,next_state,reward,terminate_flag=transition_li

                this_states_action_values[action_idx]+=(transition_probability*(reward+gamma*value_from_policy[next_state]))

        max_action_idx=np.argmax(this_states_action_values)
        new_policy[state_idx][max_action_idx]=1.0

	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #

    while(True):
        old_policy=new_policy.copy()
        V=policy_evaluation(P,nS,nA,old_policy,tol=1e-8)
        new_policy=policy_improvement(P,nS,nA,V)

        if np.allclose(old_policy,new_policy,atol=tol):
            break

	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    while(True):

        delta=np.zeros(nS)

        for state_idx,state in enumerate(range(nS)):

            v_temp=V_new[state]

            this_states_action_values=np.zeros(nA)
            for action_idx,action in enumerate(range(nA)):

                transition_lists=P[state][action_idx]

                action_total_reward=0

                for transition_li in transition_lists:

                    transition_probability,next_state,reward,terminate_flag=transition_li

                    action_total_reward+=(transition_probability*(reward+gamma*V_new[next_state]))

                this_states_action_values[action_idx]=action_total_reward

            max_action_value=max(this_states_action_values)
            V_new[state_idx]=max_action_value
            delta[state_idx]=max(delta[state_idx],abs(v_temp-V_new[state_idx]))

        

        if np.all(delta<tol):
            break

    policy_new=policy_improvement(P,nS,nA,V_new)

    

    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [nS, nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    -----
    Transition can be done using the function env.step(a) below with FIVE output parameters:
    ob, r, done, info, prob = env.step(a) 
    """
    total_rewards = 0
    for _ in range(n_episodes):
        state = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #

            action_idx=np.argmax(policy[state])

            # print(policy[state][action_idx])

            state,reward,done,_,info=env.step(action_idx)
            # if state==15:
            #     done=True
            total_rewards+=reward
            
    return total_rewards


