import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from itertools import product
from scipy.optimize import fsolve
from matplotlib.collections import LineCollection

from abc import ABC, abstractmethod
from joblib import Parallel, delayed

"""
This module contains the classes and methods I used throughout my thesis to simulate and analyze MARL dynamics. 
It is grown over the span of a year and contains a lot of confusing and redundant code that is only understandable to me. 
I will try to clean it up in the future.

A lot of the design choices were made in the beginning to be able to extend the code to repeated normal-form and stochastic games, which ended up not being necessary for my thesis. 
Most of the implemented code is not used in the final version of my thesis and not tested.
"""

################################### Agent classes ###################################

class Agent(ABC): 
    """ 
    This class implements an agent that can play a multiplayer prisoners dilemma game.
    """

    # parameters of the agent:
    agent_id = None
    """The id of the agent."""
    player_id = None
    """The id of the player that this agent will play.""" 
    num_players = None
    """The number of players of the game."""
    action_space = None
    """The action space of the agent."""
    num_actions = None
    """The number of actions of the agent."""
    discount_factor = None
    """The discount factor of the agent."""
    selection_method = None
    """The selection method of the agent."""
    temperature = None
    """The temperature of the agent."""
    reward_func = None
    """The reward function of the agent."""
    observation_length = None #TODO: rename to observation length to be consistent with the literature
    """The observation length of the agent."""

    # learning hyperparameters of the agent:
    learning_rate = None
    """The learning rate of the agent."""
    exploration_rate = None
    """The exploration rate value of the agent."""

    # variables changing during the episode from time step to time step:
    state = None
    """The state of the agent."""
    state_history = None
    """The state history of the agent."""
    action = None
    """The action of the agent."""
    reward = None
    """The utility of the agent."""
    observation = None
    """The observation of the agent.""" 

    # learning-related variables changing during the episode from time step to time step:
    q_table = None
    """The Q-table of the agent."""
    q_table_history = None
    """The Q-table history of the agent."""

    # variables to be saved in history:
    q_table_history = None
    """The Q-table history of the agent."""

    def __init__(self, 
                 player_id = None,
                 action_space = None, 
                 learning_rate = 0.1, 
                 discount_factor=0.0, 
                 exploration_rate=0.1,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="Boltzmann"):
        """
        This function initializes an agent object.

        Args:
            player_id (int): The id of the player that this agent will play.
            action_space (array): The action space of the agent.
            learning_rate (float, optional): Defaults to 0.1. The learning rate of the agent. 
            discount_factor (float, optional): Defaults to 0.9. The discount factor of the agent.
            exploration_rate (float, optional): Defaults to 0.2. The exploration rate value of the agent.
            cooperation_probability (float, optional): Defaults to None. The cooperation probability of the agent.
            num_players (int, optional): Defaults to None. The number of players of the game.
            observation_length (int, optional): Defaults to 0. The observation length of the agent.
            temperature (int, optional): Defaults to 1. The temperature of the agent.
            reward_func (method, optional): Defaults to None. The reward function of the agent. Requirements: 
                                            Input: an action_vector array and the player_id of the agent. 
                                            Returns: a reward value (float). 
            state (int, optional): Defaults to None. The state of the agent.
            q_table (numpy.array, optional): Defaults to None. The Q-table of the agent.
            agent_id (int, optional): Defaults to None. The id of the agent.
            selection_method (str, optional): Defaults to "epsilon_greedy". The selection method of the agent.
        """     

        assert player_id is not None, "The player_id has to be specified."
        assert action_space is not None, "The action space has to be specified."
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= player_id < num_players, "The player_id has to be between 0 and the number of players."
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= discount_factor <= 1, "The discount factor has to be between 0 and 1."
        assert 0 <= exploration_rate <= 1, "The epsilon value has to be between 0 and 1."
        assert 0 <= learning_rate <= 1, "The learning rate has to be between 0 and 1." #TODO: how to throw an error if learning_rate is changed to a value outside of [0,1] during the learning process?
        assert num_players is not None, "The number of players has to be specified."
        assert 0 <= observation_length, "The observation_length has to be a positive integer."
        assert 0 <= temperature, "The temperature has to be a positive float."

        self.player_id = player_id
        self.agent_id = agent_id
        self.num_players = num_players
        self.action_space = action_space #TODO: Should be a list of action spaceS of the agents. Each entry of the list should be one action_space of the specific player.
        #TODO: implementiere möglichkeit dass man eine Liste oder nur ein einziges action_space übergeben kann
        self.num_actions = len(action_space) #das wäre dann auch eine Liste --> len(action_space[0]) --> das ist die Anzahl der Aktionen des ersten Spielers
        self.observation_length = observation_length
        self.observation = '' # empty string to save the observation of the agent
        self.num_states = (self.num_actions**self.num_players)**self.observation_length
        if state == None:
            self.state = - self.observation_length
        else:
            self.state = state
        self.state_history = []

        self._learning_rate = learning_rate # Initialize learning_rate with the provided value
        self.discount_factor = discount_factor
        self.selection_method = selection_method # by default: Boltzmann if not specified otherwise
        self.exploration_rate = exploration_rate # for epsilon-greedy selection method
        self.temperature = temperature # by default: 1 if not specified otherwise
        self.reward_func = reward_func # by default: None if not specified otherwise

        if q_table is None:
            self.q_table = np.zeros((self.num_states, self.num_actions)) # initialize Q-table, shape: (num_states, num_actions), num_states = (2**num_players)**observation_length
            self.initial_q_table = self.q_table.copy()
        else:
            self.q_table = q_table 
            self.initial_q_table = self.q_table.copy()
        self.q_table_history = [self.initial_q_table.copy()]
        
    # make learning_rate a property to ensure that it stays within [0, 1]
    @property
    def learning_rate(self):
        return self._learning_rate
    # setter method for learning_rate
    @learning_rate.setter
    def learning_rate(self, value):
        # Ensure learning_rate stays within [0, 1]
        if 0 <= value <= 1:
            self._learning_rate = value
        else:
            raise ValueError("Invalid learning rate: {}. Learning rate must be in the range [0, 1].".format(value))
    
    def update_policy(self, current_info):
        pass

    def update_attributes(self, current_info):
        """
        This function updates the attributes of the agent.

        Args:
            current_info (dict): Dictionary containing 'state', 'action_vector', 'reward' and 'next_state'.
        """
        state = current_info['state']
        self.state_history.append(state) # save state in attribute of agent
        next_state = current_info['next_state']
        self.state = next_state

    def reset(self):
        """
        This function resets the agent.
        """
        #print("function reset is called")
        self.state = - self.observation_length # reset state
        self.state_history = [] # reset state history
        #reset q_table
        self.q_table = self.initial_q_table.copy()
        self.q_table_history = [self.initial_q_table.copy()] # reset Q_history
        
    def get_action_probabilities(self, q_table=None):
        """
        This function calculates the action probabilities of the agent.

        Args:
            q_table (numpy.array): The Q-table of the agent.

        Returns:
            numpy.array: The action probabilities of the agent.
        """
        if q_table is None:
            q_table = self.q_table
        
        if self.selection_method == "epsilon_greedy":
            num_rows, num_cols = q_table.shape
            non_max_prob = self.exploration_rate / (num_cols - 1)
            # Initialize new array
            action_probabilities = np.full((num_rows, num_cols), non_max_prob)
            # Get indices of max values in each row
            max_indices = np.argmax(q_table, axis=1)
            # Set max value positions to (1 - epsilon)
            action_probabilities[np.arange(num_rows), max_indices] = 1 - self.exploration_rate
        
        if self.selection_method == "Boltzmann":
            temperature = self.temperature
            action_probabilities = np.exp(q_table / temperature) / np.sum(np.exp(q_table / temperature), axis=1)[:, np.newaxis] 

        return action_probabilities
    
    def choose_action(self, state): #TODO: aufspalten in zwei methoden 
        """
        This function chooses an action for the agent. It also updates the action attribute of the agent.
        For the first observation_length time steps, the agent chooses the action according to a fixed strategy (always cooperate / defect, choose randomly).

        Args:
            state (int): The state of the agent.

        Returns:
            action (int): The chosen action of the agent.
        """
        # The first obersvation_length time steps have negative states. For these states, choose the actions according to a fixed strategy (always cooperate / defect, choose randomly)
        if state < 0:
            # always cooperate the first obervation_length steps
            self.action = 1
            return self.action

        action_chosen = np.random.choice(self.action_space, p=self.get_action_probabilities(self.q_table)[state])
        self.action = action_chosen
        return self.action

    def calculate_reward(self, action_vector):
        """
        This function calculates the reward of the agent

        Args:
            action_vector (numpy.array): array of the actions of all players

        Raises:
            ValueError: "The reward function has to be specified."

        Returns:
            reward (float): reward
        """

        ''' 
        This function calculates the utility of the agent. It also updates the utility attribute of the agent.
        :param action_vector: The action vector of the agents. 
        :param utility_function: The utility function of the agent.
        :return: The utility of the agent. 
        '''
        if self.reward_func is None: 
            raise ValueError("The reward function has to be specified.")
        
        self.reward = self.reward_func(action_vector, self.player_id)
        return self.reward
    
    def observe(self, current_info):
        """
        This function updates the policy and the attributes of the agent.

        Args:
            current_info (list): List of dictionaries containing the current information which is presented to the agents.
        """
        #print("observe is called")
        self.update_policy(current_info)
        self.update_attributes(current_info)

    def update_observation(self, action_vector):
        """
        This function updates the observation of the agent by appending the action_vector to the observation.
        If the observation_length is zero, no observation is maintained.
        If the number of time steps played is greater than the observation_length, 
        the first num_players digits are cut off from the existing observation before appending the new action_vector.

        Args:
            action_vector (numpy.array): The action vector of the current time step.
        """
        # Return immediately if observation_length is zero
        if self.observation_length == 0:
            return

        action_str = ''.join(map(str, action_vector.astype(int)))
        if len(self.observation) < (self.num_players * self.observation_length):
            self.observation += action_str
        else:
            # Remove the oldest action_vector and append the new one
            self.observation = self.observation[self.num_players:] + action_str

    def get_next_state(self, action_vector):
        if self.state >= 0:
            self.update_observation(action_vector)
            next_state = self.translate_key_to_state(self.observation)
            return next_state
        else:
            next_state = self.state + 1 if self.state < -1 else self.translate_key_to_state(self.observation)
            return next_state

    def translate_key_to_state(self, state_key):
        """
        This function translates a key to a state. A key is a string of 0s and 1s with length num_players * observation_length. It is a binary representation of the actions of the last observation_length time steps of all players.
        The state is an integer between 0 and num_states-1. It is a integer representation of the key. The state can be used to access the Q-table.

        Args:
            state_key (string): binary representation of the actions of the last observation_length time steps of all players

        Raises:
            ValueError: _description_

        Returns:
            int: state as an integer between 0 and num_states-1
        """
        key_length_max = self.num_players * self.observation_length # maximum length of key
        if state_key == '': 
            state = 0 # if the key is empty, the state is 0 because there is only one state 
            return state
        elif len(state_key) == key_length_max:
            state = int(state_key, 2)
            return state
        else:
            raise ValueError(f'There is a problem with the key (key: {state_key}).')
    
    def translate_state_to_key(self, state):
        """
        This function translates a state to a key. A key is a string of 0s and 1s with length num_players * observation_length. It is a binary representation of the actions of the last observation_length time steps of all players.
        The state is an integer between 0 and num_states-1. It is a integer representation of the key. The state can be used to access the Q-table.

        Args:
            state (int): state as an integer between 0 and num_states-1

        Raises:
            ValueError: _description_

        Returns:
            float: binary representation of the actions of the last observation_length time steps of all players
        """
        key_length_max = self.num_players * self.observation_length # maximum length of key
        if state < 0 or state >= 2 ** key_length_max:
            raise ValueError(f'State {state} is out of range for key length {key_length_max}.')

        binary_representation = format(state, f'0{key_length_max}b')
        return binary_representation

    def get_learning_history(self):
        """
        This function returns the learning history of the agent. 

        Returns:
            dictionary: dictionary with the learning history of the agent
        """
        return { "q_table": self.q_table_history }

class QLearningAgent(Agent):
    """
    This class implements a Q-learning agent that can play a multiplayer prisoners dilemma game.

    Args:
        Agent (Agent): parent class.
    """
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.0, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        # Add any QLearningAgent-specific initialization here
        self.name = "QL"

        if use_prefactor:
            self.prefactor = (1 - self.discount_factor)
        else:
            self.prefactor = 1

    def update_policy(self, current_info):
        """
        This function updates the Q-table of the QLearningAgent according to the Q-Learning algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss:
        "factor (1 - self.discount_factor) normalizes the state- action values to be on the same numerical scale as the rewards." - Barfuss "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4

        Args:
            current_info (dict): Dictionary containing the current information which is presented to the agents.
        """
        state = current_info['state']
        action = current_info['action']
        reward = current_info['reward']
        next_state = current_info['next_state']
        action_id = np.where(self.action_space == action)

        # update Q-value
        self.q_table[state, action_id] = (1 - self.learning_rate) * self.q_table[state, action_id] + self.learning_rate * ( self.prefactor * reward + self.discount_factor * np.max(self.q_table[next_state, :]) ) 
        self.q_table_history.append(self.q_table.copy()) 
        
class SarsaAgent(Agent):
    """
    This class implements a SARSA agent that can play a multiplayer prisoners dilemma game.

    Args:
        Agent (_type_): _description_
    """
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        # Add any Sarsa-specific initialization here
        self.name = "SARSA"
        if use_prefactor:
            self.prefactor = (1 - self.discount_factor)
        else:
            self.prefactor = 1
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
    
    def reset(self):
        super().reset()
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

    def update_policy(self, current_info):
        """
        This function updates the Q-table of the SarsaAgent according to the SARSA algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss:
        "factor (1 - self.discount_factor) normalizes the state- action values to be on the same numerical scale as the rewards." - Barfuss "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4


        Args:
            current_info (dict): Dictionary containing 'prev_state', 'prev_action', 'prev_reward', 'state', 'action', 'reward' and 'next_state'.
        """
        prev_state = current_info['prev_state']
        prev_action = current_info['prev_action']
        prev_reward = current_info['prev_reward']
        state = current_info['state']
        action = current_info['action']
        prev_action_id = np.where(self.action_space == prev_action)
        action_id = np.where(self.action_space == action)

        # don't update policy if prev_state is None or negative
        if prev_state == None or prev_state < 0:
            return

        # update Q-value
        self.q_table[prev_state, prev_action_id] = (1 - self.learning_rate) * self.q_table[prev_state, prev_action_id] + self.learning_rate * ( self.prefactor * prev_reward + self.discount_factor * self.q_table[state, action_id] ) 
        self.q_table_history.append(self.q_table.copy())   
        
    def update_attributes(self, current_info):
        """
        This function updates the attributes of the SarsaAgent.

        Args:
            current_info (dict): Dictionary containing 'state', 'action', 'reward' and 'next_state'.
        """
        state = current_info['state']
        action = current_info['action']
        reward = current_info['reward']
        next_state = current_info['next_state']

        self.state_history.append(state) # save state in attribute of agent

        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward
        self.state = next_state

class ExpSarsaAgent(SarsaAgent):
    """
    This class implements an expected SARSA agent that can play a multiplayer prisoners dilemma game.

    Args:
        SarsaAgent (_type_): _description_
    """
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method, 
                         use_prefactor)
        # Add any ExpSarsa-specific initialization here
        self.name = "ExpSARSA"
        if use_prefactor:
            self.prefactor = (1 - self.discount_factor)
        else:
            self.prefactor = 1
    
    def update_policy(self, current_info):
        """
        This function updates the Q-table of the ExpSarsaAgent according to the expected SARSA algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss:
        "factor (1 - self.discount_factor) normalizes the state- action values to be on the same numerical scale as the rewards." - Barfuss "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4

        Args:
            current_info (dict): Dictionary containing 'prev_state', 'prev_action', 'prev_reward', 'state', 'action', 'reward' and 'next_state'.
        """
        prev_state = current_info['prev_state']
        prev_action = current_info['prev_action']
        prev_reward = current_info['prev_reward']
        state = current_info['state']
        action = current_info['action']
        prev_action_id = np.where(self.action_space == prev_action)
        action_id = np.where(self.action_space == action)

        # don't update policy if prev_state is None or negative
        if prev_state == None or prev_state < 0:
            return

        # calculate expected Q-Value of next state
        expected_q_value = np.sum(self.get_action_probabilities(self.q_table)[state] * self.q_table[state])
        # update Q-value
        self.q_table[prev_state, prev_action_id] = (1 - self.learning_rate) * self.q_table[prev_state, prev_action_id] + self.learning_rate * ( self.prefactor * prev_reward + self.discount_factor * expected_q_value ) 
        self.q_table_history.append(self.q_table.copy())   

class TitForTatAgent(Agent):
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 use_prefactor = False,
                 selection_method="epsilon_greedy"):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        self.name = "TitForTat"

    def choose_action(self, state):
        # The first obersvation_length time steps have negative states. For these states, choose the actions according to a fixed strategy (always cooperate / defect, choose randomly)
        if state < 0:
            # always cooperate the first obervation_length steps
            self.action = 1
            return self.action

        if state > 3 or state < 0:
                raise "The states allowed for tit-for-tat strategy are only 0, 1, 2, 3 but the state is something else"
        
        def get_opposite_index(index): # Assuming index is either 0 or 1
            return 1 - index
        
        state_key = self.translate_state_to_key(state)
        previous_action_of_opponent = int( state_key[get_opposite_index(self.player_id)] )
        action_chosen = previous_action_of_opponent

        self.action = action_chosen
        return self.action

class FixedAgent(Agent):
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 prob_to_cooperate = 0.2, 
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy"):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        self.name = "Fixed"
        self.prob_to_cooperate = prob_to_cooperate


    def choose_action(self, state):
        # choose action with probability prob_to_cooperate
        if np.random.rand() < self.prob_to_cooperate:
            action_chosen = 1
        else:
            action_chosen = 0

        self.action = action_chosen
        return self.action

class FreqAdjustedQLearningAgent(QLearningAgent):
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = False,
                 learning_rate_adjustment = None):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method, 
                         use_prefactor)
        self.name = "FreqAdjustedQL"
        if learning_rate_adjustment is None:
            self.learning_rate_adjustment = learning_rate
        else:
            self.learning_rate_adjustment = learning_rate_adjustment #learning_rate_adjustment

    def update_policy(self, current_info):
        """
        This function updates the Q-table of the QLearningAgent according to the Q-Learning algorithm.
        The prefactor (1 - self.discount_factor) is missing in the formula in the book (Sutton & Barto, 2018, p. 131. It is taken from 2021 paper by Barfuss:
        "factor (1 - self.discount_factor) normalizes the state- action values to be on the same numerical scale as the rewards." - Barfuss "Dynamical systems as a level of cognitive analysis of multi-agent learning" 2021, p. 4

        Args:
            current_info (dict): Dictionary containing the current information which is presented to the agents.
        """
        state = current_info['state']
        action = current_info['action']
        reward = current_info['reward']
        next_state = current_info['next_state']
        action_id = np.where(self.action_space == action)

        # get probability to choose the action
        action_probability = self.get_action_probabilities(self.q_table)[state][action_id]

        # update Q-value
        self.q_table[state, action_id] = self.q_table[state, action_id] + min(self.learning_rate_adjustment/action_probability, 1) * self.learning_rate * ( self.prefactor * reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action_id]) 
        self.q_table_history.append(self.q_table.copy()) 

class CrossLearningAgent(Agent):
    def __init__(self, 
                 player_id,
                 action_space, 
                 learning_rate = 0.1, 
                 discount_factor = 0.9, 
                 exploration_rate = 0.2,
                 num_players = None,
                 observation_length = 0,
                 temperature = 1,
                 reward_func = None,
                 state = None, 
                 q_table = None,
                 agent_id = None,
                 selection_method="epsilon_greedy",
                 use_prefactor = True,
                 policy = np.array([0, 0])):
        super().__init__(player_id,
                         action_space, 
                         learning_rate, 
                         discount_factor, 
                         exploration_rate,
                         num_players,
                         observation_length,
                         temperature,
                         reward_func,
                         state, 
                         q_table,
                         agent_id,
                         selection_method)
        self.name = "CrossLearning"
        self.num_states = 0

        if policy is None:
            self.policy = np.zeros((self.num_states, self.num_actions)) # initialize Q-table, shape: (num_states, num_actions), num_states = (2**num_players)**observation_length
            self.initial_policy = self.policy.copy()
        else:
            self.policy = policy
            self.initial_policy = self.policy.copy()
        self.policy_history = [self.policy.copy()]

    def get_action_probabilities(self, q_table=None):
        """
        This function returns the action probabilities of the agent.

        Returns:
            numpy.array: The action probabilities of the agent.
        """
        #print("get_action_probabilities of CrossLearning is called")
        return self.policy
    
    def update_policy(self, current_info):
        #print("update_policy of CrossLearning is called")
        state = current_info['state']
        action = current_info['action']
        reward_original = current_info['reward']
        action_id = np.where(self.action_space == action)[0]

        # import product from itertools
        # get all possible action_vectors
        all_possible_action_vectors = np.array(list(product(self.action_space, repeat = self.num_players)))
        # for all possible action_vectors calculate the reward and get the maximum
        reward_max = max(self.calculate_reward(action_vector) for action_vector in all_possible_action_vectors)
        # normalize the reward
        reward = reward_original / reward_max
        
        for b in range(len(self.action_space)):
            if b == action_id:
                #print(" b == action_id")
                self.policy[state, b] += self.learning_rate * (reward - self.policy[state, b] * reward)
            else:
                #print(" b != action_id")
                self.policy[state, b] -= self.learning_rate * self.policy[state, b] * reward
        
        # update policy history
        self.policy_history.append(self.policy.copy())

    def reset(self):
        """
        This function resets the agent.
        """
        #print("function reset is called")
        self.state = - self.observation_length # reset state
        self.state_history = [] # reset state history
        #reset q_table
        self.q_table = self.initial_q_table.copy()
        self.q_table_history = [self.initial_q_table.copy()] # reset Q_history
        #reset policy
        self.policy = self.initial_policy.copy()
        self.policy_history = [self.initial_policy.copy()] # reset policy_history
################################### Game class ###################################

class Game:
    """
    This class implements a game that can be played by multiple agents. 
    Learning algorithm should be a seperate class or attribute of the agents
    """

    actions_history = None
    """The actions history of all steps of the game."""
    rewards_history = None
    """The rewards history of all steps of the game."""

    def __init__(self, agents):
        """
        This function initializes a game object.

        Args:
            agents (list): list of agents that play the game. One agent is one player. One agent can play multiple players. One agent is of class Agent.
        """
        self.agents = agents
        self.num_agents = len(agents)

        self.actions_history = [] # empty list to save actions of all agents
        self.rewards_history = [] # empty list to save rewards of all agents

    def reset(self):
        """
        This function resets the actions history and rewards history of the game.
        """
        # reset all attributes of the game?
        self.actions_history = [] # reset actions_history
        self.rewards_history = [] # reset rewards_history

    def step(self):
        """
        This function executes one time step of the game. It calculates the actions and rewards of the agents.
        The actions and the rewards are saved in the history of the game. 
        It returns a list of dictionaries, each containing the current information which is presented to the agents.
        The current information depends on the type of agent. For example, a Q-learning agent needs the current state, action, reward and next state.

        Returns:
            list: list of dictionaries, each containing the current information which is presented to the agents
        """
        #print("step is called")
        # choose actions based on current states
        action_vector = np.array([agent.choose_action(agent.state) for agent in self.agents])
        reward_vector = np.array([agent.calculate_reward(action_vector) for agent in self.agents])
        # History updates
        self.actions_history.append(action_vector) # save action_vector in attribute of game
        self.rewards_history.append(reward_vector) # save reward_vector in attribute of game

        # calculate next states
        for agent in self.agents:
            agent.next_state = agent.get_next_state(action_vector)

        # prepare current_info_vector for update of policy
        current_info_vector = []
        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                current_info_vector.append({'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
            if isinstance(agent, SarsaAgent):
                current_info_vector.append({'prev_state': agent.prev_state, 'prev_action': agent.prev_action, 'prev_reward': agent.prev_reward, 'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
            if isinstance(agent, TitForTatAgent):
                current_info_vector.append({'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
            #if isinstance(agent, FixedAgent):
            #    current_info_vector.append({'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
            if isinstance(agent, CrossLearningAgent):
                current_info_vector.append({'state': agent.state, 'action': agent.action, 'reward': agent.reward, 'next_state': agent.next_state})
        
        return current_info_vector

    def is_Nash_equilibrium(self, action_vector):
        for player_id in range(self.num_agents):
            for action in self.agents[player_id].action_space:
                if action == action_vector[player_id]:
                    continue
                old_reward = self.agents[player_id].calculate_reward(action_vector)
                print(f"action_vector: {action_vector}, old_reward: {old_reward}")
                new_action_vector = action_vector.copy()
                new_action_vector[player_id] = action
                new_reward = self.agents[player_id].calculate_reward(new_action_vector)

                if new_reward > old_reward:
                    return False
                elif new_reward == old_reward:
                    print(f"not a strict Nash equilibrium but maybe a simple Nash equilibrium: {action_vector}")

        return True

    def calculate_Nash_equilibria(self):
        """
        This function calculates the Nash equilibrium of the game.

        Returns:
            numpy.array: Nash equilibrium of the game
        """
        from itertools import product

        # check if all possible action vectors are Nash equilibria
        all_possible_action_vectors = np.array(list(product(self.agents[0].action_space, repeat = self.num_agents)))
        Nash_equilibria = []
        for action_vector in all_possible_action_vectors:
            if self.is_Nash_equilibrium(action_vector):
                Nash_equilibria.append(action_vector)
        
        return Nash_equilibria

    def get_reward_matrix(self, agent):
        """
        This function calculates the reward tensor of an agent.

        Args:
            agent (Agent): agent of the game

        Returns:
            numpy.array: reward tensor of the agent
        """
        reward_matrix_agent = np.array([[agent.calculate_reward([0, 0]), agent.calculate_reward([0, 1])], 
                                            [agent.calculate_reward([1, 0]), agent.calculate_reward([1, 1])]])
        return reward_matrix_agent

    def get_exp_reward(self, agents, player_id, action=1):
        """
        This function calculates the expected reward of an agent.

        Args:
            agent (Agent): agent of the game

        Returns:
            float: expected reward of the agent
        """
        agent = agents[player_id]
        other_agent = agents[1 - player_id]
        reward_matrix = self.get_reward_matrix(agent)
        action_probabilities_opponent = other_agent.get_action_probabilities(other_agent.q_table)
        exp_reward_of_action = reward_matrix[action] * action_probabilities_opponent
        return exp_reward_of_action
        
        
################################### Simulation class ###################################

class Simulation:
    """
    This class implements a simulation of a game. It can run multiple episodes of a game.
    """

    def __init__(self):
        pass

    def reset(self, game, agents):
        """
        This function resets the history of the game and the agents.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
        """
        game.reset()
        for agent in agents:
            agent.reset()

    def run_time_step(self, game, agents):
        """
        This function runs one episode of the game. At the moment one episode is one time step of the game.
        After one episode is finished, the learning values of the agents are updated via the observe function of the agents.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
        """
        # execute one time step of the game and get the information of the current time step
        current_info_vector = game.step() 
        #print("current_info_vector in run_time_step function: ", current_info_vector)

        # update the learning values of the agents
        for agent, current_info in zip(agents, current_info_vector):
            agent.observe(current_info)

    def run(self, game, agents, num_time_steps, learning_rate_func=None, temperature_func=None):
        """
        This function runs multiple episodes of the game.

        Args:
            game (Game): game object
            agents (list): list of agents that play the game. One agent is of class Agent.
            num_episodes (int): number of episodes to run
            learning_rate_func (method, optional): Defaults to None. The learning rate function of the agents. Requirements: episode as args. Returns: learning_rate (float).
            temperature_func (method), optional): Defaults to None. The temperature function of the agents. Requirements: episode as args. Returns: temperature (float).
        """

        self.reset(game, agents) # reset history of game and agents

        for time_step in range(num_time_steps):
            # if learning_rate_func is not None, update learning rate of agents according to given function
            if learning_rate_func is not None:
                for agent in agents:
                    agent.learning_rate = learning_rate_func(time_step)
            
            # if temperature_func is not None, update temperature of agents according to given function
            if temperature_func is not None:
                for agent in agents:
                    agent.temperature = temperature_func(time_step)
            
            self.run_time_step(game, agents)

    def visualize_learning_history_of_agent(self, agent, figsize=(9, 3), x_lim=None, y_lim=None, title=None, legend=True, loc='best', x_axis_log_scale=False, y_axis_log_scale=False, font_customization=False, IQL=False, savefig=False, directoryname=None, color=None, color_actions=False, dpi=150):
        """
        This function visualizes the learning history of an agent.

        Args:
            agent (Agent): _description_
            figsize (tuple, optional): _description_. Defaults to (30, 10).
            x_lim (_type_, optional): _description_. Defaults to None.
            y_lim (_type_, optional): _description_. Defaults to None.
            title (_type_, optional): _description_. Defaults to True.
            legend (_type_, optional): _description_. Defaults to True.
            loc (_type_, optional): _description_. Defaults to 'best'.
        """
        # plot the q values of an agent against the number of episodes
        if font_customization:
            SMALL_SIZE = 16
            MEDIUM_SIZE = 24
            BIGGER_SIZE = 20
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        learning_history = agent.get_learning_history()
        q_table_history = np.array(learning_history["q_table"])

        # take the mean of all q_tables
        eq_time = 1000 # equilibrium time steps after which the systems seems equilibrated and the mean of the q_tables is calculated
        q_table_mean = np.mean([q_table for q_table in agent.q_table_history[eq_time:]], axis=0)

        if q_table_history.shape == (0,): # if the q_table_history is empty, return without plotting
            return 

        num_columns = q_table_history.shape[-1]
        num_rows = q_table_history.shape[-2]

        lines_array = ["dashed", "solid", (0, (1, 10)), (0, (1, 1)), (0, (1, 1)), (5, (10, 3)), (0, (5, 10)), (0, (5, 5))]
        linewidth_array = [2., 1., 1., 1., 1., 1.]

        for column in range(num_columns): # iterate over actions
            for row in range(num_rows): # iterate over states
                q_values = q_table_history[:, row, column]
                
                actionLabel = int(agent.action_space[column])
                key = agent.translate_state_to_key(row)
                key_label = self.tranlate_key_to_state_label(key)

                if IQL:
                    if actionLabel == 0:
                        actionLabel = "D"
                    elif actionLabel == 1:
                        actionLabel = "C"
                    else:
                        "not defined action"
                    label=f'Action {actionLabel}'
                else:
                    label=f'State {key_label} - Action {actionLabel}'
                    if color_actions:
                        label=f'Action {actionLabel}'
                
                if color_actions: # paint different actions in different colors
                    if IQL is False:
                        color = plt.cm.rainbow(np.linspace(0, 1, num_columns))[column]
                    linestyle = "solid"
                    linewidth = 1.
                else: # paint different states in different colors
                    if IQL is False:
                        color = plt.cm.rainbow(np.linspace(0, 1, num_rows))[row]
                    linestyle = lines_array[int(agent.action_space[column])]
                    linewidth = linewidth_array[int(agent.action_space[column])]

                if color is None:
                    ax.plot(np.arange(len(q_table_history)), q_values, label=label, linestyle=linestyle, linewidth=linewidth)
                else:
                    ax.plot(np.arange(len(q_table_history)), q_values, label=label, linestyle=linestyle, linewidth=linewidth, color=color)

                if x_axis_log_scale:
                    ax.set_xscale('log') # plot x-axis in log scale
                if y_axis_log_scale:
                    ax.set_yscale('log') # plot y-axis in log scale

        if title == None:
            ax.set_title(rf'Q-Values of agent {agent.player_id +1}. {agent.name}, {agent.selection_method} policy, T = {agent.temperature}, $\alpha$ = {agent.learning_rate}, $\gamma =${agent.discount_factor}')
        else:
            ax.set_title(title)
        ax.set_xlabel('Time step t')
        ax.set_ylabel('Q-Values')
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if legend:
            ax.legend(loc=loc)
        ax.grid(linestyle='dotted', linewidth=0.5)
        if savefig:
            if directoryname is not None:
                plt.savefig(directoryname+f'Q-Values_agent={agent.player_id +1}_{agent.name}_y={agent.discount_factor}_T={agent.temperature}_alpha={agent.learning_rate}_time={len(q_table_history)-1}.png', dpi=100, bbox_inches='tight', facecolor='white')
            plt.savefig(f'Figures/Q-Space/Q-Values_agent={agent.player_id +1}_{agent.name}_y={agent.discount_factor}_T={agent.temperature}_alpha={agent.learning_rate}_time={len(q_table_history)-1}.png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def visualize_learning_history(self, agents, figsize = (9, 3), x_lim=None, y_lim=None, title=None, legend=True, loc='best', x_axis_log_scale=False, y_axis_log_scale=False, font_customization=False, IQL=False, savefig=False, directoryname=None, color_list=None, color_actions=False, dpi=150):
        """
        This function visualizes the learning history of multiple agents.

        Args:
            agents (Agent): 
            figsize (tuple, optional): _description_. Defaults to (30,10).
            x_lim (_type_, optional): _description_. Defaults to None.
        """
        for i, agent in enumerate(agents):
            if color_list is not None:
                color = color_list[i]
            else:
                color = None
            self.visualize_learning_history_of_agent(agent, figsize = figsize, x_lim=x_lim, y_lim=y_lim, title=title, legend=legend, loc=loc, x_axis_log_scale=x_axis_log_scale, y_axis_log_scale=y_axis_log_scale, font_customization=font_customization, IQL=IQL, savefig=savefig, directoryname=directoryname, color=color, color_actions=color_actions, dpi=dpi)

    def tranlate_key_to_state_label(self, key):
        """
        This function translates the key to a state label.

        Args:
            key (str): key of the state

        Returns:
            str: state label
        """
        state_label = ""
        for i, char in enumerate(key):
            if char == "0":
                state_label += "D"
            elif char == "1":
                state_label += "C"
            else:
                state_label += "X"
        return state_label

    def get_prob_trajectories(self, agents, temperature_func=None):
        # calculate the probability trajectories of all agents
        prob_trajectories = np.zeros((len(agents), len(agents[0].q_table_history), agents[0].q_table.shape[0], agents[0].q_table.shape[1]))
        for agent_index, agent in enumerate(agents):
            for time_step, q_table in enumerate(agent.q_table_history):
                if temperature_func is not None:
                    temperature = temperature_func(time_step)
                else:
                    temperature = agent.temperature
                exp_values = np.exp(q_table / temperature) 
                probabilities = exp_values / np.sum(exp_values, axis=1)[:, np.newaxis]
                prob_trajectories[agent_index, time_step, :, :] = probabilities
        
        return prob_trajectories

    def average_multi_run_prob_trajectories(self, game, agents, initial_q_tables, num_time_steps, num_runs, learning_rate_func = None, temperature_func=None):
        # set the initial q_tables of the agents to the given initial_q_tables
        for agent, initial_q_table in zip(agents, initial_q_tables):
            agent.q_table = initial_q_table
            agent.initial_q_table = initial_q_table
        
        # run the game num_runs times and save the probability trajectory of the agents
        if type(agents[0]) == SarsaAgent or type(agents[0]) == ExpSarsaAgent:
            multi_run_prob_trajectories = np.zeros((num_runs, len(agents), num_time_steps, agents[0].q_table.shape[0], agents[0].q_table.shape[1]))
        else:
            multi_run_prob_trajectories = np.zeros((num_runs, len(agents), num_time_steps + 1, agents[0].q_table.shape[0], agents[0].q_table.shape[1]))

        # Use joblib to parallelize the runs
        results = Parallel(n_jobs=-1)(delayed(self.run_and_calculate_trajectories)(game, agents, num_time_steps, learning_rate_func = learning_rate_func, temperature_func = temperature_func) for _ in range(num_runs))
        for i, result in enumerate(results):
            multi_run_prob_trajectories[i, :, :, :, :] = result

        # split the trajectories into two categories: One where the endpoints lie below and on y = -x + 1 and one where the endpoints lie above y = -x + 1 
        # create mask for the trajectories where the endpoints lie below and on y = -x + 1
        is_above_diagonal = np.zeros(num_runs, dtype=bool)
        for run, prob_trajectory in enumerate(multi_run_prob_trajectories):
            end_point = (prob_trajectory[0, :, :, -1][-1], prob_trajectory[1, :, :, -1][-1]) # extract the endpoint of the probability trajectory
            if end_point[0] + end_point[1] > 1:
            #if end_point[1] > 0.2:
                is_above_diagonal[run] = True
        
        # seperate the trajectories 
        prob_trajectory_below_diagonal = multi_run_prob_trajectories[~is_above_diagonal]
        prob_trajectory_above_diagonal = multi_run_prob_trajectories[is_above_diagonal]
        # calculate the rate of occurence of the trajectories below and above the diagonal
        rate_below_diagonal = prob_trajectory_below_diagonal.shape[0] / num_runs
        rate_above_diagonal = prob_trajectory_above_diagonal.shape[0] / num_runs
        rate_of_occurence = np.array([rate_below_diagonal, rate_above_diagonal])

        # get the index of cooperation in the action space
        cooperation_index = np.where(agents[0].action_space == 0)[0][0]

        # Check if prob_trajectory_below_diagonal is empty before calculating the mean
        if prob_trajectory_below_diagonal.size > 0:
            # calculate the mean of the probability trajectories below the diagonal
            mean_prob_trajectory_below_diagonal = np.mean(prob_trajectory_below_diagonal, axis=0)
            # extract the probabilities to fully cooperate of the agents as a trajectory tuple: 
            mean_prob_trajectory_below_diagonal = (mean_prob_trajectory_below_diagonal[0, :, :, cooperation_index], mean_prob_trajectory_below_diagonal[1, :, :, cooperation_index])
        else:
            mean_prob_trajectory_below_diagonal = None

        # Check if prob_trajectory_above_diagonal is empty before calculating the mean
        if prob_trajectory_above_diagonal.size > 0:
            mean_prob_trajectory_above_diagonal = np.mean(prob_trajectory_above_diagonal, axis=0)
            # extract the probabilities to fully cooperate of the agents as a trajectory tuple: 
            mean_prob_trajectory_above_diagonal = (mean_prob_trajectory_above_diagonal[0, :, :, cooperation_index], mean_prob_trajectory_above_diagonal[1, :, :, cooperation_index])
        else:
            mean_prob_trajectory_above_diagonal = None
        
        return mean_prob_trajectory_below_diagonal, mean_prob_trajectory_above_diagonal, rate_of_occurence

    def run_and_calculate_trajectories(self, game, agents, num_time_steps, learning_rate_func = None, temperature_func = None):
        self.run(game, agents, num_time_steps, learning_rate_func = learning_rate_func, temperature_func = temperature_func)
        prob_trajectories = self.get_prob_trajectories(agents)
        return prob_trajectories

    def calculate_q_value(self, probability, temperature):
            if probability == 0 :
                q = - temperature * np.log( 1 / 10e-20 - 1)
            else:
                q = - temperature * np.log( 1 / probability - 1)
            return q
    
    def generate_q_values(self, prob_to_coop, temperature, base_value):
    # Calculate the difference between Q-values
        delta_Q = temperature * np.log(1/prob_to_coop - 1) # difference between Q-values: delta_Q = Q_D - Q_C
        
        # Calculate Q_D and Q_C centered around the base value
        Q_D = base_value + delta_Q / 2
        Q_C = base_value - delta_Q / 2
        
        return np.array([[Q_C, Q_D]])

    def calculate_q_tables(self, initial_probabilities, agents, base_value=0):
        # calculate the initial q_tables of the agents from the initial_probabilities, which is list of tuples of probabilities of the agents to cooperate
        initial_q_tables_array = []
        for initial_probabilities_tuple in initial_probabilities:
            initial_q_tables = []
            for player_id, initial_probability_agent in enumerate(initial_probabilities_tuple):
                #initial_q_table = np.array([[self.calculate_q_value(initial_probability_agent, agents[player_id].temperature), 0]])
                initial_q_table = self.generate_q_values(initial_probability_agent, agents[player_id].temperature, base_value)
                initial_q_tables.append(initial_q_table)
            initial_q_tables_array.append(initial_q_tables)
        return initial_q_tables_array

    def calculate_action_probabilities(self, initial_q_tables, agents):
        action_probabilities_array = []
        for initial_q_tables_tuple in initial_q_tables:
            action_probabilites_tuple = []
            for player_id, initial_q_table in enumerate(initial_q_tables_tuple):
                action_probabilites_tuple.append( agents[player_id].get_action_probabilities(initial_q_table) )
            action_probabilities_array.append(action_probabilites_tuple)
        return action_probabilities_array

    def flow_plot(self, game, agents, initial_probabilities, num_time_steps, num_runs, learning_rate_func = None, temperature_func=None, figsize=(10,10), dpi=150, equilibration_time=None, 
                  plot_end_marker=False, plot_mean_marker=False, marker_1 = "x", marker_2 = "p", marker_size_1=50, marker_size_2=100, start_marker_color="#6a0dad", end_marker_color="red", eq_marker_color="gray", mean_marker_color="black",
                  title=None, 
                  initial_arrow_probabilities=None,
                  cmap = 'viridis',
                  plot_time_evolution=False,
                  plot_analytical_solution=False,
                  save_path=None,
                  FAQ=False,
                  base_value=0,
                  fontsize=14,
                  plot_legend_underneath=False,
                  plot_colorbar=True,
                  plot_detailed_legend=True):
        # calculate the initial q_tables of the agents from the initial_probabilities, which is list of tuples of probabilities of the agents to cooperate
        initial_q_tables_array = self.calculate_q_tables(initial_probabilities, agents, base_value=base_value)

        # Create figure and axes outside of the loop
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        colors_prob = plt.cm.plasma(np.linspace(0, 1, len(initial_probabilities))) # Creating a color map from blue to red for the different initial probabilities
        # Create a Patch object for the custom legend entry
        custom_patches = []

        if plot_legend_underneath:
            if plot_detailed_legend:
                custom_patches.append( mpatches.Patch(color='none', label=f'T = {agents[0].temperature}, $\\alpha =${agents[0].learning_rate}, $\gamma =${agents[0].discount_factor}, $Q_{{base}}$ = {base_value}') )
            else:
                custom_patches.append( mpatches.Patch(color='none', label=f'$\gamma =${agents[0].discount_factor}, $Q_{{base}}$ = {base_value}') )
            if initial_arrow_probabilities is not None:
                if plot_detailed_legend:
                    custom_patches.append( mpatches.Patch(color='none', label=f'Arrows show Deterministic Model (2019)') )
        else:
            for agent in agents:
                if FAQ:
                    custom_patches.append( mpatches.Patch(color='none', label=f'Agent {agent.player_id +1}: \n{agent.name}, \n$\gamma =${agent.discount_factor}, \n$\\alpha =${agent.learning_rate}, \n$\\beta =${agent.learning_rate_adjustment}, \n$T =${agent.temperature} \n') )
                else:
                    custom_patches.append( mpatches.Patch(color='none', label=f'Agent {agent.player_id +1}: \n{agent.name}, \n$\gamma =${agent.discount_factor}, \n$\\alpha =${agent.learning_rate}, \n$T =${agent.temperature} \n') )

        
        # run simulation of the game for num_episodes
        endpoints = []
        mean_eq_points = []
        change_label = False
        if equilibration_time is None or equilibration_time >= num_time_steps:
            change_label = True
            equilibration_time = num_time_steps 

        if plot_detailed_legend:
            labels_added = False
        else:
            labels_added = True
        for i, initial_q_tables in enumerate(initial_q_tables_array):
            mean_prob_trajectory_below_diagonal, mean_prob_trajectory_above_diagonal, rate_of_occurence = self.average_multi_run_prob_trajectories(game, agents, initial_q_tables = initial_q_tables, num_time_steps = num_time_steps, num_runs = num_runs, learning_rate_func = learning_rate_func, temperature_func = temperature_func)
            size_array = 2 * rate_of_occurence

            for j, trajectory in enumerate([mean_prob_trajectory_below_diagonal, mean_prob_trajectory_above_diagonal]):
                if trajectory is None:
                    continue
                
                if plot_time_evolution:
                    # Create a list of lines
                    lines = [((trajectory[0][i][0], trajectory[1][i][0]), (trajectory[0][i+1][0], trajectory[1][i+1][0])) for i in range(num_time_steps-1)]
                    # Create a LineCollection
                    lc = LineCollection(lines, cmap='plasma', linewidths=size_array[j], alpha=1)
                    # Set the colors of the lines
                    lc.set_array(np.linspace(0, 1, num_time_steps))
                    # Add the LineCollection to the plot
                    plt.gca().add_collection(lc)
                else:
                    # plot the probability trajectories of the agents
                    plt.plot(trajectory[0], trajectory[1], color = colors_prob[i], linewidth=size_array[j], alpha=1)

                # Calculate the mean point after equilibration
                mean_eq_point = (np.mean(trajectory[0][equilibration_time:]), np.mean(trajectory[1][equilibration_time:]))
                mean_eq_points.append(mean_eq_point)

                # get the endpoint 
                endpoints.append((trajectory[0][-1], trajectory[1][-1]))

                start_label = "Start" if not labels_added else ""
                end_label = f"End" if not labels_added else ""
                start_scatter = plt.scatter(trajectory[0][0], trajectory[1][0], color=start_marker_color, label=start_label, marker=marker_1, s=marker_size_1, linewidths=2, zorder=8)
                if plot_end_marker:
                    end_scatter = plt.scatter(trajectory[0][-1], trajectory[1][-1], color=end_marker_color, label=end_label, marker=marker_2, s=marker_size_1, linewidths=2, zorder=9)
                if plot_mean_marker:
                    if change_label:
                        mean_eq_point_scatter = plt.scatter(mean_eq_point[0], mean_eq_point[1], color=eq_marker_color, marker=marker_1, s=marker_size_1, linewidth=2, zorder=10, label=f'End')
                    else:
                        mean_eq_point_scatter = plt.scatter(mean_eq_point[0], mean_eq_point[1], color=eq_marker_color, marker=marker_1, s=marker_size_1, linewidth=2, zorder=10, label=f'Mean endpoint over final {num_time_steps-equilibration_time} steps')
                if not labels_added:
                    custom_patches.append(start_scatter)
                    if plot_end_marker:
                        custom_patches.append(end_scatter)
                    if plot_mean_marker:
                        custom_patches.append(mean_eq_point_scatter)
                labels_added = True

        mean_endpoint = np.mean(endpoints, axis=0)
        # calculate the mean of the equilibtrated endpoints and the standard deviation
        mean_eq_point = np.mean(mean_eq_points, axis=0)
        std_eq_point = np.std(mean_eq_points, axis=0)

        #mean_endpoint_scatter = plt.scatter(mean_endpoint[0], mean_endpoint[1], c=mean_marker_color, label=f"\nMean endpoint:\n{np.round(mean_endpoint[0], 3)[0], np.round(mean_endpoint[1], 3)[0]} \n", marker=marker_2, s=marker_size_2, linewidths=5, zorder=12)
        mean_eq_point_scatter = plt.scatter(mean_eq_point[0], mean_eq_point[1], c=mean_marker_color, label=f"({np.round(mean_eq_point[0], 3)}, {np.round(mean_eq_point[1], 3)}) $\pm$ ({np.round(std_eq_point[0], 3)}, {np.round(std_eq_point[1], 3)})", marker=marker_2, s=marker_size_2, linewidths=5, zorder=12)
        #custom_patches.append(mean_endpoint_scatter)
        custom_patches.append(mean_eq_point_scatter)

        if plot_analytical_solution:  
            # Analyytical solution: 2D-equation system in policy space
            T1 = agents[0].temperature
            T2 = agents[1].temperature
            def equations(variables):
                x, y = variables
                eq1 = x - np.exp((y+5*(1-y))/T1) / ( np.exp((y+5*(1-y))/T1) + np.exp((3*(1-x))/T1) )
                eq2 = y - np.exp((x+5*(1-x))/T2) / ( np.exp((x+5*(1-x))/T2) + np.exp((3*(1-y))/T2) )
                return [eq1, eq2]
            initial_guess = [0, 0]
            # Solve the system numerically and get information
            prob_D_1_solution, prob_D_2_solution = fsolve(equations, initial_guess)
            prob_C_1_solution = 1 - prob_D_1_solution
            prob_C_2_solution = 1 - prob_D_2_solution
            analytical_solution_scatter = plt.scatter(prob_C_1_solution, prob_C_2_solution, c='red', label=f"Fixed Point Det. Model (2019): {np.round(prob_C_1_solution, 3), np.round(prob_C_2_solution, 3)}", marker='x', s=marker_size_2, linewidths=5, zorder=11)
            if plot_detailed_legend:
                custom_patches.append(analytical_solution_scatter)
            #custom_patches.append(analytical_solution_scatter)

        if initial_arrow_probabilities is not None:
            next_arrow_probabilities = self.calculate_next_probabilities(agents, initial_arrow_probabilities)
            lengths = [np.sqrt((next_arrow_probabilities[i][0] - initial_arrow_probability[0])**2 + 
                            (next_arrow_probabilities[i][1] - initial_arrow_probability[1])**2) 
                    for i, initial_arrow_probability in enumerate(initial_arrow_probabilities)]
            # Normalize the lengths to the range [0, 1]
            lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
            # Create a colormap
            if cmap is not None:
                cmap = plt.get_cmap(cmap)
            for i, initial_arrow_probability in enumerate(initial_arrow_probabilities):
                dx = next_arrow_probabilities[i][0] - initial_arrow_probability[0]
                dy = next_arrow_probabilities[i][1] - initial_arrow_probability[1]
                direction_length = np.sqrt(dx**2 + dy**2)
                dx /= direction_length
                dy /= direction_length
                dx *= lengths[i] * 0.05
                dy *= lengths[i] * 0.05
                if cmap is not None:
                    color = cmap(lengths[i])
                    plt.arrow(initial_arrow_probability[0], initial_arrow_probability[1], dx, dy, 
                        head_width=0.01, head_length=0.01, fc=color, ec=color)
                else:
                    plt.arrow(initial_arrow_probability[0], initial_arrow_probability[1], dx, dy, 
                        head_width=0.01, head_length=0.01, fc='black', ec='black')

        # plot colorbar for time evolution
        if plot_colorbar:
            # Add a colorbar for the time evolution
            cbar = plt.colorbar(lc, fraction=0.046, pad=0.04)
            cbar.set_label('time steps')
            # lc is normalized so we need to set the ticks manually
            tick_values = [0, 1]
            tick_labels = [0, num_time_steps]
            cbar.set_ticks(tick_values)  # Set the ticks to the desired positions
            cbar.set_ticklabels(["{:.0e}".format(tick_label) for tick_label in tick_labels]) # Format tick labels into scientific notation manually

        if plot_legend_underneath:
            plt.legend(handles=custom_patches, bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=fontsize)
        else:
            plt.legend(handles=custom_patches, bbox_to_anchor=(1.01, 1.01), loc='upper left', fontsize=fontsize)
        if title is None:
            if num_runs > 1:
                plt.title(f"Mean Q-learning trajectories\nMean taken over {num_runs} runs", fontsize = fontsize + 2)
            else:
                plt.title(f"Q-learning trajectories\n", fontsize = fontsize + 2)
        else:
            plt.title(title, fontsize = 16)
        plt.xlabel(f"Policy of Agent 1, $\pi^1_C$", fontsize=fontsize)
        plt.ylabel(f"Policy of Agent 2, $\pi^2_C$", fontsize=fontsize)
        # set ticks of label to fontsize
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim(-0.01,1.01)
        plt.ylim(-0.01,1.01)
        if initial_arrow_probabilities is None:
            plt.grid()
        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def arrowplot(self, agents, initial_arrow_probabilities, save_path=None, cmap='binary', figsize=(8,8), dpi=150, axislabel='C', fontsize=14, marker_size=100, title=None, loc='upper center'):
        # Set the figure size
        fig = plt.figure(figsize=figsize, dpi=dpi)
        next_arrow_probabilities = self.calculate_next_probabilities(agents, initial_arrow_probabilities)
        lengths = [np.sqrt((next_arrow_probabilities[i][0] - initial_arrow_probability[0])**2 + 
                        (next_arrow_probabilities[i][1] - initial_arrow_probability[1])**2) 
                for i, initial_arrow_probability in enumerate(initial_arrow_probabilities)]
        # Normalize the lengths to the range [0, 1]
        lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))
        for i, initial_arrow_probability in enumerate(initial_arrow_probabilities):
            dx = next_arrow_probabilities[i][0] - initial_arrow_probability[0]
            dy = next_arrow_probabilities[i][1] - initial_arrow_probability[1]
            direction_length = np.sqrt(dx**2 + dy**2)
            dx /= direction_length
            dy /= direction_length
            dx *= lengths[i] * 0.05
            dy *= lengths[i] * 0.05
            if cmap is None:
                # plot all arrows in black
                plt.arrow(initial_arrow_probability[0], initial_arrow_probability[1], dx, dy, 
                    head_width=0.01, head_length=0.01, fc='black', ec='black')
            else:
                # Create a colormap
                cmap = plt.get_cmap(cmap)
                color = cmap(lengths[i])
                plt.arrow(initial_arrow_probability[0], initial_arrow_probability[1], dx, dy, 
                    head_width=0.01, head_length=0.01, fc=color, ec=color)
            
        if True:  
            # Analyytical solution: 2D-equation system in policy space (works only for symmetric games)
            T1 = agents[0].temperature
            T2 = agents[1].temperature
            def equations(variables):
                x, y = variables # probabilities to choose the second action (for PD. probabilities to cooperate)
                R1_00 = agents[0].calculate_reward([0, 0]) # reward for agent 1 if both agents choose action 0 (both cooperate)
                R1_01 = agents[0].calculate_reward([0, 1]) # reward for agent 1 if agent 0 chooses action 0 and agent 2 chooses action 1 
                R1_10 = agents[0].calculate_reward([1, 0]) # reward for agent 1 if agent 0 chooses action 1 and agent 2 chooses action 0
                R1_11 = agents[0].calculate_reward([1, 1]) # reward for agent 1 if both agents choose action 1

                R2_00 = agents[1].calculate_reward([0, 0]) # reward for agent 2 if both agents choose action 0
                R2_01 = agents[1].calculate_reward([0, 1]) # reward for agent 2 if agent 0 chooses action 0 and agent 1 chooses action 1
                R2_10 = agents[1].calculate_reward([1, 0]) # reward for agent 2 if agent 0 chooses action 1 and agent 1 chooses action 0
                R2_11 = agents[1].calculate_reward([1, 1]) # reward for agent 2 if both agents choose action 1

                # x: probability of agent 0 to choose action 0
                # y: probability of agent 1 to choose action 0
                eq1 = x - np.exp((y*R1_00 + (1-y)*R1_01)/T1) / ( np.exp((y*R1_00 + (1-y)*R1_01)/T1) + np.exp((y*R1_10 + (1-y)*R1_11)/T1) )
                eq2 = y - np.exp((x*R2_00 + (1-x)*R2_10)/T2) / ( np.exp((x*R2_00 + (1-x)*R2_10)/T2) + np.exp((x*R2_01 + (1-x)*R2_11)/T2) )
                return [eq1, eq2]
            initial_guesses = [[0, 0], [1/3, 1/3], [0.5, 0.5], [2/3, 2/3], [1, 1]]
            eq_solutions = []
            for initial_guess in initial_guesses:
                # Solve the system numerically and get information
                prob_C_1_solution, prob_C_2_solution = np.round(fsolve(equations, initial_guess), 3)
                eq_solutions.append((prob_C_1_solution, prob_C_2_solution))
                print(f"Initial guess: {initial_guess}, Solution: {prob_C_1_solution, prob_C_2_solution}")
            # print only the unique solutions
            eq_solutions = list(set(eq_solutions))
            for eq_solution in eq_solutions:
                analytical_solution_scatter = plt.scatter(eq_solution[0], eq_solution[1], c='red', label=f"$(\pi^1_{{{axislabel}*}}, \pi^2_{{{axislabel}*}}) \\approx${np.round(eq_solution[0], 3), np.round(eq_solution[1], 3)}", marker='x', s=marker_size, linewidths=5, zorder=11)
            #analytical_solution_scatter = plt.scatter(prob_C_1_solution, prob_C_2_solution, c='red', label=f"$(\pi^1_{{{axislabel}*}}, \pi^2_{{{axislabel}*}}) \\approx${np.round(prob_C_1_solution, 3), np.round(prob_C_2_solution, 3)}", marker='x', s=100, linewidths=5, zorder=11)
        
        if fontsize is None:
            fontsize_title = None
        else:
            fontsize_title = fontsize + 2
        if title is None:
            #plt.title(f"Deterministic Dynamics (2019)\nin Policy Space. T = {agents[0].temperature}", fontsize=fontsize_title)
            pass
        else:
            plt.title(title, fontsize=fontsize_title)
        #plt.xlabel(f"Agent 1: prob. of {axislabel}, $\pi^1_{axislabel}$", fontsize=fontsize)
        plt.xlabel(f"Policy of Agent 1, $\pi^1_{axislabel}$", fontsize=fontsize)
        #plt.ylabel(f"Agent 2: prob. of {axislabel}, $\pi^2_{axislabel}$", fontsize=fontsize)
        plt.ylabel(f"Policy of Agent 2, $\pi^2_{axislabel}$", fontsize=fontsize)
        plt.xlim(-0.01,1.01)
        plt.ylim(-0.01,1.01)
        # ensure that the aspect ratio is equal 
        plt.gca().set_aspect('equal', adjustable='box')
        # ensure that the x-axis is the same length as the y-axis
        # ensure that ticks are set for both axis equally
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.xticks(ticks, fontsize=fontsize)
        plt.yticks(ticks, fontsize=fontsize)
        plt.legend(loc=loc, fontsize=fontsize)
        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.show()

    def plot_strategy_flow(self, game, agents, initial_prob_tuple_array, figsize=(10,10)):
        # calculate the initial q_tables of the agents from the initial_probabilities, which is list of tuples of probabilities of the agents to cooperate
        initial_q_tables_array = self.calculate_q_tables(initial_prob_tuple_array, agents)
        new_prob_tuple_array = []
        for initial_q_table in initial_q_tables_array:
            for agent in agents:
                agent.q_table = initial_q_table[agent.player_id]
                agent.initial_q_table = initial_q_table[agent.player_id]
            
            new_probabilities = [None, None]
            for i, agent in enumerate(agents):
                other_agent = agents[1 - agent.player_id]
                exp_reward_of_action = game.get_exp_reward(agents, agent.player_id)
                exp_V = agent.get_action_probabilities(agent.q_table) * agent.q_table
                quality_of_next_state = np.sum(other_agent.get_action_probabilities(other_agent.q_table) * exp_V)

                cooperation_index = np.where(agent.action_space == 1)[0][0]
                TDe = exp_reward_of_action[0][cooperation_index] + agent.discount_factor * quality_of_next_state - agent.q_table[0, cooperation_index]
                Q_new = agent.q_table[0, cooperation_index] + agent.learning_rate * TDe
                agent.q_table[0, cooperation_index] = Q_new
                probability_to_cooperate_new = agent.get_action_probabilities(agent.q_table)[0][cooperation_index]
                new_probabilities[i] = probability_to_cooperate_new
            
            new_prob_tuple = tuple(new_probabilities)
            new_prob_tuple_array.append(new_prob_tuple)
        
        return new_prob_tuple_array

        
        print(f"dP: \n{dP}")
        return dP

    def calculate_next_probabilities(self, agents, initial_probabilities):
        reward_matrix_agent_0 = np.array([[agents[0].reward_func([i, j], agents[0].player_id) for j in range(2)] for i in range(2)])
        reward_matrix_agent_1 = np.array([[agents[1].reward_func([i, j], agents[1].player_id) for j in range(2)] for i in range(2)])

        # get learning rates and temperature of the agents
        learning_rate_agent_0 = agents[0].learning_rate
        learning_rate_agent_1 = agents[1].learning_rate
        temperature_agent_0 = agents[0].temperature
        temperature_agent_1 = agents[1].temperature

        next_probabilities_array = []
        for probabilities in initial_probabilities:
            prop_agent_0, prop_agent_1 = probabilities # prob to cooperate
            # construct probability vectors for both agents
            p_0_vector = np.array([prop_agent_0, 1 - prop_agent_0]) # prob to coop, prob to defect
            p_1_vector = np.array([prop_agent_1, 1 - prop_agent_1]) # prob to coop, prob to defect

            # intermediate calculations
            P_0_vector = p_0_vector * np.exp( learning_rate_agent_0 / temperature_agent_0 * (np.dot(reward_matrix_agent_0, p_1_vector) - temperature_agent_0 * np.log(p_0_vector)))
            P_1_vector = p_1_vector * np.exp( learning_rate_agent_1 / temperature_agent_1 * (np.dot(reward_matrix_agent_1.T, p_0_vector) - temperature_agent_1 * np.log(p_1_vector)))

            # calculate the next probabilities
            prop_agent_0_next = P_0_vector[0] / np.sum(P_0_vector) # expected prob to cooperate for agent 1 in next time step
            prop_agent_1_next = P_1_vector[0] / np.sum(P_1_vector) # expected prob to cooperate for agent 2 in next time step

            next_probabilities_array.append((prop_agent_0_next, prop_agent_1_next))
        
        return next_probabilities_array
    