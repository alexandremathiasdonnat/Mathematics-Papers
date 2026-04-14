#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
November 2025
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse

from agent import BaseAgent


class PolicyEvaluation:
    """Evaluation of a policy by dynamic programming.
    
    Parameters
    ----------
    environment : object of class Environment
        Environment of the agent.
    policy : function
        Policy of the agent.
    gamma : float
        Discount factor (between 0 and 1).
    n_eval : int
        Number of iterations of Bellman's equation for policy evaluation.
    """
    def __init__(self, environment, policy='random', gamma=1, n_eval=100):
        self.environment = environment
        agent = BaseAgent(environment, policy)
        self.policy = agent.policy
        self.player = agent.player
        self.gamma = gamma
        self.n_eval = n_eval
        self.index_states()
        if self.n_states == 0:
            raise ValueError("Not applicable. The state space is too large.")
        self.get_rewards()
        self.get_transitions()
        self.values = None
        
    def index_states(self):
        """Index all states."""
        self.states = self.environment.get_all_states()
        self.n_states = len(self.states)
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}
        
    def get_state_id(self, state):
        """Get state index."""
        return self.state_id[self.environment.encode(state)]

    def get_rewards(self):
        """Get the reward of each state."""
        rewards = np.zeros(self.n_states)
        for i, state in enumerate(self.states):    
            rewards[i] = self.environment.get_reward(state)
        self.rewards = rewards
        
    def get_actions(self, state, player=None):
        """Get the available actions of the agent."""
        if self.environment.is_game():
            if player is None:
                player = self.player
            actions = self.environment.get_actions(state, player)
        else:
            actions = self.environment.get_actions(state)
        return actions
    
    def get_transitions(self):
        """Get the transitions (probabilities to move from one state to another) for each action."""
        actions = self.environment.get_all_actions()
        transitions = {action: sparse.lil_matrix((self.n_states, self.n_states)) for action in actions}
        for i, state in enumerate(self.states):    
            actions = self.get_actions(state)
            for action in actions:
                probs, next_states = self.environment.get_transition(state, action)
                indices = np.array([self.get_state_id(next_state) for next_state in next_states])
                transitions[action][i, indices] = np.array(probs)
        self.transitions = {action: sparse.csr_matrix(transition) for action, transition in transitions.items()}
            
    def evaluate_policy(self):
        """Evaluate a policy by iteration of Bellman's equation."""
        transitions = self.transitions
        # probability of each action over the states
        probs = {action: np.zeros(self.n_states) for action in transitions}
        for state in self.states:    
            i = self.get_state_id(state)
            for prob, action in zip(*self.policy(state)):
                probs[action][i] = prob
        # Bellman's equation
        values = np.zeros(self.n_states)
        for t in range(self.n_eval):
            next_values = self.rewards + self.gamma * values
            values = np.zeros(self.n_states)
            for action, transition in transitions.items():
                values += probs[action] * transition.dot(next_values)
        self.values = values
            
    def get_best_actions(self, state, player=None):
        """Get the best actions in some state according to the value function.""" 
        if player is None:
            player = self.player 
        actions = self.get_actions(state, player)
        if len(actions) > 1:
            i = self.get_state_id(state)
            transitions = self.transitions
            next_values = self.rewards + self.gamma * self.values
            values = [transitions[action].dot(next_values)[i] for action in actions]
            if player == 1:
                best_value = max(values)
            else:
                best_value = min(values)
            actions = [action for action, value in zip(actions, values) if value==best_value]
        return actions        
    
    def improve_policy(self, player=None, update=False):
        """Improve the policy according to the value function."""
        def policy(state):
            actions = self.get_best_actions(state, player)
            if len(actions):
                probs = np.ones(len(actions)) / len(actions)
            else:
                probs = []
            return probs, actions
        if update:
            self.policy = policy
        return policy

    
class PolicyIteration(PolicyEvaluation):
    """Policy iteration.
    
    Parameters
    ----------
    environment : object of class Environment
        Environment of the agent.
    gamma : float
        Discount factor (between 0 and 1).
    n_eval : int
        Number of iterations of Bellman's equation for policy evaluation.
    n_iter : int
        Maximum number of policy iterations.
    """
    def __init__(self, environment, gamma=1, n_eval=100, n_iter=10):
        super(PolicyIteration, self).__init__(environment, gamma=gamma, n_eval=n_eval)  
        self.n_iter = n_iter
    
    def get_optimal_policy(self):
        """Iterate evaluation and improvement, stop if no change."""
        for t in range(self.n_iter):
            self.evaluate_policy() 
            self.improve_policy(update=True)
        return self.policy
    

class ValueIteration(PolicyEvaluation):
    """Value iteration. 
    
    Parameters
    ----------
    environment : object of class Environment
        Environment of the agent.      
    gamma : float
        Discount factor (between 0 and 1).
    n_iter : int
        Maximum number of value iterations.
    """
    
    def __init__(self, environment, gamma=1, n_iter=100):
        super(ValueIteration, self).__init__(environment, gamma=gamma)  
        self.n_iter = n_iter
        
    def get_optimal_policy(self):
        """Get the optimal policy by iteration of Bellman's optimality equation."""
        transitions = self.transitions
        # Bellman's optimality equation
        values = np.zeros(self.n_states)
        for t in range(self.n_iter):
            next_values = self.rewards + self.gamma * values
            action_value = {action: transition.dot(next_values) for action, transition in self.transitions.items()}
            values = np.zeros(self.n_states)
            for i, state in enumerate(self.states):
                if not self.environment.is_terminal(state):
                    actions = self.get_actions(state)
                    if self.player == 1:
                        values[i] = max([action_value[action][i] for action in actions])
                    else:
                        values[i] = min([action_value[action][i] for action in actions])
        self.values = values
        policy = self.improve_policy()
        return policy
    
    def get_perfect_players(self):
        """Get perfect players for games, with the best response of the adversary."""
        if not self.environment.is_game():
            raise ValueError("This method applies to games only.")
        # get transitions for each player
        actions = self.environment.get_all_actions()
        transitions = {action: sparse.lil_matrix((self.n_states, self.n_states)) for action in actions}
        for i, state in enumerate(self.states):    
            actions = self.environment.get_available_actions(state)
            for action in actions:
                next_state = self.environment.get_next_state(state, action)
                j = self.get_state_id(next_state)
                transitions[action][i, j] = 1
        transitions = {action: sparse.csr_matrix(transition) for action, transition in transitions.items()}
        self.transitions = transitions
        # Bellman's optimality equation
        values = np.zeros(self.n_states)
        for t in range(self.n_iter):
            next_values = self.rewards + self.gamma * values
            action_value = {action: transition.dot(next_values) for action, transition in transitions.items()}
            values = np.zeros(self.n_states)
            for i, state in enumerate(self.states):
                if not self.environment.is_terminal(state):
                    player, _ = state
                    actions = self.environment.get_available_actions(state)
                    if player == 1:
                        values[i] = max([action_value[action][i] for action in actions])
                    else:
                        values[i] = min([action_value[action][i] for action in actions])
        self.values = values
        # policies
        policy = self.improve_policy(self.player)
        adversary_policy = self.improve_policy(-self.player)
        return policy, adversary_policy
        