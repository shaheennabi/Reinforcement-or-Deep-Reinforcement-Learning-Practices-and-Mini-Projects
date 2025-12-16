## Gradient Bandits with Baseline Implementation

import numpy as np


class BanditEnv():
    def __init__(self, true_rewards):
        self.true_rewards = np.array(true_rewards)
        self.std = 1.0
        

    def give_reward(self, action):
        """Returns noisy reward for the chosen action"""
        reward = np.random.normal(self.true_rewards[action], self.std)
        return reward
    

class BanditAgent():
    def __init__(self, number_of_actions):
        self.preferences = np.zeros(number_of_actions)
        self.k_actions = number_of_actions
        self.average_reward = 0
        self.n = 0
        self.alpha = 0.01


    def softmax(self, preferences):
        """Convert preferences to probabilities"""
        exp_preferences = np.exp(preferences - np.max(preferences))  
        probabilities = exp_preferences / np.sum(exp_preferences)
        return probabilities
    

    def select_action(self):
        """Select action based on softmax probabilities"""
        probabilities = self.softmax(self.preferences)
        action = np.random.choice(self.k_actions, p=probabilities)
        return action
    

    def update(self, action, reward):
        """Update preferences based on received reward"""
        self.n += 1
        
        # Baseline update (incremental average)
        self.average_reward += (reward - self.average_reward) / self.n

        # Error (how much better/worse than baseline)
        error = reward - self.average_reward

        # Get current probabilities
        probability = self.softmax(self.preferences)

        # Update preferences for all actions
        for a in range(self.k_actions):
            if a == action:
                # BUG FIX: Use self.alpha, not alpha
                self.preferences[a] += self.alpha * error * (1 - probability[a])
            else: 
                # BUG FIX: Use self.alpha, not alpha
                self.preferences[a] -= self.alpha * error * probability[a]



def run_experiment(n_times):
    """Run the gradient bandit experiment"""
    true_rewards = [1.25, 2.5, 2.0, 1.75]
    env = BanditEnv(true_rewards)
    # BUG FIX: Pass number of actions (length), not the list itself
    agent = BanditAgent(len(true_rewards))

    total_reward = 0
    
    for i in range(n_times):
        action = agent.select_action()
        reward = env.give_reward(action)
        agent.update(action, reward)
        total_reward += reward

    # Return useful information
    return agent.preferences, agent.softmax(agent.preferences), total_reward / n_times


