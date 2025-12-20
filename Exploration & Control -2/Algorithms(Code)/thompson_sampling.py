import numpy as np


# -------- Environment --------
# This simulates the world (slot machines)
class ThompsonEnv:
    def __init__(self, true_probabilities):
        # True (hidden) success probability of each arm
        self.true_probabilities = np.array(true_probabilities)

    def give_rewards(self, action):
        """
        Simulate pulling an arm.
        Returns True (success) with probability true_probabilities[action],
        otherwise False (failure).
        """
        return np.random.rand() < self.true_probabilities[action]


# -------- Agent --------
# This is Thompson Sampling
class ThompsonAgent:
    def __init__(self, n_actions):
        # Alpha = successes + 1 (prior)
        # Beta  = failures  + 1 (prior)
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)

    def select_action(self):
        """
        1. Sample one value from each arm's belief (Beta distribution)
        2. Pick the arm with the highest sampled value
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, action, reward):
        """
        Update belief of the selected arm only.
        reward = 1 -> success
        reward = 0 -> failure
        """
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1


# -------- Experiment loop --------
def run_experiment(true_probs, n_times):
    env = ThompsonEnv(true_probs)
    agent = ThompsonAgent(len(true_probs))

    actions = []
    rewards = []

    for _ in range(n_times):
        # Agent chooses an action based on current beliefs
        action = agent.select_action()

        # Environment returns reward (0 or 1)
        reward = int(env.give_rewards(action))

        # Agent updates belief using observed reward
        agent.update(action, reward)

        actions.append(action)
        rewards.append(reward)

    # Return history and final beliefs
    return actions, rewards, agent.alpha, agent.beta
