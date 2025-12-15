## ok in this file we will implement multi-arm bandits....



import numpy as np

class MultiArmEnv:
    def __init__(self, true_mean_of_rewards):
        self.true_mean_of_rewards = np.array(true_mean_of_rewards)   ##  this will be the alread known true value or rewards
        self.reward_std = 1.0   ## scaling factor for sampling a reward from normal distribution,,,  mean as value of true reward vlaue (already given), std as 1.0

    def reward(self, action):
        return np.random.normal(
            loc=self.true_mean_of_rewards[action],
            scale=self.reward_std
        )   ## this function will sample from true value or reward ans std=1, and that reward will be later improved


class MultiArmAgent:
    def __init__(self, number_of_actions, epsilon=0.1):
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon

        # Average reward estimates
        self.rewards = np.zeros(number_of_actions)

        # Count of how many times each action is tried
        self.actions = np.zeros(number_of_actions)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.number_of_actions)
        else:
            return np.argmax(self.rewards)

    def update_reward_estimate(self, reward, action):
        # increment count FIRST
        self.actions[action] += 1

        alpha = 1.0 / self.actions[action]

        self.rewards[action] += alpha * (reward - self.rewards[action])






def run_experiment(iterations, true_mean_of_rewards):
    env = MultiArmEnv(true_mean_of_rewards)
    agent = MultiArmAgent(len(true_mean_of_rewards))

    rewards = []
    actions = []

    for _ in range(iterations):
        action = agent.choose_action()
        reward = env.reward(action)
        agent.update_reward_estimate(reward, action)

        rewards.append(reward)
        actions.append(action)

    return rewards, actions, agent





rewards, actions, agent = run_experiment(
    iterations=1000,
    true_mean_of_rewards=[1.2, 2.0, 1.7, 1.5]
)

print("Estimated means:", agent.rewards)
print("True means:", [1.2, 2.0, 1.7, 1.5])
