## in this file we will implement the, ucb algorithm in multi-arm bandits 


import numpy as np

class UCBEnv():
    def __init__(self, true_rewards):
        self.true_rewards = np.array(true_rewards)
        self.std = 1.0  ## this is the standard deviation


    def give_rewards(self, action):
        reward = np.random.normal(self.true_rewards[action], self.std)

        return reward 
    

class UCBAgent():
    def __init__(self, all_actions, c = 2):
        self.all_actions = all_actions
        self.average_reward_of_every_action = np.zeros(all_actions)
        self.count_of_every_action_selected = np.zeros(all_actions)
        self.c = c   ## exploration hyper-parameters
        self.t = 0   ## overall count of action-selection





    def select_actions(self):

        ## increament the count of actions, taken so far
        self.t += 1
        
        ## check first, if every action is atleast tried once
        for a in range(self.all_actions):
            if self.count_of_every_action_selected[a] == 0:
                return a 
            
        ## when all the action are atleast once tried
        ucb_score = self.average_reward_of_every_action + self.c * np.sqrt(np.log(self.t) / self.count_of_every_action_selected)

        return np.argmax(ucb_score)
        


    def update(self, action, reward):
        
        ## action_count of each arm
        self.count_of_every_action_selected[action] += 1

        ## mean reward, of selected action
        n = self.count_of_every_action_selected[action]
        self.average_reward_of_every_action[action] += (reward - self.average_reward_of_every_action[action]) / n


def run_experiment(true_means, iterations):
    env = UCBEnv([1.5, 2.5, 2.0, 1.7])
    agent = UCBAgent(len(true_means))


    actions = []

    for i in range(iterations):
        action = agent.select_actions()
        reward = env.give_rewards(action)
        agent.update(action, reward)

        actions.append(action)


    return agent.average_reward_of_every_action, agent.count_of_every_action_selected


