import numpy as np


# Environment
class Environment:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.terminal_state = (rows - 1, cols - 1)
        self.current_state = None

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        row, col = self.current_state

        if action == 0:      # up
            new_row, new_col = row - 1, col
        elif action == 1:    # right
            new_row, new_col = row, col + 1
        elif action == 2:    # left
            new_row, new_col = row, col - 1
        elif action == 3:    # down
            new_row, new_col = row + 1, col
        else:
            raise ValueError("Invalid action")

        # boundary handling
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            new_row, new_col = row, col

        next_state = (new_row, new_col)

        if next_state == self.terminal_state:
            reward, done = 0, True
        else:
            reward, done = -1, False

        self.current_state = next_state
        return next_state, reward, done



# Agent (Double Q-learning)
class Agent:
    def __init__(self, rows, cols, num_actions, alpha, gamma):
        self.Q_A = np.zeros((rows, cols, num_actions))
        self.Q_B = np.zeros((rows, cols, num_actions))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state, action, reward, next_state, done):
        r, c = state
        nr, nc = next_state

        if np.random.rand() < 0.5:
            # update Q_A
            best_action = np.argmax(self.Q_A[nr, nc])
            target = reward
            if not done:
                target += self.gamma * self.Q_B[nr, nc, best_action]

            self.Q_A[r, c, action] += self.alpha * (target - self.Q_A[r, c, action])
        else:
            # update Q_B
            best_action = np.argmax(self.Q_B[nr, nc])
            target = reward
            if not done:
                target += self.gamma * self.Q_A[nr, nc, best_action]

            self.Q_B[r, c, action] += self.alpha * (target - self.Q_B[r, c, action])





 
# Epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return np.argmax(q_values)







# Training loop
def train_double_qlearning(episodes, rows, cols, num_actions, epsilon, gamma, alpha):
    env = Environment(rows, cols)
    agent = Agent(rows, cols, num_actions, alpha, gamma)


    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            # combine Q-values ONLY for behavior
            combined_q = agent.Q_A[state] + agent.Q_B[state]

            action = epsilon_greedy_policy(combined_q, epsilon)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state

    return agent



# Main
if __name__ == "__main__":
    agent = train_double_qlearning(episodes=50, rows=10, cols=5, num_actions=4, epsilon=0.1, gamma=0.99, alpha=0.1)

    # YES — this is how you inspect combined values
    Q_combined = agent.Q_A + agent.Q_B

    print("Q_A shape:", agent.Q_A.shape)
    print("Q_B shape:", agent.Q_B.shape)
    print("Combined Q:", Q_combined)
    print(agent.Q_A, agent.Q_B)

   
