import numpy as np

class Environment:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.terminal_state = (rows - 1, cols - 1)
        self.current_state = None

    def _coord_to_state_id(self, coord):
        row, col = coord
        return row * self.cols + col

    def reset_env(self):
        self.current_state = self.start
        return self._coord_to_state_id(self.current_state)

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

        # boundary check
        if (
            new_row < 0 or new_row >= self.rows or
            new_col < 0 or new_col >= self.cols
        ):
            new_row, new_col = row, col

        self.current_state = (new_row, new_col)
        next_state_id = self._coord_to_state_id(self.current_state)

        if self.current_state == self.terminal_state:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return next_state_id, reward, done



class Agent:
    def __init__(self, num_states, num_actions, alpha, gamma):
        self.q_values = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r, s_next, a_next):
        current = self.q_values[s, a]
        next_val = self.q_values[s_next, a_next]

        target = r + self.gamma * next_val
        self.q_values[s, a] += self.alpha * (target - current)



def epsilon_greedy_policy(q_vals, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_vals))
    else:
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return np.random.choice(best_actions)



def train_sarsa(episodes=50, rows=5, cols=5, alpha=0.1, gamma=0.99, epsilon=0.1):
    env = Environment(rows, cols)
    agent = Agent(
        num_states=rows * cols,
        num_actions=4,
        alpha=alpha,
        gamma=gamma
    )

    for _ in range(episodes):
        state = env.reset_env()
        action = epsilon_greedy_policy(agent.q_values[state], epsilon)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(agent.q_values[next_state], epsilon)

            agent.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

    return agent



if __name__ == "__main__":
    training = train_sarsa()
    print(training.q_values)
