import numpy as np
import random

# =========================
# Environment
# =========================

class GridWorld:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.terminal = (rows - 1, cols - 1)
        self.state = None

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        if action == 0:      # up
            r -= 1
        elif action == 1:    # right
            c += 1
        elif action == 2:    # left
            c -= 1
        elif action == 3:    # down
            r += 1
        else:
            raise ValueError("Invalid action")

        # boundary check
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            r, c = self.state

        next_state = (r, c)

        if next_state == self.terminal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done


# =========================
# Behavior Policy (ε-greedy)
# =========================

def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, Q.shape[-1] - 1)

    q_vals = Q[state]
    max_q = np.max(q_vals)
    best_actions = np.where(q_vals == max_q)[0]
    return np.random.choice(best_actions)


# =========================
# Q-learning Agent
# =========================

class QLearningAgent:
    def __init__(self, rows, cols, num_actions, alpha, gamma, epsilon):
        self.Q = np.zeros((rows, cols, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def train(self, env, episodes):
        for _ in range(episodes):
            state = env.reset()
            done = False

            while not done:
                # behavior
                action = epsilon_greedy(self.Q, state, self.epsilon)

                next_state, reward, done = env.step(action)

                # Q-learning target (OFF-POLICY)
                td_target = reward + self.gamma * np.max(self.Q[next_state])
                td_error = td_target - self.Q[state][action]

                self.Q[state][action] += self.alpha * td_error

                state = next_state


# =========================
# Run Pipeline
# =========================

if __name__ == "__main__":
    rows, cols = 4, 4
    num_actions = 4

    env = GridWorld(rows, cols)

    agent = QLearningAgent(
        rows=rows,
        cols=cols,
        num_actions=num_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    agent.train(env, episodes=500)

    print("Learned Q-values:")
    print(agent.Q)
