class Environment:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.actions = ["up", "down", "left", "right"]
        self.terminal_state = (n_rows - 1, n_cols - 1)

        # build state space once
        self.states = [
            (i, j)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

    def next_state(self, state, action):
        """
        Deterministic transition function.
        Input:
            state: (i, j)
            action: one of self.actions
        Output:
            next_state: (i', j')
        """
        if state == self.terminal_state:
            return state

        i, j = state

        if action == "up":
            i_new, j_new = i - 1, j
        elif action == "down":
            i_new, j_new = i + 1, j
        elif action == "left":
            i_new, j_new = i, j - 1
        elif action == "right":
            i_new, j_new = i, j + 1
        else:
            raise ValueError("Invalid action")

        # boundary check (out-of-index handling)
        if (
            i_new < 0 or i_new >= self.n_rows or
            j_new < 0 or j_new >= self.n_cols
        ):
            return state  # stay in place

        return (i_new, j_new)

    def reward(self, state, action, next_state):
        """
        Reward function.
        """
        if state == self.terminal_state:
            return 0
        return -1





#### more is not written yet