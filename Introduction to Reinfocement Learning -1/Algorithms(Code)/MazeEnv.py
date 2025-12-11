import random
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import time

# ---------- CONFIG ----------
GRID_N = 5                          # Size of the grid: 5x5 (states are (1,1) to (5,5))
TERMINAL = (5, 5)                   # Goal / terminal state location
STEP_REWARD = -1.0                     # Reward for every non-terminal move (living penalty)
TERMINAL_REWARD = 10.0              # Reward received when reaching the goal
GAMMA = 0.99                        # Discount factor — how much future rewards are worth
ACTIONS = ["Right", "Left", "Up", "Down"]  # All possible actions

# ---------- State-value table ----------
# Dictionary mapping every state (r,c) → its current estimated value V(s)
# Initially all values are 0.0
V = {(r, c): 0.0 for r in range(1, GRID_N + 1) for c in range(1, GRID_N + 1)}

# ---------- Helpers ----------
def in_bounds(state):
    """Check if a position is inside the grid boundaries."""
    r, c = state
    return 1 <= r <= GRID_N and 1 <= c <= GRID_N

def is_terminal(state):
    """Return True if the state is the goal state."""
    return state == TERMINAL


def transition_function(state, action):
    """
    Deterministic transition: given current state and action,
    return the next state. If move would go out of bounds → stay in place.
    """
    r, c = state
    if action == "Up":    
        candidate = (r - 1, c)
    elif action == "Down":
        candidate = (r + 1, c)
    elif action == "Left":
        candidate = (r, c - 1)
    elif action == "Right":
        candidate = (r, c + 1)
    else:
        raise ValueError("Bad action")
    
    # Stay in current state if candidate is out of bounds
    return candidate if in_bounds(candidate) else state

def state_transition(state, action):
    """
    Full transition model:
    - If already in terminal → stay there and give terminal reward
    - Otherwise → move, give step reward (or terminal reward if goal reached)
    Returns: (next_state, reward, done)
    """
    if is_terminal(state):
        return state, TERMINAL_REWARD, True                     # Already at goal

    next_s = transition_function(state, action)

    if is_terminal(next_s):
        return next_s, TERMINAL_REWARD, True                    # Just reached goal
    return next_s, STEP_REWARD, False                           # Normal move









# ---------- Value & Policy Functions (unchanged) ----------
def evaluate_state_once(state, chosen_action):
    """
    One-step Bellman backup for a fixed action.
    Computes: r + γ × V(next_state)  (or 0 if terminal)
    This is used during policy evaluation.
    """
    next_s, r, done = state_transition(state, chosen_action)
    next_val = 0.0 if done else V[next_s]               # V(terminal) not used after done
    return r + GAMMA * next_val






def policy_for_state(state, first_iteration=False):
    """
    Return the best action according to current value function V.
    - If first_iteration=True → return random action (used only at start)
    - Otherwise → greedy action(s) w.r.t. V(s') for each possible next state
    Ties are broken randomly.
    """
    if is_terminal(state):
        return None                                             # No action in goal

    actions = ["Right", "Left", "Up", "Down"]
    moves = {a: transition_function(state, a) for a in actions}

    if first_iteration:
        return random.choice(actions)                           # Random policy at beginning

    # Greedy policy improvement
    best_value = -float("inf")
    best_actions = []
    for a, ns in moves.items():
        val = V[ns]                                             # Value of resulting state
        if val > best_value:
            best_value = val
            best_actions = [a]                                  # New best
        elif val == best_value:
            best_actions.append(a)                              # Tie → keep it

    return random.choice(best_actions)                          # Random among best







def evaluate_policy_once(chosen_actions, theta=1e-6, max_sweeps=1000):
    """
    Policy Evaluation step:
    Repeatedly apply Bellman expectation backup for the fixed policy
    until the value function stops changing significantly (delta < theta).
    Uses synchronous updates (old V → new V in one full sweep).
    Updates the global V when done.
    """
    global V
    local_V = deepcopy(V)                                       # Work on a copy

    for _ in range(max_sweeps):
        delta = 0.0
        V_new = local_V.copy()

        for s in local_V:
            if is_terminal(s):
                V_new[s] = TERMINAL_REWARD                          # Goal value is fixed
                continue

            a = chosen_actions.get(s)
            if a is None:
                continue                                            # Shouldn't happen

            old_val = local_V[s]
            # Bellman expectation update: V(s) ← r + γV(s')
            V_new[s] = evaluate_state_once(s, a)

            delta = max(delta, abs(V_new[s] - old_val))

        local_V = V_new                                          # Synchronous update

        if delta < theta:                                        # Converged?
            break

    V = local_V                                                  # Write final values back


## above code is designed or skelton designed by me, refined by chat and gemini  (can't write all myself it takes time...)

## the below code is all UI and running the entire maze stack, generated by grok 

"""# ---------- REAL-TIME ANIMATED UI ----------
plt.ion()  # Interactive mode on
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0.5, GRID_N + 0.5)
ax.set_ylim(0.5, GRID_N + 0.5)
ax.set_xticks(range(1, GRID_N + 1))
ax.set_yticks(range(1, GRID_N + 1))
ax.grid(True, linewidth=2)
ax.set_title("Policy Iteration - Live Update", fontsize=16, pad=20)

# Text objects for values and arrows
value_texts = {}
arrow_patches = {}

# Direction mapping for arrows
arrow_map = {
    "Right": ( 0.4,  0.0),
    "Left":  (-0.4,  0.0),
    "Down":  ( 0.0, -0.4),
    "Up":    ( 0.0,  0.4),
    None:    (0, 0)
}

for r in range(1, GRID_N + 1):
    for c in range(1, GRID_N + 1):
        state = (r, c)
        # Value text in center
        txt = ax.text(c, r, f"{V[state]:5.2f}", ha='center', va='center',
                      fontsize=11, fontweight='bold')
        value_texts[state] = txt
        
        # Arrow (initially invisible)
        dx, dy = 0, 0
        arr = ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.15,
                       fc='red', ec='red', alpha=0.9, length_includes_head=True)
        arrow_patches[state] = arr

# Mark terminal state
tx, ty = TERMINAL
value_texts[TERMINAL].set_text("GOAL\n10.0")
value_texts[TERMINAL].set_color("white")
value_texts[TERMINAL].set_backgroundcolor("green")
value_texts[TERMINAL].set_ha("center")
value_texts[TERMINAL].set_fontsize(12)

plt.draw()
plt.pause(1.0)

def update_display(policy_dict, iteration):
    ax.set_title(f"Policy Iteration — Iteration {iteration}  (V is updating...)", fontsize=16, color='blue')
    
    for state in V:
        r, c = state
        val = V[state]
        action = policy_dict.get(state)
        
        # Update value text
        if is_terminal(state):
            txt = "GOAL\n10.0"
            value_texts[state].set_backgroundcolor("green")
        else:
            txt = f"{val:5.2f}"
            value_texts[state].set_backgroundcolor("white" if val > 0 else "black")
            value_texts[state].set_color("black" if val > 0 else "white")
        
        value_texts[state].set_text(txt)
        
        # Update arrow
        dx, dy = arrow_map[action]
        arr = arrow_patches[state]
        arr.set_data(x=c, y=r, dx=dx, dy=dy)
        arr.set_alpha(0 if action is None else 0.9)
        if action is not None and not is_terminal(state):
            arr.set_color("red")
        else:
            arr.set_color("gray")

    plt.draw()
    plt.pause(0.8)  # Watch it slowly

# ---------- MAIN LOOP WITH LIVE UI ----------
if __name__ == "__main__":
    V = {(r, c): 0.0 for r in range(1, GRID_N + 1) for c in range(1, GRID_N + 1)}
    
    # Initial random policy
    current_policy_actions = {s: policy_for_state(s, first_iteration=True) for s in V.keys()}
    
    print("Starting live Policy Iteration... Watch the window!")
    update_display(current_policy_actions, 0)
    time.sleep(1.5)

    policy_stable = False
    iteration_count = 0

    while not policy_stable and iteration_count < 50:
        iteration_count += 1
        print(f"Iteration {iteration_count} running...")

        # Policy Evaluation
        evaluate_policy_once(current_policy_actions)

        # Policy Improvement
        policy_stable = True
        new_policy = {}
        for s in V:
            if is_terminal(s):
                new_policy[s] = None
                continue
            old_a = current_policy_actions[s]
            new_a = policy_for_state(s)  # Greedy w.r.t current V
            new_policy[s] = new_a
            if new_a != old_a:
                policy_stable = False
        current_policy_actions = new_policy

        # LIVE UPDATE
        update_display(current_policy_actions, iteration_count)

    # Final highlight
    ax.set_title("OPTIMAL POLICY & VALUE FUNCTION FOUND!", fontsize=18, color='green', fontweight='bold')
    plt.draw()
    print("\nDone! Optimal policy reached.")
    plt.ioff()
    plt.show()  # Keep window open at the end"""