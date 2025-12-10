## In this example, we focus ONLY on how to define rewards in Python
## for a simple maze environment (model-based setting).

## Common reward structure in gridworld/maze:
## - Normal step:         -1
## - Hitting a wall:      -1   (same as a normal step in this example)
## - Reaching the goal:  +10   (terminal reward)

## Let's assume a 4x4 maze (4 rows, 4 columns)
## And define the goal and wall coordinates.

goal_state = (4, 4)  # Terminal state with +10 reward

# Some example walls (you can choose any coordinates)
wall_states = {(2, 3), (2, 4), (3, 3), (3, 4)}

## Reward function:
## It takes:
## - current_state:          (row, col)
## - action or next_state:   direction or coordinate (not important here)
## - next_state:             resulting coordinate after applying the action

def reward(current_state, action, next_state):
    
    # Reward for reaching goal
    if next_state == goal_state:
        return 10
    
    # Reward for hitting a wall (same as step cost)
    if next_state in wall_states:
        return -1
    
    # Normal step reward
    return -1




## refined with chatgpt