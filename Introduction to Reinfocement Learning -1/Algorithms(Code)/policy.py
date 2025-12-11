## ok in this file we are going to implement the policy, i mean how the policy looks like,
## assuming we have the model of the env given.
## we are doing policy iteration: first we evaluate, all state values start from 0,
## then follow the random policy initially. In the next step we will do the
## state value function update, and then go greedy by taking the max (policy improvement).

## we are assuming we have the maze of 4*4, where every state is holding the value of 0 initially.

## ok here we will loop over the state values or define the states here

maze_states = {}

for row in range(1, 5):
    for col in range(1, 5):
        maze_states[(row, col)] = 0
## ok so here these are the states and their values are initialized to 0 (for all states)

import random

def policy(maze_states):
    for s in maze_states:                           ## ok here looping over all the maze states
        if maze_states[s] == 0 and not is_terminal(s):
            ## initially the policy will be random
            next_state = random.choice(transition_function(s))

        evaluate = state_update_function(maze_states)
        ## after random the policy will be evaluated separately

        next_state = max(transition_function(s))
        ## after the first pass, the policy will become greedy,
        ## taking the max over possible transitions / actions

        ## here we are assuming we have the transition and state update
        ## functions written separately somewhere else and handled properly

    return next_state


## assuming everything here need is handled separately ...

## this code is refined by chat...