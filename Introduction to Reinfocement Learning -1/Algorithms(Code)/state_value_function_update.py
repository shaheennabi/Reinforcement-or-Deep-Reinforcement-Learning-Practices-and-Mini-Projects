## so we are going to update the state value function, considering that we have the model of the environment, and we are doing the policy iteration
## so one of the steps of the policy iteration is updating the state-value function


## ok so state-value function update will happen here by having some env(or in this case maze) maybe the 4*4 maze 
## it means we have the 16 values to update...
## so the update will happen only via policy (the action or state transition) like from (1,1) -- (1,2) it means we need some things
## like current state, policy, discount factor (easy to assign--scalar value), immediate_reward(when we transition into new state), value of the next state

## let's see how we are going to do this
## for current state we are looping over all the states in the 4*4 maze example
## immediate reward --> will be given by the policy (so we also have to define the policy here)
## discount factor ---> assign value like 0.99 (in most cases)
## next state ---> next state will be given by the state transition or simply i will show --like we will increament the current state (let's see)


## ok defining some things

## first let's build the 4*4 maze 

# current_state  -> (r, c)
# next_state     -> the coordinate given by the policy(state)
# immediate_reward: from reward function
# discount: gamma
# V: dictionary storing state values

def update_state_value(current_state, next_state, immediate_reward, discount, V): ## v for maze states
    V[current_state] = immediate_reward + discount * V[next_state]



