import numpy as np

# initialize parameters
gamma = 0.9  # discount factor
theta = 0.0001  # small threshold value to stop iteration
n_states = 3  # number of states
n_actions = 2  # number of actions

# fill in probability and reward matrices (example values)
p = [
    [[0.8, 0.1, 0.1], [0, 0, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 0.7, 0.3], [0, 0.8, 0.2]],
    [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
]
r = [[[0,200,0],[0,0,0],[0,0,0]],
     [[0,0,0],[0,50,0],[0,0,0]],
     [[0,0,0],[0,0,0],[-250,0,0]]
     ]

# initialize policy and value function arrays
policy = np.zeros(n_states, dtype=int)
V = np.zeros(n_states)

# run policy iteration algorithm
while True:
    # policy evaluation step
    delta = 0
    for s in range(n_states):
        v = V[s]
        a = policy[s]
        V[s] = sum(p[s][a][s1] * (r[s][a][s1] + gamma * V[s1]) for s1 in range(n_states))
        delta = max(delta, abs(v - V[s]))
    if delta < theta:
        break

    # policy improvement step
    policy_stable = True
    for s in range(n_states):
        old_action = policy[s]
        policy[s] = np.argmax(
            [sum(p[s][a][s1] * (r[s][a][s1] + gamma * V[s1]) for s1 in range(n_states)) for a in range(n_actions)])
        if old_action != policy[s]:
            policy_stable = False
    if policy_stable:
        break

q = np.zeros((3,3))
for s in range(0,3):
    for j in range(0,3):
        q[s][j] = sum([(V[k]*0.9+r[s][j][k])*p[s][j][k]+V[s] for k in range(0,3)])

print("Optimal Policy:", policy)
print("Optimal Value Function:", V)
print("Optimal action value function:\n",q)
print("\nWhere the market states are bear, bull, stagnant accordingly,\n"
      "and the actions are buy, hold and sell accordingly.")
