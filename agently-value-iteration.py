# First reveiw for some prerequirements
# agent = trader
# environment:
# state: {bear, bull, stagnant}
# action: {buy, hold, sell}
# reward: {}

from scipy.spatial.distance import euclidean


# Using value iteration method to implement the whole process of algorithm trading
# We need to initialize values first
# We claim state 1, 2, 3 as bear market, bull market and the stagnant market accordingly
v = [1,1,1]
gamma = 0.9
# Probability matrix is a 3D matrix the first dimension is its original status and the
# second one is actions
# last is its moving-forward status
p = [[[0.8,0.1,0.1],[0,0,0],[0,0,0]],
     [[0,1,0],[0,0.7,0.3],[0,0.8,0.2]],
     [[0,0,0],[0,0,1],[1,0,0]]
]
# Next the reward matrix with transition conditions
r = [[[0,200,0],[0,0,0],[0,0,0]],
     [[0,0,0],[0,50,0],[0,0,0]],
     [[0,0,0],[0,0,0],[-250,0,0]]
     ]
# Here is an while function
delta = 1
while delta>0.001:
    temp = v
    for s in range(0,3):
        v[s] = max([sum([(temp[k]*0.9+r[s][j][k])*p[s][j][k]+v[s] for k in range(0,3)]) for j in range(0,3)])
    delta = min(delta,euclidean(temp,v))
print(v)