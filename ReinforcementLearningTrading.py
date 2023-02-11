import numpy as np
import random

# Define the environment
class Environment:
    def __init__(self,data):
        self.data = data
        self.current_step = 0
        self.num_steps = len(self.data)-1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.profit = 0
        self.inventory = []

    def take_action(self,action):
        current_price = self.data[self.current_step]
        if action == "buy" and self.profit >= current_price:
            self.inventory.append(current_price)
            self.profit -= current_price
        elif action =="sell" and len(self.inventory)>0:
            bought_price = self.inventory.pop(0)
            self.profit += current_price - bought_price
        self.current_step +=1
        done =self.current_step == self.num_steps
        reward = self.profit
        return reward, done

# Define the Q-table
class QTable:
    def __init__(self,actions):
        self.actions = actions
        self.q_table = {}

    def get_q_value(self,state,action):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def update_q_value(self,state,action,value):
        if state not in self.q_table:
            self.q_table[state] = {a:0.0 for a in self.actions}
        self.q_table[state][action] = value


# Define the Q-Learning agent
class QLearningAgent:
    def __init__(self,actions,learning_rate=0.01,discount_factor=0.95,exploration_prob=0.1):
        self.actions = actions
        self.learning_rate = discount_factor
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = QTable(self.actions)

    def choose_action(self,state,exploration_prob):
        if np.random.uniform(0,1)<exploration_prob:
            np.random.choice(self.actions)
        else:
            q_values = [self.q_table.get_q_value(state,a) for a in self.actions]
            action = self.actions[np.argmax(q_values)]
        return action

    def learn(self,state,action,reward,next_state,done):
        q_value = self.q_table.get_q_value(state,action)
        if done:
            q_target = reward
        else:
            q_target = reward+self.discount_factor*max([self.q_table.get_q_value(next_state,a) for a in self.actions])
        q_new = q_value+self