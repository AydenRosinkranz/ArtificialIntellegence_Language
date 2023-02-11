import pandas as pd
from ReinforcementLearningTrading import Environment, QLearningAgent

# Load the data into a pandas dataframe
df = pd.read_csv("stockprices(2021).csv")
data = df["Adj Close"].tolist()

# Initialize the environment with the data
env = Environment(data)


# Define the columns for the dataframe
columns = ['State','Action','Reward','Next State','Done']

# Initialize the dataframe
df_q_learning = pd.DataFrame(columns=columns)


# Define the actions:
actions = ['buy','hold','sell']

# Initialize the Q-Learning agent
agent = QLearningAgent(actions)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.profit
    done = False
    while not done:
        action = agent.choose_action(state, agent.exploration_prob)
        reward, done = env.take_action(action)
        next_state = env.profit
        agent.learn(state,action,reward,next_state,done)
        state = next_state


        # Update the dataframe
        df_q_learning = df_q_learning.append({'State':state,'Action':action,'Reward':reward,'Next State':next_state,'Done':done},ignore_index=True)


    # Reset the environment for the next episode
    env.reset()

# Print the dataframe
print(df_q_learning)
