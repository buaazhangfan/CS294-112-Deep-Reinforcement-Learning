import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('DQN_Pong.pkl', 'rb') as f:
    data = pickle.loads(f.read())
time_step = data['Timestep']
mean_reward = data['mean']
best_reward = data['best']
best_vanilla = best_reward[-1]
print(best_vanilla)

with open('DDQN_Pong.pkl', 'rb') as l:
    data_d = pickle.loads(l.read())
time_step_d = data_d['Timestep']
mean_reward_d = data_d['mean']
best_reward_d = data_d['best']
best_DDQN = best_reward_d[-1]
print(best_DDQN)

plt.figure()
plt.plot(time_step, mean_reward, color='green', linestyle = '-')
plt.plot(time_step, best_reward, color='green', linestyle = '--')
plt.plot(time_step_d, mean_reward_d, color='red', linestyle = '-')
plt.plot(time_step_d, best_reward_d, color='red', linestyle = '--')
plt.title('Vanilla Q-Learning Vs. Double Q-Learning on Pong', fontsize=11)
plt.xlabel('Timesteps')
plt.ylabel('Mean Episode Reward')
plt.legend(['Mean_DQN', 'Best Mean_DQN', 'Mean_DDQN', 'Best Mean_DDQN'])
plt.grid()
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,0))
plt.show()