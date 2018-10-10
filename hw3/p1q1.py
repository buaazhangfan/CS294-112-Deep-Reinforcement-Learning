import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('DQN_Pong.pkl', 'rb') as f:
    data = pickle.loads(f.read())
time_step = data['Timestep']
mean_reward = data['mean']
best_reward = data['best']
best_vanilla = best_reward[-1]
plt.figure()
plt.plot(time_step, mean_reward, color='red', linestyle = '-')
plt.plot(time_step, best_reward, color='blue', linestyle = '--')
plt.xlabel('Timesteps')
plt.ylabel('Mean Episode Reward')
plt.legend(['Mean_DQN','Best Mean_DQN'])
plt.title('Vanilla Q-Learning on Pong', fontsize=12)
plt.grid()
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,0))
plt.show()