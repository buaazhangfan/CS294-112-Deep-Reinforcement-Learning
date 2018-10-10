import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('DQN_Pong.pkl', 'rb') as a:
    data = pickle.loads(a.read())
time_step_a = data['Timestep'][0:300]
mean_reward_a = data['mean'][0:300]
best_reward_a = data['best'][0:300]

with open('DQNAtari_Ponglr_multi0.1.pkl', 'rb') as b:
    data = pickle.loads(b.read())
time_step_b = data['Timestep'][0:300]
mean_reward_b = data['mean'][0:300]
best_reward_b = data['best'][0:300]

with open('DQNAtari_Ponglr_multi5.0.pkl', 'rb') as c:
    data = pickle.loads(c.read())
time_step_c = data['Timestep'][0:300]
mean_reward_c = data['mean'][0:300]
best_reward_c = data['best'][0:300]

with open('DQNAtari_Ponglr_multi10.0.pkl', 'rb') as d:
    data = pickle.loads(d.read())
time_step_d = data['Timestep'][0:300]
mean_reward_d = data['mean'][0:300]
best_reward_d = data['best'][0:300]


plt.figure()
plt.plot(time_step_a, mean_reward_a, color='green', linestyle = '-')
plt.plot(time_step_a, best_reward_a, color='green', linestyle = '--')

plt.plot(time_step_b, mean_reward_b, color='red', linestyle = '-')
plt.plot(time_step_b, best_reward_b, color='red', linestyle = '--')

plt.plot(time_step_c, mean_reward_c, color='blue', linestyle = '-')
plt.plot(time_step_c, best_reward_c, color='blue', linestyle = '--')

plt.plot(time_step_d, mean_reward_d, color='magenta', linestyle = '-')
plt.plot(time_step_d, best_reward_d, color='magenta', linestyle = '--')

plt.title('Q-learning on Pong with different learning rate', fontsize=11)
plt.xlabel('Timesteps')
plt.ylabel('Mean Episode Reward')
plt.grid()
plt.legend(['Mean_lr_multi = 1', 'Best_lr_multi = 1', 'Mean_lr_multi = 0.1', 'Best_lr_multi = 0.1', 'Mean_lr_multi = 5', 'Best_lr_multi = 5', 'Mean_lr_multi = 10', 'Best_lr_multi = 10'])
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,0))
plt.show()