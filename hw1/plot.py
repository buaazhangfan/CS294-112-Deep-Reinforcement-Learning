import pickle
import numpy as np 
import matplotlib.pyplot as plt

def load_expert_data (filename):
	with open (filename, 'rb') as f:
		data = pickle.loads(f.read())
	return data


def main():


	#Behavior cloning result with the number of epoch
	
	BC_path = 'Hopper-v2_20_bc_data.pkl'
	BC_result = load_expert_data(BC_path)
	BC_mean = np.array(BC_result['mean_reward'])
	BC_std = np.array(BC_result['std_reward'])
	epoch = np.arange(0, 101, 10)
	BC_plot = plt.figure(1)
	p1, = plt.plot(epoch, BC_mean, color='blue', label='Behavor_cloning' )
	plt.errorbar(epoch, BC_mean, ecolor='r', color='blue', yerr = BC_std, fmt = '-o',  elinewidth=2, capsize=4)
	plt.suptitle('Behavorial Cloning: Epoches vs. Reward', fontsize=20)
	plt.xlabel('Number of Training Epoches')
	plt.ylabel('Mean Reward')
	plt.legend()
	plt.show()




	DAgger_path = './Hopper-v2_20_data.pkl'
	DAgger_result = load_expert_data(DAgger_path)
	mean = np.array(DAgger_result['mean_reward'])
	std = np.array(DAgger_result['std_reward'])
	iteration = np.arange(std.shape[0])
	iteration = iteration + 1;


	DAgger_plot = plt.figure(2)
	Dag, = plt.plot(iteration, mean, marker = '*', color='b', label='DAgger Policy')
	plt.errorbar(iteration, mean, yerr = std, fmt = '-*',color='b',ecolor='r' , elinewidth=2, capsize=4)
	plt.suptitle('DAgger Iterations vs. Reward', fontsize=20)
	plt.xlabel('DAgger Iteration')
	plt.ylabel('Mean Reward')
	plt.xlim([0, 6.5])
	plt.ylim([1000, 4000])
	expert = plt.axhline(y=3778.4842779089204, color='k', label='Expert Policy')
	bc = plt.axhline(y=2009.9990, color='g', label='Behaviorial Cloning')
	plt.legend(loc= 4)
	plt.show()





if __name__ == '__main__':
    main()
