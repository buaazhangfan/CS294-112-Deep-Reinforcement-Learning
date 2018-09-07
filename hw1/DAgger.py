# Implementing DAgger 



import pickle
import numpy as np 
import tensorflow as tf 
import tf_util
import gym
import load_policy
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import argparse

# Parameters

learning_rate = 0.001
num_epoch = 100
batch_size = 128

# Network Parameters

num_hid_1 = 128
num_hid_2 = 128

#DAgger parameters

num_DAgger = 6

def load_expert_data (filename):
	with open (filename, 'rb') as f:
		data = pickle.loads(f.read())
	return data

def data_preprocessing(x, y):

	x, y = shuffle(x, y, random_state=0)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
	y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
	y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

	return x_train, x_test, y_train, y_test

def next_batch(batch_size, x, y):

	indices = np.random.randint(low = 0, high = len(x), size = batch_size)
	input_batch = x[indices]
	label_batch = y[indices]

	return input_batch, label_batch

def network_model(num_obs, num_act):

	x = tf.placeholder(tf.float32, shape = [None, num_obs], name = 'x')
	y = tf.placeholder(tf.float32, shape = [None, num_act], name = 'y')
	layer_1 = tf.layers.dense(x, num_hid_1, activation = tf.nn.relu, use_bias=True)
	layer_2 = tf.layers.dense(layer_1, num_hid_2, activation = tf.nn.relu, use_bias = True)
	output = tf.layers.dense(layer_2, num_act, activation = None, use_bias = True)

	return output, x, y

def train_network(output, y):
	loss = tf.losses.mean_squared_error(output, y)
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	return loss, train_op

def main():


	parser = argparse.ArgumentParser();
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
	args = parser.parse_args()

	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	training_data = 'expert_data/' + args.envname + '_' + str(args.num_rollouts) + '_data.pkl'
	task = args.envname

	data = load_expert_data(training_data)
	observations = np.array(data['observations'])
	actions = np.array(data['actions'])
	num_obs = observations.shape[1]
	num_act = actions.shape[2]

	# Load network model
	prediction, input_obs, input_act = network_model(num_obs, num_act)
	loss_fun, optimize = train_network(prediction, input_act)

	init = tf.global_variables_initializer()

	#DAgger begins
	mean_reward = []
	std_reward = []
	for i in range(num_DAgger):

		print("The No. %d current size of dataset is: %d" % (i, observations.shape[0]))
		print("Start No. %d training..." % i)
		obs_train, obs_test, act_train, act_test = data_preprocessing(observations, actions)
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(num_epoch):

				num_batch = int(len(obs_train) / batch_size)
				for num in range(num_batch):
					obs_train_batch, act_train_batch = next_batch(batch_size, obs_train, act_train)
					sess.run(optimize, feed_dict = {input_obs: obs_train_batch, input_act: act_train_batch})
			loss = sess.run(loss_fun, feed_dict = {input_obs: obs_train, input_act: act_train})
			print("Training finished, the %d epoch's training loss is: %.08f" % (epoch, loss))

			#Run trained policy on gym and use expert policy to label 

			env = gym.make(args.envname)
			max_steps = args.max_timesteps or env.spec.timestep_limit

			returns = []
			new_obs = []
			new_act = []

			for rollouts in range(args.num_rollouts):
				print('iter', rollouts)
				obs = env.reset()
				done = False
				totalr = 0.
				steps = 0
				while not done:

					my_action = sess.run(prediction, feed_dict = {input_obs:obs[None,:]})
					obs, r, done, _ = env.step(my_action)
					new_obs.append(obs)
					totalr += r
					steps += 1
					if args.render:
						env.render()
					if steps >= max_steps:
						break
				returns.append(totalr)
			print('No. %d' % i)
			print('returns', (returns))
			print('mean return', np.mean(returns))
			print('std of return', np.std(returns))

			mean_reward.append(np.mean(returns))
			std_reward.append(np.std(returns))


			new_act = policy_fn(np.array(new_obs))
			
		observations_agg = np.array(new_obs)

		actions_agg = np.array(new_act)

		print("The No. %d DAgger append %d data" % (i, observations_agg.shape[0]))

		observations = np.concatenate((observations, observations_agg))

		actions = np.concatenate((actions, actions_agg[:,None,:]))

		print('new observations size: ', observations.shape)
		print('new actions size: ', actions.shape)


	print(mean_reward)
	print(std_reward)

	DAgger_result = {'mean_reward': np.array(mean_reward),
					 'std_reward': np.array(std_reward)}


	outfilename = './' + args.envname + '_' + str(args.num_rollouts) + '_data.pkl'

	with open((outfilename), 'wb') as f:
		pickle.dump(DAgger_result, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
	main()











