# CS294-112 HW 3: Q-Learning


---
Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

###Problem 1

#####Question 1

Run `python run_dqn_atari.py` directly with vanilla Q-learning and random seed with learning multiplier 1

Plot
`python p1q1.py` (Replace the `.pkl` filename)

#####Question 2

Run `python run_dqn_atari.py --double` with double Q-learning and random seed.

Plot
`python p1q2.py` (Replace the `.pkl` filename) to plot the vanilla Q-learning and Double Q-learning.

#####Question 3

Run `python run_dqn_atari.py -m <> --seed <--double>` with the learning multiplier and a fixed seed number **5000**, if `--double` then with double Q-learning else vanilla Q-learning.

Plot
`python p1q3.py` (Replace the `.pkl` filename) to plot different learning curves with learning multiplier.

###Problem 2

#####Question 1
Run 
`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1`
`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1-ngsptu 100`
`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu100 -ngsptu 1`
`python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu10 -ngsptu 10`

Plot
`python plot.py data_CartPole/*`

#####Question 2
Run
`python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10` for InvertedPendulum task
Run
`python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2-s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10` for HalfCheetah task

Plot
`python plot.py data_InvertedPendulum/*`

`python plot.py data_HalfCheetah/*`

