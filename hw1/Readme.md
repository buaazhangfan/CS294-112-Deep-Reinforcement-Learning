# CS294-112 HW 1: Imitation Learning

---

###Run the bash script
###### `./hw1.bash` 
###to get all results of hw1




---
#####The following steps are detailed guidance to section 2 and section 3 of hw1 
In order to run this assignment, first you need to make a folder names **expert_data** which saves the output data for expert_policy

`mkdir expert_data`

1. Load up expert policy and run data<br>
Run `python run_expert.py experts/task.pkl task --render --num_rollouts [num]` to run expert policy<br>
Eg. 
`python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 20` for the Hopper task
`python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts 400` for the Reacher task

2. Implement Behavior_cloning<br>
Run `python behavior_cloning.py experts/task.pkl task --render --num_rollouts [num]`to implement BC<br>
Eg. 
`python behavior_cloning.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 20`for the Hopper task
`python behavior_cloning.py experts/Reacher-v2.pkl Reacher-v2 --render --num_rollouts 400`for the Reacher task<br>
This command will generate a `.pkl` file which saves mean value of reward and std of reward with epoch increasing

3. Implement DAgger<br>
Run `python DAgger.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 20`for the Hopper task<br>
This command will also generate a `.pkl` file which saves mean value of reward and std of reward with DAgger iterations.

4. Plot<br>
With the `.pkl` files generated from step2 and step3, run
`python plot.py Hopper-v2 --num_rollouts 20` to generate figures for behavior cloning and DAgger

