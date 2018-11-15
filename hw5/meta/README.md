# CS294-112 HW 5c: Meta-Learning

Dependencies:

 * Python **3.5**
 * Numpy version 1.14.5
 * TensorFlow version 1.10.5
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==2.3.2

Instructions: [HW5c PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5c.pdf)

### 1. Problem1 Context as Task ID

Run the following command:

`python train_policy.py 'pm-obs' --exp_name <experiment_name> --history 1 -lr 5e-5 -n 200 --num_tasks 4`

### 2. Problem2 Meta-Learned Context

Run the following command:

**With MLP model**

`python train_policy.py 'pm' --exp_name <experiment_name> --history <history> --discount 0.90 -lr 5e-4 -n 60`


**With RNN model**

`python train_policy.py 'pm' --exp_name <experiment_name> --history <history> --discount 0.90 -lr 5e-4 -n 60 --recurrent`

### 3. Problem3 Generalization

Run the following command:

`python train_policy.py 'pm' --exp_name <experiment_name> --history <history> --discount 0.90 -lr 5e-4 -n 60 --recurrent --generalized --granularity <granularity>`

if `--generalized`, the training goals and testing goals will be chosen from chessboard space where 1 corresponds to testing goals and 0 corresponds to training goals. The size of pattern in chessboard is defined by `--granularity`. The value can be chosen from the list `[1,2,4,5,10]` to construct a balanced chessboard.

