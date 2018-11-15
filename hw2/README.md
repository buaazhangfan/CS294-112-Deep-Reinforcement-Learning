# CS294-112 HW 2: Policy Gradient

For all command-line expressions that used to run my experiments, they are stored in the `hw2.bash` script with annotations of different problems.

If you want to run the whole expriment, just run:
`./hw2.bash` in the master folder of `train_pg_f18.py`

(For this bash script, it will store all data file into the `./data` folder)

I also provided the data I got with expriments:

1. For problem 4, if you want to get the graph of small batch, the data is stored in `./data_small`, and run `python plot.py data_small/*` then you can get the graph. if you want to get the graph of large batch, the data is stored in `./data_large`, and run `python plot.py data_large/*` then you can get the graph.

2. For problem 5, the data is stored in `./data_InvertedPendulum` folder and run `python plot.py data_InvertedPendulum/*` to get the graph.

3. For problem 7, the data is stored in `./data_lunar` folder and run `python plot.py data_lunar/*` to get the graph.

4. For problem 8, the folder `./data_HalfCheetah`contains all result with different batch size and learning rate. The folder `./data_HalfCheetah_8` stored the result of optimal for four runs. Run `python plot.py data_HalfCheetah/*` to get the graph.


