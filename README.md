# Code for the Paper "Maximizing Contrasting Opinions in Signed Social Networks"

The associated datasets can be obtained from this google drive link: [https://drive.google.com/drive/folders/1hHn14eYehzRp8nk_supRfnhahXDDjjmn?usp=sharing](https://drive.google.com/drive/folders/1hHn14eYehzRp8nk_supRfnhahXDDjjmn?usp=sharing).

To run, simply all the data files from any social network of your choice (from the Google drive link above) and call `main.py` program from python3. Please install `numpy` and `scipy` first.

## Input Parameters

Program parameters:
1. `--t`: timestep
2. `--k`: value of `k` - the number of seeds
3. `--tkp`: alternative means of specifying `k`, as a percentage of the total nodes of the network
4. `--target`: the percentage of the network to be considered as the campaigners target

So, an example influence maximization run on Epinions for timestep `5`, with `10%` of the network as seed and `100%` of the network as target yields:
```
... $ ls
adjacency-matrix.npz             main.py         README.md
baseline3-transition-matrix.npz  partitions.npy  transition-matrix.npz
... $ python3 main.py --t=5 --tkp=10 --target=100
	5,	13258,	132585,	12515.39,	14002.75,	15074.54,	17050.17,	6.97,	52.39,	68.44,	135.17,	0.05936,	0.10723,	0.26792,	0.13931
```

## Output Values

The output values are designed for easy piping into csv files for plotting and analysis. The columns are:

1. timetep
2. size of seed set: k
3. number of nodes in campaigners target
4. Expected number of correctly influenced nodes by Random (Baseline)
5. Expected number of correctly influenced nodes by Degree (Baseline)
6. Expected number of correctly influenced nodes by Individual InfMax (Baseline)
7. Expected number of correctly influenced nodes by COSiNe Max
8. Influence percentage w.r.t. all targets as seeds by Random (Baseline)
9. Influence percentage w.r.t. all targets as seeds by Degree (Baseline)
10. Influence percentage w.r.t. all targets as seeds by Individual InfMax (Baseline)
11. Influence percentage w.r.t. all targets as seeds by COSiNe Max
12. Running time to find seed set by Random (Baseline)
13. Running time to find seed set by Degree (Baseline)
14. Running time to find seed set by Individual InfMax (Baseline)
15. Running time to find seed set by COSiNe Max


