## Experiment on Multi-MNIST and Fashion dataset

There is one training script `train.py`. Below are the default options:
```
python train.py --help

       USAGE: train.py [flags]
flags:

train.py:
  --arch: <lenet>: network architecture to use
    (default: 'lenet')
  --[no]debug: Produces debugging output.
    (default: 'false')
  --dset: name of dataset to use;
    repeat this option to specify a list of values
    (default: "['all']")
  --epochs: number of training epochs
    (default: '100')
    (a non-negative integer)
  --lr: learning rate
    (default: '0.001')
    (a non-negative number)
  --n_tasks: number of tasks
    (default: '2')
    (an integer in the range [2, 2])
  --outdir: Output dir to save results
    (default: 'out')
  --solver: <epo|graddrop|gradnorm|individual|itmtl|linscalar|mgda|pcgrad|pmtl>: name of method/solver
    (default: 'epo')

Try --helpfull to get a list of all flags.
```

For example, to run the `epo` solver on all datasets
(`mnist, fashion, fashion_and_mnist`), run the command

```bash
python train.py --solver epo
```


This will create `.pkl` files in the `out` directory (configurable using `--outdir`).
Then use `display_result.py` to obtain the figures.
