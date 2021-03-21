## Experiment on Multi-MNIST and Fashion dataset

First run the training scripts:
```
python individual_train.py
python linscalar_train.py
python epo_train.py
python pmtl_train.py
python mgda_train.py
python pcgrad_train.py
python itmtl_train.py
python graddrop_train.py
python gradnorm_train.py
```

This will create `.pkl` files in the `results` folder. Then use `display_result.py` to obtain the figures.
