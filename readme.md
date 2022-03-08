### prepare command
```
dvc run -f -n prepare \
 -d code/src/prepare.py -d data/train_40k.csv \
 -o data/prepared \
  python code/src/prepare.py data/train_40k.csv 

```

## train command

```
dvc run -f -n train \
    -d code/src/train.py \
    -d data/prepared/train_x.pkl \
    -d data/prepared/train_y.pkl \
    -d data/prepared/test_x.pkl \
    -d data/prepared/test_y.pkl \
    -o model/model.pkl \
    python code/src/train.py 
```

## evaluate command

```
dvc run -f -n evaluate \
    -d code/src/evaluate.py \
    -d model/model.pkl \
    -d data/prepared/original_x.pkl \
    -d data/prepared/original_y.pkl \
    -o report \
    python code/src/evaluate.py
```