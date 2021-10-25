# Replication Guidance

## the simple way: to compute the evaluation scores by yourself

__you should own the prediction files__

1. directly run `t2t_review.sh` if you have access to a cluster (or run the command inside `t2t_review.job` manually)
2. once you already have all evaluation scores (in a CSV file), then check them with the reported ones

### the moderate way: to generate prediction files by yourself, and then repeat the repeat above practice

__you should own the model checkpoints__

1. directly run `t2t.sh` if you have access to a cluster (or run the command inside `t2t_review.job` manually)
2. once you already have all prediction files, then execute the simple way

### the complex way: to train and evaluate the model by yourself, and then repeat the repeat above practices

__you should own the code repo (of course)__

1. replace *evaluate*` in `t2t.sh` with *train* by string matching
2. comment *cp-da-both* and *cp-ta-both* (4 lines totally) to save the effort
3. when all tasks are done, uncomment that 4 lines, and revoke the step.1
4. once you already have all model checkpoints, then execute the moderate way
