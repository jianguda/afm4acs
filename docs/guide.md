# Replication Guidance

## train and evaluate models

__please make sure the datasets are ready__

1. replace *evaluate* in `t2t.sh` with *train* by string matching
2. comment *cp-da-both* and *cp-ta-both* (4 lines totally) to save the effort
3. when all tasks are done, uncomment that 4 lines, and revoke the step.1
4. once you already have all model checkpoints, then execute the following steps

## generate prediction files

__please make sure the checkpoints are ready__

1. directly run `t2t.sh` if you have access to a cluster (or run the command inside `t2t_review.job` manually)
2. once you already have all prediction files, then execute the following steps

## compute the evaluation scores

__please make sure the prediction files are ready__

1. directly run `t2t_review.sh` if you have access to a cluster (or run the command inside `t2t_review.job` manually)
2. once you already have all evaluation scores (in a CSV file), then check them with the reported ones (the experimental environment may cause very slight differences in results)
