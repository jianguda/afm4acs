# acs
Automatic Code Summarization

## Repo Structure

```markdown
**data** datasets
----**basts** the placeholder directory for the basts data (a shell script included)
----**gh** the placeholder directory for the gh data (a shell script included)
----**sit** the placeholder directory for the sit data (a shell script included)
**docs** documentations
----`guide.md` the guidance for experiment replication
**logs** experiments logs
----**adamo** the scores of all models except models mentioned below
----**adamo1** the scores of CP models (24 hours)
----**adamo2** the scores of CP models (48 hours)
----**noisy** the scores of noise models
**models** model checkpoints
----**checkpoints** the placeholder directory for checkpoints of all models except the mentioned ones below
----**checkpoints1** the placeholder directory for checkpoints of CP models (24 hours)
----**checkpoints2** the placeholder directory for checkpoints of CP models (48 hours)
----**pretrained** the placeholder directory for model dumpfile of all models except the mentioned ones below
----**pretrained1** the placeholder directory for model dumpfile of CP models (24 hours)
----**pretrained2** the placeholder directory for model dumpfile of CP models (48 hours)
**results** experiments logs (prediction files)
----**adamo** the results of all models except the mentioned ones below
----**adamo1** the results of CP models (24 hours)
----**adamo2** the results of CP models (48 hours)
----**noisy** the results of noise models
----`scores.csv` the CSV file containing the evaluation scores of all experiments
**src** implementations
----`metric.py` the code file for computing scores
----`noisy_model.py` the code file for noise models
----`t2t.py` the code file for all models except noise models
----... the script files
`conda.yml` the configuration file for conda environment
```

## How to Reproduce Experiments

Please check **`docs/guide.md`**

## Where to Get Model Checkpoints

1. Download files from zenodo.org

- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5594865.svg)](https://doi.org/10.5281/zenodo.5594865)
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5595080.svg)](https://doi.org/10.5281/zenodo.5595080)

2. Once you download the model checkpoints, place them at thr right place (refer to the Repo Structure)

## Where to Get the Data

Please check scripts inside **`data`**

- BASTS data: the dataset used in BASTS
- SiT data: the dataset used in SiT
- GH data: the code_x_glue dataset in transformers.datasets
