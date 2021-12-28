# summarization-research

Setup:
```bash
git submodule update --init --recursive
# Setup ~/.aws/credentials personal profile
# Change this to be public
conda create -n summarization3.6 python=3.6
conda activate summarization3.6
pip install -r factCC/requirements.txt
pip install 'dvc[s3]'
dvc pull
```


Local: Data Generation
```bash
cd cnn-dailymail
python make_datafiles.py ../data/cnndm/cnn/stories ../data/cnndm/dailymail/stories
```

ICHEC: pointer-generator on dataset:
```bash
unzip cnn-dailymail/finished_files.zip -d cnn-dailymail
unzip pointer-generator/checkpoint/pretrained_model_tf1.2.1.zip -d pointer-generator/checkpoint
cd pointer-generator
sbatch modeling/pointer-generator-ichec.sh
```

ICHEC: FactCC factual correctness score:
```bash
cd factCC
tar xzf checkpoint/factcc-checkpoint.tar.gz -C checkpoint
sbatch modeling/scripts/factcc-ichec-score.sh
```


# Todo
* Implement fast, trimmed down score function based on score.py.
* Coreference resolution in loader.
* Refactoring of current loader.
* Implement loader functions for loading pointer-generator output.
* Implement OpenIE triplet precision score with coreference resolution using Spacy.
* Run experiments with various configurations to create table.
* Evaluate results in Athena.
* Implement distributed torch.