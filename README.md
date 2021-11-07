# summarization-research

```bash
git submodule update --init --recursive
# Create project wide requirements or use poetry
# Setup ~/.aws/credentials personal profile
# Change this to be public
conda create -n summarization3.7 python=3.7
conda activate summarization3.7
pip install -r factCC/requirements.txt
pip install 'dvc[s3]'
dvc pull
tar xzf factCC/checkpoint/factcc-checkpoint.tar.gz -C factCC/checkpoint
cd factCC
sbatch modeling/scripts/factcc-ichec-score.sh
```

# Todo
* Create top-level requirements.txt.
* Implement fast, trimmed down score function based on score.py.
* Coreference resolution in loader.
* Refactoring of current loader .
* Modify pre-loading pointer-generator dataset so that stories are available.
* Implement loader functions for loading pointer-generator output.
* Implement OpenIE triplet precision score with coreference resolution using Spacy.
* Run experiments with various configurations to create table.
* Evaluate results in Athena.
* Implement distributed torch.
