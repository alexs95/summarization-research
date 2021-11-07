# summarization-research

```bash
git submodule update --init --recursive
conda create -n summarization3.7 python=3.7
conda activate summarization3.7
pip install -r factCC/requirements.txt
pip install dvc[s3]
dvc pull
tar xzf factCC/checkpoint/factcc-checkpoint.tar.gz -C factCC/checkpoint
cd factCC
sbatch modeling/scripts/factcc-ichec-score.sh
```