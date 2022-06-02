# summarization-research

Setup:
```bash
git submodule update --init --recursive
# Setup ~/.aws/credentials personal profile
# Change this to be public
conda create -n summarization3.6 python=3.6
conda activate summarization3.6
pip install -r requirements.txt
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

ICHEC: Rouge evaluate
```bash
unzip cnn-dailymail/finished_files.zip -d cnn-dailymail
unzip pointer-generator/checkpoint/pretrained_model_tf1.2.1.zip -d pointer-generator/checkpoint
cd pointer-generator
sbatch modeling/pointer-generator-ichec.sh
```

Local: FactCC factual correctness data generation:
```bash
python modeling/score.py --mode preprocess --cnndm $PWD/../data/cnndm --summaries $PWD/../summarization-research/pointer-generator/checkpoint/pretrained_model_tf1.2.1/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/reference --evaluation /Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/factCC/evaluation/reference
```

ICHEC: FactCC factual correctness score:
```bash
cd factCC
tar xzf checkpoint/factcc-checkpoint.tar.gz -C checkpoint
sbatch modeling/scripts/factcc-ichec-score.sh
```


# Todo
* Rouge score: need to copy and paste all summaries and references labelled as 000_decoded.txt
* Refactoring of current loader.
* Implement and run OpenIE triplet precision score with coreference resolution using Spacy.
* Implement and run FEQA score https://github.com/esdurmus/feqa