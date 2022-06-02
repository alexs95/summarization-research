# Abstractive Text Summarization with Reinforced Factual Correctness

## Setup

```bash
git submodule update --init --recursive
# Setup ~/.aws/credentials personal profile
# Change this to be public
conda create -n summarization3.6 python=3.6
conda activate summarization3.6
pip install -r requirements.txt
dvc pull
```

## Processes

cnn-dailymail -> pointer-generator -> data/cnndm -> factCC -> openIE -> FEQA


## Summary Generation

* pointer-generator must be re-run to match reference stories with their corresponding decoded stories.
* In the pre-trained outputs it is impossible to do this above, the code needed to be modified to inject identifiers into the files.

### Prepare CNN/DM for pointer-generator (local)

```bash
cd cnn-dailymail
python make_datafiles.py ../data/cnndm/cnn/stories ../data/cnndm/dailymail/stories
```

### Run pointer-generator (ICHEC)

```bash
unzip cnn-dailymail/finished_files.zip -d cnn-dailymail
unzip pointer-generator/checkpoint/pretrained_model_tf1.2.1.zip -d pointer-generator/checkpoint
cd pointer-generator
sbatch modeling/pointer-generator-ichec.sh
```


## ROUGE Evaluation (ICHEC)

```bash
TODO
```


## OpenIE Triplet Precision Scoring

### Data Preprocessing

```bash
TODO
```

### Score Calculation


```bash
TODO
```


## FactCC Scoring

### Data Preprocessing

* The data needs to be preprocessed into the correct input format to run the FactCC scorer.
* Several preprocessing options are available:
    1. sentence vs paragraph 
    2. co-reference resolution
    3. test set vs validation set
    4. reference stories vs decoded stories
* Below command generates data for the cross product of these options (16 datasets).

```bash
cd factCC
tar xzf checkpoint/factcc-checkpoint.tar.gz -C checkpoint
sbatch modeling/scripts/factcc-ichec-preprocess-submit.sh
```

### Score Calculation

* Runs score calculation on cross product of options (output of above).

```bash
cd factCC
tar xzf checkpoint/factcc-checkpoint.tar.gz -C checkpoint
sbatch modeling/scripts/factcc-ichec-score-submit.sh
```


# Todo

* Create table and fill in values for FactCC.
* Rouge score: need to copy and paste all summaries and references labelled as 000_decoded.txt
* Refactoring of current loader.
* Implement and run OpenIE triplet precision score with coreference resolution using Spacy.
* Implement and run FEQA score https://github.com/esdurmus/feqa