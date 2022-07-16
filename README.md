# Reinforcing Factual Correctness into Deep Summarizers

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

## FEQA Scoring

### Data Preprocessing

```bash
TODO
```

### Score Calculation


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

Create table with:
```bash
cat slurm-880069.out| egrep '/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/|0\.'
>>>
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_paragraph_decoded_resolved/
0.49508145
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_paragraph_decoded_unresolved/
0.49885604
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_paragraph_reference_resolved/
0.45128733
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_paragraph_reference_unresolved/
0.45417878
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_sentence_decoded_resolved/
0.96813047
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_sentence_decoded_unresolved/
0.97131264
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_sentence_reference_resolved/
0.9266849
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/test_sentence_reference_unresolved/
0.9245219
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_paragraph_decoded_resolved/
0.5040538
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_paragraph_decoded_unresolved/
0.5057918
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_paragraph_reference_resolved/
0.45776337
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_paragraph_reference_unresolved/
0.46124858
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_sentence_decoded_resolved/
0.9680663
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_sentence_decoded_unresolved/
0.97193044
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_sentence_reference_resolved/
0.9255798
/ichec/home/users/ashapovalov/projects/summarization-research/factCC/evaluation/val_sentence_reference_unresolved/
0.924659
```

## RLSeq2Seq
MLE Train
```
--mode=train --data_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/chunked/train_* --vocab_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/vocab --log_root=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/RLSeq2Seq/model --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=20000 --use_temporal_attention=True --intradecoder=True --rl_training=False
```

Convert:
```
--mode=train --data_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/chunked/train_* --vocab_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/vocab --log_root=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/RLSeq2Seq/model --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --convert_to_reinforce_model=True
```

RL Train:
```
--mode=train --data_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/chunked/train_* --vocab_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/vocab --log_root=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/RLSeq2Seq/model --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True
PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=train \
--data_path="$PWD/data/finished_files/chunked/train_*" \
--vocab_path="$PWD/data/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-withpretraining \
--batch_size='80' \
--max_iter=40000 \
--intradecoder=True \
--use_temporal_attention=True \
--eta=2.5E-05 \
--rl_training=True
```

Decode:
```
PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=decode
--data_path="/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/chunked/train_*"
--vocab_path="/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/data/finished_files/vocab"
--log_root="/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/RLSeq2Seq/model"
--exp_name=intradecoder-temporalattention-withpretraining
--rl_training=True
--intradecoder=True
--use_temporal_attention=True
--single_pass=1
--beam_size=4
--decode_after=0
```


--mode=train
--data_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/cnn-dailymail/finished_files/chunked/train_*
--vocab_path=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/cnn-dailymail/finished_files/vocab
--log_root=/Users/Alexey.Shapovalov@ig.com/Projects/nuig/summarization-research/RLSeq2Seq/model
--exp_name=intradecoder-temporalattention-withpretraining
--batch_size=80
--max_iter=40000
--intradecoder=True
--use_temporal_attention=True
--eta=2.5E-05
--rl_training=True


# TODO

* Create table and fill in values for FactCC.
* Rouge score: need to copy and paste all summaries and references labelled as 000_decoded.txt
* Refactoring of current loader.
* Implement and run OpenIE triplet precision score with coreference resolution using Spacy.
* Implement and run FEQA score https://github.com/esdurmus/feqa