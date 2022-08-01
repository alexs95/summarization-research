# Deep abstractive summarization with reinforced factual correctness

## Project Structure

The project is made up of four git submodules pointing to forks of other papers used in this research.
DVC is used to manage data and persist experiments.

## Setup

### Local

```bash
git submodule update --init --recursive
# Setup ~/.aws/credentials personal profile
# Change this to be public
conda create -n summarization3.7 python=3.7
conda activate summarization3.7
pip install -r requirements.txt
dvc pull
```

### ICHEC

```bash
cd /ichec/work/ngcom023c
git clone https://github.com/alexs95/summarization-research.git
cd summarization-research
git submodule update --init --recursive
# Setup ~/.aws/credentials personal profile to access S3
conda create -n summarization3.7 python=3.7
conda activate summarization3.7
pip install -r requirements-gpu.txt
dvc pull
```


## Prepare CNN/DM for pointer-generator

In order to be able to evaluate factual correctness scores it is required that the corresponding input article of each
decoded summary is known. In current implementations it is not possible to easily do so. As such the cnn-dailymail
project was modified to save the articles alongside the decoded stories with a unique identifier.

### Modify pre-processing steps to output the article

You can do this locally, on a Macbook Pro this takes ~2 hours.

```bash
cd cnn-dailymail
python make_datafiles.py ../data/cnndm/cnn/stories ../data/cnndm/dailymail/stories
```


## FactCC Scoring

To evaluate the FactCC scorer, a publicly available pre-trained pointer-generator network was used
to generate article, summary pairs to evaluate. Similarly, a publicly available pre-trained FactCC network was used to
generate correctness probabilities from which the scores were calculated.

### Run pre-trained pointer-generator

This takes 7 hours running on ICHEC with a GPU.

```bash
unzip cnn-dailymail/finished_files.zip -d cnn-dailymail
unzip pointer-generator/checkpoint/pretrained_model_tf1.2.1.zip -d pointer-generator/checkpoint
cd pointer-generator
sbatch modeling/pointer-generator-ichec.sh
```


### Pre-trained pointer-generator FactCC score evaluation

Values for Table X.X are based on this.
Several data generation and preprocessing options are available for experimentation purposes.
An experiment is run based on the pre-trained pointer-generator for the cross product of these options (16 datasets).
This takes 8 hours running on ICHEC with a GPU.

1. sentence vs paragraph 
2. co-reference resolution
3. test set vs validation set
4. reference stories vs decoded stories


```bash
cd factCC
tar xzf checkpoint/factcc-checkpoint.tar.gz -C checkpoint
sbatch modeling/scripts/factcc-ichec-preprocess-submit.sh
```


## Policy Gradient w. FactCC scoring

Training the model is a two-step process:

1. Train the pointer-generator with MLE loss as per normal.
2. Further train the pointer-generator via policy gradient, the reward being a combination of ROUGE and FactCC scoring.

### Step 1: Train the pointer-generator
* 20 hours per 50000 iterations.
* Need to train for approximately 200K iterations.

#### Train the pointer-generator with MLE loss
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=train \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/train_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-mleloss \
--max_iter=25000 \
--use_temporal_attention=True \
--intradecoder=True \
--rl_training=False \
--gpu_num=0
```

#### Evaluate the trained pointer-generator
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode='eval' \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/val_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-mleloss \
--use_temporal_attention=True \
--intradecoder=True \
--rl_training=False \
--gpu_num=0
```

#### Decode the test set
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=decode \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/test_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-mleloss \
--rl_training=False \
--intradecoder=True \
--use_temporal_attention=True \
--single_pass=1 \
--beam_size=5 \
--decode_after=0 \
--gpu_num=0
```

#### Run ROUGE evaluation on the decoded test set
```bash
# Install pyrouge into home directory
# See: https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu
conda install -c perl-xml-libxml
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
git clone https://github.com/andersjo/pyrouge.git rouge
pyrouge_set_rouge_path "$PWD/rouge/tools/ROUGE-1.5.5"
python -m pyrouge.test

# Copy files into expected format expected by pyrouge
python RLSeq2Seq/src/rouge_convert.py --path "$PWD/RLSeq2Seq/model/intradecoder-temporalattention-mleloss/decode_val_train_400maxenc_5beam_35mindec_100maxdec_train-ckpt-0"

PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=rouge \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/test_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-mleloss \
--rl_training=False \
--intradecoder=True \
--use_temporal_attention=True \
--single_pass=1 \
--beam_size=5 \
--decode_after=0
```

#### Run FactCC score evaluation
```bash
# Preprocess
PYTHONPATH="${PYTHONPATH}:factCC" python3 factCC/modeling/score.py --mode preprocess --cnndm $PWD/data/cnndm --summaries $PWD/RLSeq2Seq/model/intradecoder-temporalattention-mleloss/decode_val_train_400maxenc_5beam_35mindec_100maxdec_train-ckpt-0/decoded --evaluation $PWD/evaluation/mleloss

# Run Evaluation
PYTHONPATH="${PYTHONPATH}:factCC" python3 factCC/modeling/score.py --mode evaluate --evaluation $PWD/evaluation/mleloss
```


### Step 2: Train the pointer-generator via policy gradient

#### Convert the pointer-generator into a policy gradient model
```bash
# Copy the model from the previous stage
cp -R $PWD/RLSeq2Seq/model/intradecoder-temporalattention-mleloss $PWD/RLSeq2Seq/model/intradecoder-temporalattention-rlloss 

srun -p GpuQ -N 1 -A ngcom023c -t 0:15:00 --pty bash
module unload cuda
module load cuda/11.2
conda activate summarization3.7
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=train \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.0001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/train_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-rlloss \
--max_iter=40000 \
--intradecoder=True \
--use_temporal_attention=True \
--eta=0.0016 \
--rl_training=True \
--convert_to_reinforce_model=True \
--factcc_gpu_num=0 \
--gpu_num=1
```

#### Train the model via policy gradient
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=train \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.0001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/train_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-rlloss \
--max_iter=40000 \
--intradecoder=True \
--use_temporal_attention=True \
--eta=0.0016 \
--rl_training=True \
--factcc_gpu_num=0 \
--gpu_num=1
```

#### Evaluate the policy gradient model
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode='eval' \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.0001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/val_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-rlloss \
--use_temporal_attention=True \
--eta=0.0016 \
--intradecoder=True \
--rl_training=True \
--factcc_gpu_num=0 \
--gpu_num=1
```

#### Decode the test set
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=decode \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.0001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/test_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-rlloss \
--rl_training=True \
--intradecoder=True \
--use_temporal_attention=True \
--eta=0.0016 \
--single_pass=1 \
--beam_size=5 \
--decode_after=0 \
--factcc_gpu_num=0 \
--gpu_num=1
```

#### Run ROUGE evaluation on the decoded test set
```bash
# Copy files into expected format expected by pyrouge
python RLSeq2Seq/src/rouge_convert.py --path "$PWD/RLSeq2Seq/model/intradecoder-temporalattention-rlloss/decode_val_train_400maxenc_5beam_35mindec_100maxdec_train-ckpt-0"

PYTHONPATH="${PYTHONPATH}:factCC" python RLSeq2Seq/src/run_summarization.py \
--mode=rouge \
--enc_hidden_dim=320 \
--dec_hidden_dim=320 \
--batch_size=50 \
--lr=0.0001 \
--data_path="$PWD/cnn-dailymail/finished_files/chunked/test_*" \
--vocab_path="$PWD/cnn-dailymail/finished_files/vocab" \
--log_root="$PWD/RLSeq2Seq/model" \
--exp_name=intradecoder-temporalattention-rlloss \
--rl_training=True \
--intradecoder=True \
--use_temporal_attention=True \
--single_pass=1 \
--beam_size=5 \
--decode_after=0
```

#### Run FactCC score evaluation
```bash
# Preprocess
PYTHONPATH="${PYTHONPATH}:factCC" python3 factCC/modeling/score.py --mode preprocess --cnndm $PWD/data/cnndm --summaries $PWD/RLSeq2Seq/model/intradecoder-temporalattention-rlloss/decode_val_train_400maxenc_5beam_35mindec_100maxdec_train-ckpt-0/decoded --evaluation $PWD/evaluation/rlloss

# Run Evaluation
PYTHONPATH="${PYTHONPATH}:factCC" python3 factCC/modeling/score.py --mode evaluate --evaluation $PWD/evaluation/rlloss
```