# This is the source code of "Enhancing Transformer-based Models’ Natural Language Understanding based on Word Importance Investigation".
# The source code is composed of a fine-tuning, model, and a WI validation part.

For fine-tuning, these are the modification of run_glue.py, run_squad.py, and run_swag.py from transformer version 4.20.1.
For model, these are the modification of modeling_bert.py, modeling_electra.py, and modeling_roberta.py from transformer version 4.20.1.

## Usage
1. Copy each modeling_"model".py file to the corresponding model directory within the transformer library.
2. Run the run_"task".py file.
3. After the preprocessing process is done, please enter 3 for wandb option.

## Model
- "bert-base-uncased": BERT
- "roberta-base": RoBERTa
- "google/electra-base-discriminator": ELECTRA

## Metric
- 1: basic TFIDF
- 2: TFIDF w/ norm 1
- 3: TFIDF w/ norm 2

# Fine-tune GLUE
1. run_glue.py
- Following the instructions below.
- python glue.py --model_name_or_path "model_name" --tokenizer_name "model_name" --no_use_fast --task_name "task_name" --do_train --do_eval --output_dir "your_path" --max_seq_length 128 --per_gpu_train_batch_size 32 --num_train_epochs 3 --use_wi "choose_metric"

2. run_squad.py
- Following the instructions below.
- python run_squad.py --model_name_or_path "model_name" --tokenizer_name "model_name" --do_train --do_eval --output_dir "your_path" --per_gpu_train_batch_size 32 --num_train_epochs 3 --dataset_name squad --use_wi "choose_metric"

3. run_swag.py
- Following the instructions below.
- python run_swag.py --model_name_or_path "model_name" --tokenizer_name "model_name" --no_use_fast_tokenizer --do_train --do_eval --output_dir "your_path" --max_seq_length 128 --per_gpu_train_batch_size 16 --num_train_epochs 3 --pad_to_max_length True --use_wi "choose_metric"

# Fine-tune SuperGLUE
[jiant] (https://github.com/nyu-mll/jiant).

# Few-shot learning
[instruct-eval] (https://github.com/declare-lab/instruct-eval).

# WI validation model
## 1. Dependency Dataset Generator.ipynb
- Perform WI validation on the target dataset, randomly select 2000 sentences, and generate the dataset by choosing positions of tokens with inter-token dependencies as well as positions of tokens without inter-token dependencies. 
- Requires preprocessed data from the respective downstream task.

## 2. WI Validation Performance Evaluation.ipynb
- Based on the extracted sentences and relationships between tokens within the sentences, extract attention values from those tokens. Then, compare the performance of dependency relationship prediction. 
- Requires the trained model and its configuration.