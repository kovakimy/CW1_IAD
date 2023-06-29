from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    OPTForCausalLM,
)
from datasets import DatasetDict
import evaluate
import torch
import numpy as np
import re
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

print("imported success!\n")

model_checkpoint = "facebook/galactica-125M"
model_checkpoint = "facebook/galactica-1.3b"
lr = 1e-3
batch_size = 128
num_epochs = 3
datasets = load_dataset("flax-sentence-embeddings/stackexchange_math_jsonl")
used_dataset, unused_dataset= datasets['train'].train_test_split(test_size=0.8).values()
used_dataset, unused_dataset= used_dataset.train_test_split(test_size=0.9).values()
used_dataset, unused_dataset= used_dataset.train_test_split(test_size=0.9).values()
train_dataset, testval_dataset = used_dataset.train_test_split(test_size=0.5).values()
test_dataset, val_dataset = testval_dataset.train_test_split(test_size=0.5).values()
datasets_full = DatasetDict({
   'train': train_dataset,
   'test': test_dataset,
   'validation': val_dataset})
#datasets_full = load_dataset("./lm_datasets")
print("dataset download success!\n")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenizer.pad_token_id = 1
# tokenizer.pad_token = "<pad>"
# tokenizer.padding_side = "left"

# # setup truncation
# tokenizer.truncation_side = "left"

# # setup special tokens
# tokenizer.bos_token_id = 0
# tokenizer.bos_token = "<s>"

# tokenizer.eos_token_id = 2
# tokenizer.eos_token = "</s>"

# tokenizer.unk_token = "<unk>"
# tokenizer.unk_token_id = 3

# def _insert_split_marker(m: re.Match):
#     """
#     Applies split marker based on a regex match of special tokens such as
#     [START_DNA].
#     Parameters
#     ----------
#     n : str
#         Input text to split
#     Returns
#     ----------
#     str - the text with the split token added
#     """
#     start_token, _, sequence, end_token = m.groups()
#     sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
#     return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"
# def escape_custom_split_sequence(text):
#     return re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])").sub(_insert_split_marker, text)
# def _tokenize(input_text, new_doc) -> torch.LongTensor:
#     """
#     Apply custom preprocessing to input texts and tokenize them.
#     Returns
#     -------
#         input_text : list[str]
#             Texts to be tokenized
#         new_doc : bool
#             If True, prepends the end-of-document (</s>) token to each sequence and fixes
#             padding.
#     """
#     texts = []
#     for text in input_text:
#         text = escape_custom_split_sequence(text)
#         if not text:
#             warnings.warn(
#                 "Found an empty input text. Changing to end-of-document token instead.",
#                 UserWarning
#             )
#             text = tokenizer.eos_token
#         texts.append(text)

#     if new_doc:
#         pad_token = tokenizer.pad_token
#         texts = [pad_token + t for t in texts]

#     encoded = tokenizer(
#         texts,
#         padding="longest",
#         max_length=500,
#         truncation=True
#     )
#     context_tokens = encoded["input_ids"]
#     input_v = torch.LongTensor(context_tokens).to(model.device)

#     if new_doc:
#         input_v[input_v[:, 0] == tokenizer.pad_token_id, 0] = tokenizer.eos_token_id
#     return input_v

# from tqdm import tqdm
# for example in tqdm(datasets_full['train']):
#     example = _tokenize(example['Problem'] + ' <work> ' + example['Rationale'] + " " + example["annotated_formula"], False)
# for example in tqdm(datasets_full['train']):
#     example = _tokenize(example['Problem'] + ' <work> ' + example['Rationale'] + " " + example["annotated_formula"], False)
# for example in tqdm(datasets_full['train']):
#     example = _tokenize(example['Problem'] + ' <work> ' + example['Rationale'] + " " + example["annotated_formula"], False)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["upvoted_answer"]], truncation=True)
tokenized_datasets = datasets_full.map(preprocess_function, batched=True, num_proc=4, remove_columns=datasets_full["train"].column_names)
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    #for k in examples.keys():
    #  print(k)
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print("tokenization datasets success!\n")

#model = OPTForCausalLM.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
print("model download success!\n")

num_epochs = 3
batch_size = 1
training_args = TrainingArguments(
    output_dir="try_few"+'1.3b',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    
)
print("training_args success!\n")
lm_datasets = datasets_full
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    
    #data_collator=data_collator,
    #callbacks=[PeftSavingCallback]
    #compute_metrics=compute_metrics,
)
print("trainer success!\n")
trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.save_model("./my_model_1.3b")