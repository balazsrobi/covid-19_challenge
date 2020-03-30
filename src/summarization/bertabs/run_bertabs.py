'''
Based on https://www.kaggle.com/etonydev/abstract-summarization-with-transformers-bart/notebook
and
https://github.com/huggingface/transformers/blob/master/examples/summarization/bertabs/run_summarization.py
'''

# https://ipywidgets.readthedocs.io/en/stable/user_install.html


import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import pipeline
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, SequentialSampler

from tqdm import tqdm

from modeling_bertabs import BertAbs, build_predictor
from transformers import BertTokenizer

from utils_summarization import (
    CovidDataset,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    truncate_or_pad,
)

Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

# DATA_ROOT = '/home/cszsolnai/Projects/covid19/data/CORD-19-research-challenge'
DATA_ROOT = '/home/cszsolnai/Projects/covid19/data/dummy_covid_ds'
BATCH_SIZE = 4
MIN_LENGTH = 50
MAX_LENGTH = 200
BEAM_SIZE = 5
DEVICE = torch.device("cuda")

args = {
    'documents_dir': DATA_ROOT,
    'summaries_output_dir': 'output_dir',
    'compute_rouge': False,
    'no_cuda': False,
    'batch_size': BATCH_SIZE,
    'min_length': MIN_LENGTH,
    'max_length': MAX_LENGTH,
    'beam_size': BEAM_SIZE,
    'alpha': 0.95,
    'block_trigram': True,
    'device': DEVICE
}


# Function definitions


def load_and_cache_examples(documents_dir, tokenizer):
    dataset = CovidDataset(documents_dir)
    return dataset


def save_rouge_scores(str_scores):
    with open("rouge_scores.txt", "w") as output:
        output.write(str_scores)


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    encoded_text = [encode_for_summarization(story, summary, tokenizer) for _, story, summary in data]
    encoded_stories = torch.tensor(
        [truncate_or_pad(story, block_size, tokenizer.pad_token_id) for story, _ in encoded_text]
    )
    encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch


def save_summaries(summaries, path, original_document_name):
    """ Write the summaries in fies that are prefixed by the original
    files' name with the `_summary` appended.

    Attributes:
        original_document_names: List[string]
            Name of the document that was summarized.
        path: string
            Path were the summaries will be written
        summaries: List[string]
            The summaries that we produced.
    """
    for summary, document_name in zip(summaries, original_document_name):
        # Prepare the summary file's name
        if "." in document_name:
            bare_document_name = ".".join(document_name.split(".")[:-1])
            extension = document_name.split(".")[-1]
            name = bare_document_name + "_summary." + extension
        else:
            name = document_name + "_summary"

        file_path = os.path.join(path, name)
        with open(file_path, "w") as output:
            output.write(summary)


def build_data_iterator(args, tokenizer):
    dataset = load_and_cache_examples(args['documents_dir'], tokenizer)
    sampler = SequentialSampler(dataset)

    def collate_fn(data):
        return collate(data, tokenizer, block_size=512, device=args['device'])

    iterator = DataLoader(dataset, sampler=sampler, batch_size=args['batch_size'], collate_fn=collate_fn, )

    return iterator


def format_rouge_scores(scores):
    return """\n
****** ROUGE SCORES ******

** ROUGE 1
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE 2
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE L
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}""".format(
        scores["rouge-1"]["f"],
        scores["rouge-1"]["p"],
        scores["rouge-1"]["r"],
        scores["rouge-2"]["f"],
        scores["rouge-2"]["p"],
        scores["rouge-2"]["r"],
        scores["rouge-l"]["f"],
        scores["rouge-l"]["p"],
        scores["rouge-l"]["r"],
    )


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
            .replace("[unused3]", "")
            .replace("[PAD]", "")
            .replace("[unused1]", "")
            .replace(r" +", " ")
            .replace(" [unused2] ", ". ")
            .replace("[unused2]", "")
            .strip()
    )

    return summary


# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df = pd.read_csv(
    os.path.join(DATA_ROOT, 'metadata.csv'),
    usecols=["title", "abstract", "authors", "doi", "publish_time"],
)
print(df.shape)
# drop duplicates
# df=df.drop_duplicates()
df = df.drop_duplicates(subset="abstract", keep="first")
# drop NANs
df = df.dropna()
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()
# show 5 lines of the new dataframe
print(df.shape)
df.head()

# Make model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
model.to(args['device'])
model.eval()

symbols = {
    "BOS": tokenizer.vocab["[unused0]"],
    "EOS": tokenizer.vocab["[unused1]"],
    "PAD": tokenizer.vocab["[PAD]"],
}

data_iterator = build_data_iterator(args, tokenizer)
predictor = build_predictor(args, tokenizer, symbols, model)

if args['compute_rouge']:
    reference_summaries = []
    generated_summaries = []

    import rouge
    import nltk

    nltk.download("punkt")
    rouge_evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=args.beam_size,
        length_limit_type="words",
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )

# these (unused) arguments are defined to keep the compatibility
# with the legacy code and will be deleted in a next iteration.
args['result_path'] = ""
args['temp_dir'] = ""

data_iterator = build_data_iterator(args, tokenizer)
predictor = build_predictor(args, tokenizer, symbols, model)

for batch in tqdm(data_iterator):
    batch_data = predictor.translate_batch(batch)
    translations = predictor.from_batch(batch_data)
    summaries = [format_summary(t) for t in translations]
    save_summaries(summaries, args['summaries_output_dir'], batch.document_names)

    if args['compute_rouge']:
        reference_summaries += batch.tgt_str
        generated_summaries += summaries

if args['compute_rouge']:
    scores = rouge_evaluator.get_scores(generated_summaries, reference_summaries)
    str_scores = format_rouge_scores(scores)
    save_rouge_scores(str_scores)
    print(str_scores)
