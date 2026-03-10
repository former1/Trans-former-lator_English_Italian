import torch
from torch.utils.data import DataLoader
from tokenizer import ItalianTokenizer
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm

class TranslationCollator:
    def __init__(self, src_tokenizer, tgt_tokenizer):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __call__(self, batch):
        src_ids = [torch.tensor(i["src_ids"]) for i in batch]
        tgt_ids = [torch.tensor(i["tgt_ids"]) for i in batch]

        src_pad_token = self.src_tokenizer.pad_token_id
        src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_token)
        src_pad_mask = (src_padded != src_pad_token)

        tgt_pad_token = self.tgt_tokenizer.special_tokens_dict["[PAD]"]
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_token)

        input_tgt = tgt_padded[:, :-1].clone()
        output_tgt = tgt_padded[:, 1:].clone()

        input_tgt_mask = (input_tgt != tgt_pad_token)
        output_tgt[output_tgt == tgt_pad_token] = -100

        return {
            "src_input_ids": src_padded,
            "src_pad_mask": src_pad_mask,
            "tgt_input_ids": input_tgt,
            "tgt_pad_mask": input_tgt_mask,
            "tgt_outputs": output_tgt,
        }

if __name__ == "__main__":
    path_to_data = "/Users/omer/Desktop/dataset_hf/tokenized_english2italian_corpus"
    dataset = load_from_disk(path_to_data)

    tgt_tokenizer = ItalianTokenizer("trained_tokenizer/italian_wp.json")
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
    loader = DataLoader(dataset["train"], batch_size=128, collate_fn=collate_fn, shuffle=True, num_workers=10)

    for samples in tqdm(loader):
        pass