import os
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader

from model import Transformer, TransformerConfig
from tokenizer import ItalianTokenizer
from data import TranslationCollator

CHECKPOINT_PATH = "final_checkpoint/model.safetensors"
DATA_PATH       = "/Users/omer/Desktop/dataset_hf/tokenized_english2italian_corpus"
TOKENIZER_PATH  = "trained_tokenizer/italian_wp.json"
NUM_SAMPLES     = 500      
MAX_GEN_LEN     = 100     
BATCH_SIZE      = 1        
DEVICE          = "mps"   
def load_model(checkpoint_path, config, device):
    from safetensors.torch import load_file
    model = Transformer(config)
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device(DEVICE)

    # Tokenizers
    tgt_tokenizer = ItalianTokenizer(TOKENIZER_PATH)
    src_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    config = TransformerConfig(
        embedding_dimension=512,
        num_attention_heads=8,
        encoder_depth=6,
        decoder_depth=6,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        max_src_len=512,
        max_tgt_len=512,
    )

    print("Loading model...")
    model = load_model(CHECKPOINT_PATH, config, device)
    print("Model loaded!")

    dataset = load_from_disk(DATA_PATH)
    test_data = dataset["test"]
    if NUM_SAMPLES > 0:
        test_data = test_data.select(range(min(NUM_SAMPLES, len(test_data))))

    print(f"Evaluating on {len(test_data)} samples...")

    bos_id = tgt_tokenizer.special_tokens_dict["[BOS]"]
    eos_id = tgt_tokenizer.special_tokens_dict["[EOS]"]

    hypotheses = []
    references = []

    for sample in tqdm(test_data):
        # Encode source
        src_ids = torch.tensor(sample["src_ids"]).unsqueeze(0).to(device)

        # creating predictions
        with torch.no_grad():
            pred_ids = model.inference(
                src_ids,
                tgt_start_id=bos_id,
                tgt_end_id=eos_id,
                max_len=MAX_GEN_LEN
            )

        # decoding of predictions
        if isinstance(pred_ids, int):
            pred_ids = [pred_ids]
        hypothesis = tgt_tokenizer.decode(pred_ids, skip_special_tokens=True)

        ref_ids = sample["tgt_ids"]
        ref_ids = [i for i in ref_ids if i != -100]
        reference = tgt_tokenizer.decode(ref_ids, skip_special_tokens=True)

        hypotheses.append(hypothesis)
        references.append(reference)

    # compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n{'='*50}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"{'='*50}")

    print("\nSample translations:")
    for i in range(min(5, len(hypotheses))):
        print(f"\n[{i+1}]")
        print(f"  Hypothesis : {hypotheses[i]}")
        print(f"  Reference  : {references[i]}")

if __name__ == "__main__":
    main()


# Initial Implementation Score:
"""
==================================================
BLEU Score: 21.76
==================================================

Sample translations:

[1]
  Hypothesis : ho parlato come un vero servo pubblico.
  Reference  : parli come una vera dipendente pubblica.

[2]
  Hypothesis : ti ho scritto una lettera quando ero all ' accademia.
  Reference  : e ' un piacere conoscerla. le... le ho scritto quando ero all ' accademia.

[3]
  Hypothesis : - gli scopiti nel negozio...
  Reference  : il bombarolo si intrufola nel magazzino,

[4]
  Hypothesis : non a casa mia, non lo farai.
  Reference  : non a casa mia, non se ne parla.
"""

# Not really good, we can see it gets the meaning but not able to structure good sentences,
# model is like a baby now

