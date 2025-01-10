import os
import pandas as pd
import pathlib
import sys
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tqdm.contrib.concurrent import process_map


AA_EXTRA_TOKENS = ['<P>A', '<P>C', '<P>D', '<P>E', '<P>F', '<P>G', '<P>H', '<P>I', '<P>K', '<P>L', '<P>M', '<P>N', '<P>P', '<P>Q', '<P>R', '<P>S', '<P>T', '<P>V', '<P>W', '<P>Y'] # Style 2

def train_tokenizer(source_files, add_aa=True, add_sentinel=True):
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")
    trainer = tokenizer.model.get_trainer()
    trainer.vocab_size = 100
    trainer.special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>","<mod>", "</mod>", "Span-Mask:", "Product:",
                              "Descriptors:", "FunctionPrediction:", "Binding:", "Chem-Mask:"]
    tokenizer.train(source_files, trainer)
    if add_aa:
        tokenizer.add_tokens(AA_EXTRA_TOKENS)
    tokenizer.add_tokens([">"])
    if add_sentinel:
        extra_tokens = ["<extra_id_" + str(i) + ">" for i in range(100)]
        tokenizer.add_tokens(extra_tokens)
    return tokenizer
"""
 TODO add method for post trained tokenizer to add new special tokens
    This should remove the extra tokens, and then add them back... Unless there is a better way to do this. 
"""

def prepend(string):
    return "<P>" + "<P>".join(list(string))

def load_and_prepend(file):
    df = pd.read_csv(str(file), header=None)
    df = df.applymap(prepend)
    return df
def preprocess_aa_sequences(files):
    processed_files = process_map(load_and_prepend, files, chunksize=100000)
    for index, df in enumerate(processed_files):
        save_path = files[index].parents[0] / f"processed{files[index].stem}.txt"
        df.to_csv(save_path, header=None, index=False)


if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == "preprocess":
        base = pathlib.Path("/scratch/tk2801/t5chem_pretrain_data/prot/train_prot")
        files = [base / "train.txt", base / "val.txt"]
        preprocess_aa_sequences(files)
    elif sys.argv[1] == "train":
        assert len(sys.argv) == 3
        base = pathlib.Path("/scratch/tk2801/t5chem_pretrain_data/smiles/train_0")
        files = [base / "train.txt", base / "val.txt"]
        files = [str(file) for file in files]
        tokenizer = train_tokenizer(files)
        os.makedirs("vocab", exist_ok=True)
        vocab_file = f"vocab/{sys.argv[2]}"
        tokenizer.save(vocab_file)
    else:
        raise ValueError("No clue what you are trying to do")

