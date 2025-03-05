import linecache
import os
import random
import re
from copy import deepcopy
from functools import partial
from typing import Dict, Tuple
import h5py
import lmdb
import numpy as np
import pandas as pd
import polars as pr
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import selfies
import torch
from Bio import SeqIO
from rdkit.Chem import MolFromSequence, MolToSmiles
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerFast, AutoTokenizer


def process_protein(protein_string):
    return "<P>" + "<P>".join(list(protein_string))


class LMDB(Dataset):
    def __init__(self, data_dir, teardown=False):
        self.data_dir = data_dir
        # Check if the data_dir folder exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist.")
        self.env = None
        self.teardown = teardown

    def _init_db(self):
        self.env = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        if self.env is None:
            self._init_db()
        if self.teardown:
            self.env.close()
            self.env = None
        return self.env.stat()['entries']

    def __getitem__(self, idx, selection_function=None):
        if self.env is None:
            self._init_db()
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        with self.env.begin(write=False) as txn:
            data = txn.get(str(idx).encode()).decode()
        if selection_function is not None:
            data = selection_function(data)
        if self.teardown:
            self.env.close()
            self.env = None
        return data


class LMDBTokenizer(LMDB):

    def __init__(self, data_dir, tokenizer, max_length=1024, prefix="", teardown=False):
        super().__init__(data_dir, teardown)
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx, selection_function=None):
        data = super().__getitem__(idx, selection_function)
        return self.tokenizer(self.prefix + data, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.max_length)






class EsmAlignDataset(LMDB):
    def __init__(self, data_dir, tokenizer_t5chem, max_length=1024, prefix="", teardown=False):
        super().__init__(data_dir)
        self.data_dir = data_dir
        self.env = None
        self.tokenizer_t5chem = tokenizer_t5chem
        self.tokenizer_esm = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.max_length = max_length
        self.prefix = prefix
        self.teardown = teardown

    def __getitem(self, idx):
        data = super().__getitem__(idx)
        t5chem_inputs = self.tokenizer_t5chem(data, return_tensors="pt", padding=True,
                                              truncation=True, max_length=self.max_length, return_token_type_ids=False)
        data = data.replace("<P>", "")
        esm_inputs = self.tokenizer_esm(data, return_tensors="pt", padding=True, truncation=True,
                                        max_length=self.max_length)
        return {"t5chem": t5chem_inputs, "esm": esm_inputs}


class SpanMaskDataset(LMDBTokenizer):
    def __init__(self, data_dir, tokenizer, max_length=1024, prefix="Span-Mask:", teardown=False, mlm_probability=0.15,
                 mean_noise_span_length=3):
        super().__init__(data_dir, tokenizer, max_length, prefix, teardown)
        self.tokenizer_length = len(tokenizer)
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = max_length
        self.target_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.prefix = prefix

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        example = super().__getitem__(idx)
        batch_inputs = []
        batch_labels = []
        # batch = self.tokenizer(self.prefix + example, return_tensors="pt", padding=True, truncation=True, max_length=self.input_length)
        input_ids = example["input_ids"]
        batch_size, expandend_input_length = input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for _ in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch_inputs.append(torch.from_numpy(self.filter_input_ids(input_ids, input_ids_sentinel)).flatten())
        labels = self.filter_input_ids(input_ids, labels_sentinel)
        batch_labels.append(torch.from_numpy(labels).flatten())

        padded_inputs, padded_labels = pad_complex_sequence(batch_inputs, batch_labels, self.input_length,
                                                            self.input_length, self.pad_token_id)
        # attention mask
        padded_labels[padded_labels == self.pad_token_id] = -100
        attention_mask = (padded_inputs != self.pad_token_id).long()
        processed = BatchEncoding({"input_ids": padded_inputs, "attention_mask": attention_mask,
                                   "labels": padded_labels, })  # "decoder_input_ids": padded_labels})

        return processed

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.mlm_probability))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.tokenizer_length - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids


# TODO Needs CleanUp
class ChemMaskDataset(LMDBTokenizer):
    def __init__(self, data_dir, tokenizer, max_length=1024, fragment_length=None, prefix="Chem-Mask:", teardown=False,
                 mlm_probability=0.15, mean_noise_span_length=3):
        super().__init__(data_dir, tokenizer, max_length, prefix, teardown)
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.target_length = max_length
        self.input_length = max_length
        self.regex = re.compile(fr'(<[^>]+>)|\b{prefix}\b')
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.bos_token_id
        self.selfies = False
        self.partial_selection_function = None
        if fragment_length is not None:
            self.partial_selection_function = partial(self.selection_function, fragment_length=fragment_length)

    def __len__(self):
        return super().__len__()

    @staticmethod
    def selection_function(protein_sequence, fragment_length):
        protein_sequence_ = deepcopy(protein_sequence.replace("<P>", ""))
        if len(protein_sequence_) <= fragment_length:
            return protein_sequence
        else:
            maxmimum_possible_index = len(protein_sequence_) - fragment_length - 1
            fragment_index = random.randint(0, maxmimum_possible_index)
            frag_protein_sequence = protein_sequence_[fragment_index:fragment_index + fragment_length]
        return "<P>" + "<P>".join(frag_protein_sequence)

    def __getitem__(self, idx):
        item = super().__getitem__(idx, self.partial_selection_function)
        batch_inputs = []
        batch_labels = []
        input_ids = item['input_ids']
        batch_size, expandend_input_length = input_ids.shape
        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        batch_inputs.append(torch.from_numpy(self.filter_input_ids(input_ids, input_ids_sentinel)).flatten())
        labels = self.filter_input_ids(input_ids, labels_sentinel)
        batch_labels.append(torch.from_numpy(labels).flatten())
        decoded_aa_inputs = self.tokenizer.batch_decode(batch_labels)
        smiles_labels = self.swap_aa_for_smiles(decoded_aa_inputs)
        batch_labels = self.tokenizer(smiles_labels, return_tensors="pt", padding=True, truncation=True,
                                      max_length=self.target_length)["input_ids"]

        padded_inputs, padded_labels = pad_complex_sequence(batch_inputs, batch_labels, self.input_length,
                                                            self.input_length, self.pad_token_id)

        padded_labels[padded_labels == self.pad_token_id] = -100
        # attention mask
        attention_mask = (padded_inputs != self.pad_token_id).long()
        processed = BatchEncoding({"input_ids": padded_inputs, "attention_mask": attention_mask,
                                   "labels": padded_labels, })  # "decoder_input_ids": padded_labels})
        return processed

    def random_spans_noise_mask(self, length):
        orig_length = length
        num_noise_tokens = int(np.round(length * self.mlm_probability))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub-segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)
        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def swap_aa_for_smiles(self, decoded_inputs):  # TODO rewrite. This is unclear.
        decoded_inputs = [sample.replace(" ", "").replace("<P>", "") for sample in decoded_inputs]
        for sample_index, sample in enumerate(decoded_inputs):
            filtered_sample = list(filter(None, self.regex.split(sample)))
            for tokens_index, tokens in enumerate(filtered_sample):  # O(n2)
                if tokens.startswith("<") or self.prefix == tokens:
                    pass
                else:
                    try:
                        smiles = MolToSmiles(MolFromSequence(tokens))
                    except:
                        print(f"offending tokens {tokens}")
                        print(MolFromSequence(tokens))
                        raise Exception(
                            f"offending tokens {tokens}")  # RDKIT needs refactored. # TODO Implement a cache here.
                    if self.selfies:
                        smiles = selfies.encoder(smiles)
                    filtered_sample[tokens_index] = smiles  # Potentially unsafe
            decoded_inputs[sample_index] = "".join(filtered_sample)
        return decoded_inputs


class PreTrainPolarsDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            tokenizer,
            max_length=1024,
            prefix="",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.lazy_frame_buckets = {}
        self.length = 0
        self.max_length = max_length
        print("building a dataset")
        for item in tqdm(os.listdir(data_dir)):
            lazy_frame = pr.scan_csv(os.path.join(data_dir, item), has_header=False)
            length = rawgencount(os.path.join(data_dir, item))
            self.length += length
            self.lazy_frame_buckets[self.length] = lazy_frame

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        offset = 0
        for key in self.lazy_frame_buckets:
            if idx < key:
                sample = str(self.prefix) + self.lazy_frame_buckets[key].slice(idx - offset, 1).collect().item()
                return self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True,
                                      max_length=self.max_length)
            else:
                offset = key
        raise IndexError("Index out of range")


class TextPolarsDataset(Dataset):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
            self,
            data_dir: str,
            type_path: str = 'train',
            has_header=False,
            scaler=None
    ) -> None:
        super().__init__()
        self.source = pr.read_csv(os.path.join(data_dir, type_path + ".source"), has_header=has_header)
        self.target = pd.read_csv(os.path.join(data_dir, type_path + ".target"), header=None)
        self.string = False  # REFACTOR
        if type(self.target.iloc[0].item()) == str:
            self.string = True
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.string:
            target = self.target.iloc[idx].to_numpy().item()
        else:
            if self.scaler:
                target = torch.from_numpy(self.target.iloc[idx].to_numpy()).unsqueeze(0)
                target = torch.from_numpy(self.scaler.transform(target).squeeze())
            else:
                target = torch.from_numpy(self.target.iloc[idx].to_numpy().squeeze())
        return self.source[idx].item(), target


class LineByLineTextDataset(Dataset):
    def __init__(
            self,
            file_path: str,
            block_size: int,
            prefix: str = '',
            tokenizer: PreTrainedTokenizerFast = None
    ) -> None:
        super().__init__()
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.prefix: str = prefix
        self._file_path: str = file_path
        self._len: int = rawgencount(file_path)
        self.max_length: int = block_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        line: str = deepcopy(linecache.getline(self._file_path, idx + 1).strip())
        return deepcopy(self.prefix + line[:self.max_length])

    def __len__(self) -> int:
        return self._len


class PropertyPretrainDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            prefix: str = '',
            type_path: str = "train",
            max_source_length: int = 300,
            scaler=None,
            tokenizer: PreTrainedTokenizerFast = None
    ) -> None:
        super().__init__()

        self.prefix: str = prefix
        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".hdf5")
        self.tokenizer: PreTrainedTokenizerFast = tokenizer  # TODO Depreceate
        self.max_source_len: int = max_source_length
        self.h5py_file = h5py.File(self._target_path, "r")
        self.targets = self.h5py_file['dataset']
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):  # TODO shift to collator
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        target_ids: torch.Tensor = torch.from_numpy(self.targets[idx].reshape(1, -1)).to(torch.FloatTensor())
        if self.scaler:
            target_ids = torch.from_numpy(self.scaler.transform(target_ids)).to(torch.FloatTensor())  # TODO Necessary?
        return source_line, target_ids.squeeze()

    def __del__(self):
        self.h5py_file.close()

    def sort_key(self, ex: BatchEncoding) -> int:  # TODO Depreciate this
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


class GOTermDataset(Dataset):
    def __init__(self, data_dir, mode, lookup, domain):
        self.data_dir = data_dir
        self.mode = mode
        self.domain = domain
        self.lookup = lookup
        self.ids = pd.read_csv(os.path.join(data_dir, mode + ".txt"), header=None)
        if mode == "val":
            mode = "train"
        self.sequences_dict = SeqIO.to_dict(SeqIO.parse(os.path.join(self.data_dir, mode + ".fasta"), "fasta"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids.iloc[idx].item()
        sequence = str(self.sequences_dict[id].seq)
        if self.domain != "all":
            label = self.lookup[id][self.domain]
        else:
            mf = self.lookup[id]["mf"]
            bp = self.lookup[id]["bp"]
            cc = self.lookup[id]["cc"]
            label = torch.cat((mf, bp, cc))
        return process_protein(sequence), label

class ESMGOTermDataset(GOTermDataset):
    def __init__(self, data_dir, mode, lookup, domain):
        super().__init__(data_dir, mode, lookup, domain)
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    def __getitem__(self, idx):
        sequence, label = super().__getitem__(idx)
        return sequence, sequence.replace("<P>", ""), label

class TextPolarsDataset(Dataset):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
            self,
            data_dir: str,
            type_path: str = 'train',
            has_header=False,
            scaler=None,
            scale_target=True,
    ) -> None:
        super().__init__()
        self.source = pr.read_csv(os.path.join(data_dir, type_path + ".source"), has_header=has_header)
        self.target = pd.read_csv(os.path.join(data_dir, type_path + ".target"), header=None)
        self.string = False  # REFACTOR
        if isinstance(self.target.iloc[0].item(), str):
            self.string = True
        self.scaler = scaler
        self.scale_target = scale_target

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        if self.string:
            target = self.target.iloc[idx].to_numpy().item()
        else:
            if self.scaler and self.scale_target:
                target = torch.from_numpy(self.target.iloc[idx].to_numpy()).unsqueeze(0)
                target = torch.from_numpy(self.scaler.transform(target).squeeze())
            else:
                target = torch.tensor(self.target.iloc[idx].to_numpy().squeeze(), dtype=torch.float)
        return self.source[idx].item() + "</s>", target


class ESMPolarsDataset(TextPolarsDataset):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
            self,
            data_dir: str,
            type_path: str = 'train',
            has_header=False,
            scaler=None
    ) -> None:
        super().__init__(data_dir, type_path, has_header, scaler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_molecule, target = super().__getitem__(idx)
        molecule = protein_molecule[protein_molecule.rindex("<P>") + 4:]
        protein = protein_molecule[: protein_molecule.rindex("<P>") + 4]
        return molecule, protein.replace("<P>", ""), target
class AltPolarsDataset(TextPolarsDataset):
    def __init__(
            self,
            data_dir: str,
            type_path: str = 'train',
            has_header=False,
            scaler=None
    ) -> None:
        super().__init__(data_dir, type_path, has_header, scaler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_molecule, target = super().__getitem__(idx)
        molecule = protein_molecule[protein_molecule.rindex("<P>") + 4:]
        protein = protein_molecule[: protein_molecule.rindex("<P>") + 4]
        return molecule, protein, target

# Credit https://stackoverflow.com/a/27518377/20522929
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)


def pad_complex_sequence(batch_inputs, batch_labels, input_length, target_length, pad_token_id,
                         concat_downstream=False):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=pad_token_id)
    if concat_downstream:
        input_pad_size = input_length - padded_inputs.shape[1]
        labels_pad_size = target_length - padded_labels.shape[1]
        padded_inputs = torch.nn.functional.pad(padded_inputs, (0, input_pad_size, 0, 0), value=pad_token_id)
        padded_labels = torch.nn.functional.pad(padded_labels, (0, labels_pad_size, 0, 0), value=pad_token_id)
    return padded_inputs, padded_labels
