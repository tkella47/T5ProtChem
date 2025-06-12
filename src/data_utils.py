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
        self.env = lmdb.open(
            self.data_dir, readonly=True, lock=False, readahead=False, meminit=False
        )

    def __len__(self):
        if self.env is None:
            self._init_db()
        if self.teardown:
            self.env.close()
            self.env = None
        return self.env.stat()["entries"]

    def __getitem__(self, idx, selection_function=None):
        if self.env is None:
            self._init_db()
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
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
        return self.tokenizer(
            self.prefix + data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )


class EsmAlignDataset(LMDB):
    def __init__(
        self, data_dir, tokenizer_t5chem, max_length=1024, prefix="", teardown=False
    ):
        super().__init__(data_dir)
        self.data_dir = data_dir
        self.env = None
        self.tokenizer_t5chem = tokenizer_t5chem
        self.tokenizer_esm = AutoTokenizer.from_pretrained(
            "facebook/esm2_t30_150M_UR50D"
        )
        self.max_length = max_length
        self.prefix = prefix
        self.teardown = teardown

    def __getitem(self, idx):
        data = super().__getitem__(idx)
        t5chem_inputs = self.tokenizer_t5chem(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )
        data = data.replace("<P>", "")
        esm_inputs = self.tokenizer_esm(
            data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {"t5chem": t5chem_inputs, "esm": esm_inputs}


class SpanMaskDataset(LMDBTokenizer):
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_length=1024,
        prefix="Span-Mask:",
        teardown=False,
        mlm_probability=0.15,
        mean_noise_span_length=3,
    ):
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
        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for _ in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch_inputs.append(
            torch.from_numpy(
                self.filter_input_ids(input_ids, input_ids_sentinel)
            ).flatten()
        )
        labels = self.filter_input_ids(input_ids, labels_sentinel)
        batch_labels.append(torch.from_numpy(labels).flatten())

        padded_inputs, padded_labels = pad_complex_sequence(
            batch_inputs,
            batch_labels,
            self.input_length,
            self.input_length,
            self.pad_token_id,
        )
        # attention mask
        padded_labels[padded_labels == self.pad_token_id] = -100
        attention_mask = (padded_inputs != self.pad_token_id).long()
        processed = BatchEncoding(
            {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": padded_labels,
            }
        )  # "decoder_input_ids": padded_labels})

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
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
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

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (self.tokenizer_length - sentinel_ids), 0
        )
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
            [input_ids, np.full((batch_size, 1), self.eos_token_id, dtype=np.int32)],
            axis=-1,
        )
        return input_ids


class ChemMaskDataset(LMDBTokenizer):
    """
    A PyTorch Dataset for preparing chemical data (e.g., amino acid sequences) for a T5-style
    span masking pre-training task. The key characteristic of this dataset is that the
    masked portions (labels) are converted into SMILES strings and then tokenized.

    The process involves:
    1. Fetching a tokenized sequence from an LMDB database.
    2. Applying span-based noise/masking to the sequence.
    3. Identifying the original tokens in the masked spans (initially as amino acid tokens).
    4. Converting these amino acid tokens back to their string representation.
    5. Transforming these amino acid strings into SMILES strings using RDKit.
    6. Optionally converting SMILES strings to SELFIES strings.
    7. Tokenizing the final SMILES/SELFIES strings to serve as labels for the model.
    8. Padding both the corrupted input and the final labels.
    """

    def __init__(
        self,
        data_dir,
        tokenizer,
        max_length=1024,
        fragment_length=None,
        prefix="Chem-Mask:",
        teardown=False,
        mlm_probability=0.15,
        mean_noise_span_length=3,
    ):
        super().__init__(data_dir, tokenizer, max_length, prefix, teardown)
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.target_length = max_length  # Max length for the target (labels)
        self.input_length = max_length  # Max length for the input

        # Regex to split sequences by special tokens (e.g., <P>) or the specified prefix.
        # This helps isolate actual chemical sequence parts for conversion.
        self.regex = re.compile(rf"(<[^>]+>)|\b{re.escape(prefix)}\b")

        self.pad_token_id = tokenizer.pad_token_id
        # self.decoder_start_token_id = tokenizer.bos_token_id # Not explicitly used in __getitem__ logic shown
        self.selfies = (
            False  # If True, converts SMILES to SELFIES. Not configurable by default.
        )

        self.partial_selection_function = None
        if fragment_length is not None:
            self.partial_selection_function = partial(
                self.selection_function, fragment_length=fragment_length
            )

    def __len__(self):
        return super().__len__()

    @staticmethod
    def selection_function(protein_sequence: str, fragment_length: int) -> str:
        """
        Selects a random fragment of a given length from a protein sequence.
        Protein sequences are expected to be space-separated or otherwise processable after removing "<P>".
        """
        protein_sequence_content = protein_sequence.replace("<P>", "")
        if len(protein_sequence_content) <= fragment_length:
            return protein_sequence  # Return original if shorter or equal to fragment_length

        max_start_index = len(protein_sequence_content) - fragment_length
        fragment_start_index = random.randint(0, max_start_index)

        fragment = protein_sequence_content[
            fragment_start_index : fragment_start_index + fragment_length
        ]
        # Re-add the <P> tags if that's the expected format (based on original code context)
        return "<P>" + "<P>".join(
            list(fragment)
        )  # Assuming original format was like <P>A<P>C<P>G...

    def _apply_span_masking(
        self, original_input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies T5-style span masking to the input token IDs.

        Args:
            original_input_ids: Tensor of shape (1, sequence_length) containing the original tokenized input.

        Returns:
            A tuple containing:
            - processed_input_ids: Tensor of shape (1, new_sequence_length) with mask sentinels.
            - aa_label_ids: Tensor of shape (1, new_label_sequence_length) containing the
                            original token IDs for the masked spans (still as AA tokens).
        """
        batch_size, expanded_input_length = original_input_ids.shape

        # Generate mask_indices for each item in the batch (here, batch_size is 1)
        # The random_spans_noise_mask method itself is defined per-example.
        mask_indices_np = np.array(
            [
                self.random_spans_noise_mask(expanded_input_length)
                for _ in range(batch_size)
            ]
        )

        labels_mask_np = ~mask_indices_np

        input_ids_sentinel_np = self.create_sentinel_ids(
            mask_indices_np.astype(np.int8)
        )
        labels_sentinel_np = self.create_sentinel_ids(labels_mask_np.astype(np.int8))

        # filter_input_ids expects numpy arrays and returns numpy arrays
        processed_input_ids_np = self.filter_input_ids(
            original_input_ids.cpu().numpy(), input_ids_sentinel_np
        )
        aa_label_ids_np = self.filter_input_ids(
            original_input_ids.cpu().numpy(), labels_sentinel_np
        )

        device = original_input_ids.device
        return torch.from_numpy(processed_input_ids_np).to(device), torch.from_numpy(
            aa_label_ids_np
        ).to(device)

    def _convert_aa_labels_to_smiles_labels(
        self, aa_label_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts amino acid token ID labels to SMILES strings and then re-tokenizes them.

        Args:
            aa_label_ids: Tensor of shape (1, sequence_length) containing AA token IDs for masked spans.

        Returns:
            Tensor of shape (1, new_sequence_length) containing tokenized SMILES strings.
        """
        # Decode the amino acid token IDs to strings.
        # skip_special_tokens=False is important as swap_aa_for_smiles uses them for splitting.
        decoded_aa_sequences = self.tokenizer.batch_decode(
            aa_label_ids, skip_special_tokens=False
        )

        # Convert these AA strings to SMILES strings.
        smiles_strings = self.swap_aa_for_smiles(decoded_aa_sequences)

        # Re-tokenize the SMILES strings to get the final labels.
        # Padding is handled later; truncation is important if SMILES are too long.
        tokenized_smiles = self.tokenizer(
            smiles_strings,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.target_length,
        )
        return tokenized_smiles["input_ids"].to(aa_label_ids.device)

    def _pad_tensor(
        self, tensor: torch.Tensor, max_length: int, pad_value: int
    ) -> torch.Tensor:
        """Pads or truncates a 1D tensor to a specified maximum length."""
        seq_len = tensor.shape[0]
        if seq_len < max_length:
            padding_size = max_length - seq_len
            return torch.nn.functional.pad(tensor, (0, padding_size), value=pad_value)
        elif seq_len > max_length:
            return tensor[:max_length]
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetches an item, applies span masking, converts masked labels to SMILES,
        tokenizes SMILES, and pads inputs/labels.
        """
        # 1. Fetch initial tokenized item (includes prefix from parent class)
        # original_input_ids shape: (1, sequence_length)
        tokenized_item = super().__getitem__(idx, self.partial_selection_function)
        original_input_ids = tokenized_item["input_ids"]

        # 2. Apply span masking
        # processed_input_ids shape: (1, new_input_seq_len)
        # aa_label_ids shape: (1, new_aa_label_seq_len)
        processed_input_ids, aa_label_ids = self._apply_span_masking(original_input_ids)

        # 3. Convert AA token ID labels to SMILES strings and re-tokenize
        # final_smiles_label_ids shape: (1, new_smiles_label_seq_len)
        final_smiles_label_ids = self._convert_aa_labels_to_smiles_labels(aa_label_ids)

        # 4. Pad inputs and final labels (squeeze to 1D for padding, then unsqueeze for batch dim)
        # The original used pad_complex_sequence with flattened lists, this is more direct for a single item.
        padded_input_ids = self._pad_tensor(
            processed_input_ids.squeeze(0), self.input_length, self.pad_token_id
        )
        padded_label_ids = self._pad_tensor(
            final_smiles_label_ids.squeeze(0), self.target_length, self.pad_token_id
        )

        # 5. Prepare final BatchEncoding
        # Mark padding tokens in labels with -100 for loss functions to ignore
        padded_label_ids[padded_label_ids == self.pad_token_id] = -100
        attention_mask = (padded_input_ids != self.pad_token_id).long()

        return BatchEncoding(
            {
                "input_ids": padded_input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "labels": padded_label_ids.unsqueeze(0),
            }
        )

    # Note: random_spans_noise_mask, create_sentinel_ids, and filter_input_ids
    # are identical to those in SpanMaskDataset. For a larger refactoring,
    # these could be moved to a shared utility module or base class.
    def random_spans_noise_mask(self, length: int) -> np.ndarray:
        """Generates a T5-style random span noise mask.
        Identical to the version in SpanMaskDataset.
        """
        orig_length = length
        num_noise_tokens = int(np.round(length * self.mlm_probability))
        # Avoid degeneracy by ensuring positive numbers of noise and non-noise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))
        # Avoid degeneracy by ensuring positive number of noise spans.
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # Pick the lengths of the noise spans and the non-noise spans.
        def _random_segmentation(num_items: int, num_segments: int) -> np.ndarray:
            """Partition a sequence of items randomly into N segments."""
            if (
                num_items < num_segments
            ):  # Handle cases where items are fewer than segments
                # Fallback: create num_items segments of length 1, and num_segments - num_items segments of length 0
                lengths = np.zeros(num_segments, dtype=np.int32)
                lengths[:num_items] = 1
                np.random.shuffle(lengths)  # Distribute the lengths randomly
                return lengths

            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(
                mask_indices, [[1, 0]]
            )  # Mark the start of each segment.
            segment_id = np.cumsum(first_in_segment)
            # Count length of sub-segments assuming that list is sorted.
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )
        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
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

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def filter_input_ids(
        self, input_ids: np.ndarray, sentinel_ids: np.ndarray
    ) -> np.ndarray:
        """
        Filters input_ids using sentinel_ids, merging consecutive mask tokens.
        This reduces sequence length.
        Identical to the version in SpanMaskDataset.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # Tokens >= 0 are preserved (input_ids or sentinel_ids).
        # Tokens < 0 (from sentinel_ids logic for consecutive masks) are removed.
        # This reshapes to (batch_size, -1) by effectively concatenating non-negative tokens per row.
        input_ids_filtered = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1)
        )

        # Append EOS token to each sequence in the batch.
        eos_tokens = np.full(
            (batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32
        )
        input_ids_final = np.concatenate([input_ids_filtered, eos_tokens], axis=-1)

        return input_ids_final

    def swap_aa_for_smiles(self, decoded_aa_sequences: list[str]) -> list[str]:
        """
        Converts amino acid (AA) sequences within a list of decoded strings to SMILES strings.

        The method processes each string in the input list:
        1. Removes spaces and "<P>" tags.
        2. Splits the string by special tokens (defined by `self.regex`) or the dataset prefix.
           This isolates segments that are purely AA sequences.
        3. For each AA segment, it attempts to convert it to a SMILES string using RDKit.
           - If `self.selfies` is True, the SMILES string is further converted to a SELFIES string.
        4. Reconstructs the string with AA segments replaced by their SMILES/SELFIES counterparts.

        Args:
            decoded_aa_sequences: A list of strings, where each string may contain AA sequences
                                  interspersed with special tokens. These strings are typically
                                  the output of `tokenizer.batch_decode()`.

        Returns:
            A list of strings, with AA sequences converted to SMILES/SELFIES.

        Raises:
            Exception: If RDKit fails to convert an AA segment to SMILES, indicating an issue
                       with the input AA segment or RDKit's interpretation.
        """
        processed_sequences = []
        for aa_sequence_str in decoded_aa_sequences:
            # Clean up common sequence formatting issues.
            cleaned_sequence_str = aa_sequence_str.replace(" ", "").replace("<P>", "")

            # Split the string by special tokens or the prefix to isolate AA segments.
            # filter(None, ...) removes empty strings that can result from splitting.
            parts = list(filter(None, self.regex.split(cleaned_sequence_str)))

            converted_parts = []
            for token_segment in parts:
                # Check if the segment is a special token or the prefix itself.
                if token_segment.startswith("<") or self.prefix == token_segment:
                    converted_parts.append(token_segment)
                else:
                    # This segment is assumed to be an AA sequence.
                    try:
                        # Convert AA sequence to RDKit Mol object, then to SMILES.
                        smiles = MolToSmiles(MolFromSequence(token_segment))
                    except Exception as e:
                        # TODO: Consider more specific error handling or logging.
                        # The original code printed and then raised a generic Exception.
                        # This indicates a potential issue with the input `token_segment`
                        # or RDKit's ability to parse it.
                        print(
                            f"Error converting AA segment to SMILES: '{token_segment}'. Error: {e}"
                        )
                        raise Exception(
                            f"Offending AA tokens for SMILES conversion: {token_segment}"
                        ) from e

                    if self.selfies:
                        # TODO: Ensure `selfies.encoder` is available if this path is taken.
                        smiles = selfies.encoder(smiles)
                    converted_parts.append(smiles)
            processed_sequences.append("".join(converted_parts))
        return processed_sequences


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
                sample = (
                    str(self.prefix)
                    + self.lazy_frame_buckets[key]
                    .slice(idx - offset, 1)
                    .collect()
                    .item()
                )
                return self.tokenizer(
                    sample,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                offset = key
        raise IndexError("Index out of range")


class TextPolarsDataset(
    Dataset
):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
        self, data_dir: str, type_path: str = "train", has_header=False, scaler=None
    ) -> None:
        super().__init__()
        self.source = pr.read_csv(
            os.path.join(data_dir, type_path + ".source"), has_header=has_header
        )
        self.target = pd.read_csv(
            os.path.join(data_dir, type_path + ".target"), header=None
        )
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
        prefix: str = "",
        tokenizer: PreTrainedTokenizerFast = None,
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
        return deepcopy(self.prefix + line[: self.max_length])

    def __len__(self) -> int:
        return self._len


class PropertyPretrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        prefix: str = "",
        type_path: str = "train",
        max_source_length: int = 300,
        scaler=None,
        tokenizer: PreTrainedTokenizerFast = None,
    ) -> None:
        super().__init__()

        self.prefix: str = prefix
        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".hdf5")
        self.tokenizer: PreTrainedTokenizerFast = tokenizer  # TODO Depreceate
        self.max_source_len: int = max_source_length
        self.h5py_file = h5py.File(self._target_path, "r")
        self.targets = self.h5py_file["dataset"]
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):  # TODO shift to collator
        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        target_ids: torch.Tensor = torch.from_numpy(
            self.targets[idx].reshape(1, -1)
        ).to(torch.FloatTensor())
        if self.scaler:
            target_ids = torch.from_numpy(self.scaler.transform(target_ids)).to(
                torch.FloatTensor()
            )  # TODO Necessary?
        return source_line, target_ids.squeeze()

    def __del__(self):
        self.h5py_file.close()

    def sort_key(self, ex: BatchEncoding) -> int:  # TODO Depreciate this
        """Sort using length of source sentences."""
        return len(ex["input_ids"])


class GOTermDataset(Dataset):
    def __init__(self, data_dir, mode, lookup, domain):
        self.data_dir = data_dir
        self.mode = mode
        self.domain = domain
        self.lookup = lookup
        self.ids = pd.read_csv(os.path.join(data_dir, mode + ".txt"), header=None)
        if mode == "val":
            mode = "train"
        self.sequences_dict = SeqIO.to_dict(
            SeqIO.parse(os.path.join(self.data_dir, mode + ".fasta"), "fasta")
        )

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
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/esm2_t30_150M_UR50D"
        )

    def __getitem__(self, idx):
        sequence, label = super().__getitem__(idx)
        return sequence, sequence.replace("<P>", ""), label


class TextPolarsDataset(
    Dataset
):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
        self,
        data_dir: str,
        type_path: str = "train",
        has_header=False,
        scaler=None,
        scale_target=True,
    ) -> None:
        super().__init__()
        self.source = pr.read_csv(
            os.path.join(data_dir, type_path + ".source"), has_header=has_header
        )
        self.target = pd.read_csv(
            os.path.join(data_dir, type_path + ".target"), header=None
        )
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
                target = torch.tensor(
                    self.target.iloc[idx].to_numpy().squeeze(), dtype=torch.float
                )
        return self.source[idx].item() + "</s>", target


class ESMPolarsDataset(
    TextPolarsDataset
):  # Does it make sense to have TextPolarsDataset have floats?
    def __init__(
        self, data_dir: str, type_path: str = "train", has_header=False, scaler=None
    ) -> None:
        super().__init__(data_dir, type_path, has_header, scaler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_molecule, target = super().__getitem__(idx)
        molecule = protein_molecule[protein_molecule.rindex("<P>") + 4 :]
        protein = protein_molecule[: protein_molecule.rindex("<P>") + 4]
        return molecule, protein.replace("<P>", ""), target


class AltPolarsDataset(TextPolarsDataset):
    def __init__(
        self, data_dir: str, type_path: str = "train", has_header=False, scaler=None
    ) -> None:
        super().__init__(data_dir, type_path, has_header, scaler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_molecule, target = super().__getitem__(idx)
        molecule = protein_molecule[protein_molecule.rindex("<P>") + 4 :]
        protein = protein_molecule[: protein_molecule.rindex("<P>") + 4]
        return molecule, protein, target


# Credit https://stackoverflow.com/a/27518377/20522929
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def rawgencount(filename):
    f = open(filename, "rb")
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b"\n") for buf in f_gen)


def pad_complex_sequence(
    batch_inputs,
    batch_labels,
    input_length,
    target_length,
    pad_token_id,
    concat_downstream=False,
):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        batch_inputs, batch_first=True, padding_value=pad_token_id
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        batch_labels, batch_first=True, padding_value=pad_token_id
    )
    if concat_downstream:
        input_pad_size = input_length - padded_inputs.shape[1]
        labels_pad_size = target_length - padded_labels.shape[1]
        padded_inputs = torch.nn.functional.pad(
            padded_inputs, (0, input_pad_size, 0, 0), value=pad_token_id
        )
        padded_labels = torch.nn.functional.pad(
            padded_labels, (0, labels_pad_size, 0, 0), value=pad_token_id
        )
    return padded_inputs, padded_labels
