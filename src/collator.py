import re
from typing import List

import numpy as np
import selfies
import torch
from rdkit.Chem import MolFromSequence, MolToSmiles
from transformers import BatchEncoding, AutoTokenizer, PreTrainedTokenizerFast

# In need of refactor

class BaseCollator:
    def __init__(self, tokenizer, prefix="", return_decoder_input_id=True, max_size=1024):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.return_decoder_input_id = return_decoder_input_id
        self.max_size = max_size

    def __call__(self, examples):
        batch = self.tokenizer([self.prefix + example[0] for example in examples], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_size)
        batch_size = batch.input_ids.size(0)
        if "token_type_ids" in batch.keys():
            del batch["token_type_ids"]
        if self.return_decoder_input_id:
            batch["decoder_input_ids"] = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        batch["labels"] = torch.stack([example[1] for example in examples])
        return batch

class PosCollator(BaseCollator):
    def __init__(self, tokenizer, prefix="", return_decoder_input_id=True, max_size=1024):
        super().__init__(tokenizer, prefix, return_decoder_input_id, max_size)

    def __call__(self, examples):
        batch = super().__call__(examples)
        protein_lengths = self.find_first_out_of_range(batch["input_ids"])
        batch["protein_length"] = protein_lengths
        return batch


    def find_first_out_of_range(self, batch, lower_bound=82, upper_bound=101):
        out_of_range_mask = (batch < lower_bound) | (batch > upper_bound)
        indices = torch.argmax(out_of_range_mask.int(), dim=1)
        return indices






class AnotherCollator:
    def __init__(self):
        pass

    def __call__(self, examples):
        t5chem_inputs = [example["t5chem"]["input_ids"].squeeze() for example in examples]
        t5chem_attention = [example["t5chem"]["attention_mask"].squeeze() for example in examples]
        esm_inputs = [example["esm"]["input_ids"].squeeze() for example in examples]
        esm_attention = [example["esm"]["attention_mask"].squeeze() for example in examples]
        # Pad t5chem_inputs
        t5chem_inputs = torch.nn.utils.rnn.pad_sequence(t5chem_inputs, batch_first=True, padding_value=3)
        t5chem_attention = torch.nn.utils.rnn.pad_sequence(t5chem_attention, batch_first=True, padding_value=0)

        # Pad esm_inputs
        esm_inputs = torch.nn.utils.rnn.pad_sequence(esm_inputs, batch_first=True, padding_value=1)
        esm_attention = torch.nn.utils.rnn.pad_sequence(esm_attention, batch_first=True, padding_value=0)

        #t5chem_inputs = torch.stack(t5chem_inputs)
        #t5chem_attention = torch.stack(t5chem_attention)
        #esm_inputs = torch.stack(esm_inputs)
        #esm_attention = torch.stack(esm_attention)
        batch = {"t5chem" : { "input_ids": t5chem_inputs, "attention_mask": t5chem_attention}, "esm":{ "input_ids": esm_inputs, "attention_mask": esm_attention}}
        return batch



class ESMCollator(BaseCollator):
    def __init__(self, tokenizer, prefix="", return_decoder_input_id=True, max_size=1024):
        super().__init__(tokenizer, prefix, return_decoder_input_id, max_size)
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    def __call__(self, examples):
        batch = self.tokenizer([self.prefix + example[0] for example in examples], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_size)
        protein_batch = self.esm_tokenizer([example[1] for example in examples], return_tensors="pt", padding=True,
                                           truncation=True, max_length=self.max_size)
        if "token_type_ids" in batch.keys():
            del batch["token_type_ids"]
        batch["labels"] = torch.stack([example[2] for example in examples])
        batch["esm_input_ids"] = protein_batch["input_ids"]
        batch["esm_attention_mask"] = protein_batch["attention_mask"]
        return BatchEncoding(batch)
class AltCollator(BaseCollator):
    def __init__(self, tokenizer, prefix="", return_decoder_input_id=True, max_size=1024):
        super().__init__(tokenizer, prefix, return_decoder_input_id, max_size)

    def __call__(self, examples):
        batch = self.tokenizer([self.prefix + example[0] for example in examples], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_size)
        protein_batch = self.tokenizer([example[1] for example in examples], return_tensors="pt", padding=True,
                                           truncation=True, max_length=self.max_size)
        if "token_type_ids" in batch.keys():
            del batch["token_type_ids"]
        batch["labels"] = torch.stack([example[2] for example in examples])
        batch["decoder_input_ids"] = protein_batch["input_ids"]
        batch["decoder_attention_mask"] = protein_batch["attention_mask"]
        return BatchEncoding(batch)


class BaseStringCollator():
    def __init__(self, tokenizer, prefix="", max_length=600):
        self.tokenizer = tokenizer
        if isinstance(prefix, dict):
            self.prefix = list(prefix.keys())[0] + ":"  # hacky, I dont like this
        else:
            self.prefix = prefix
        self.max_length = max_length

    def __call__(self, examples):
        batch = self.tokenizer([self.prefix + example[0] for example in examples], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_length)
        labels = self.tokenizer([example[1] + "</s>" for example in examples], return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_length)["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        if "token_type_ids" in batch.keys():
            del batch["token_type_ids"]
        return BatchEncoding(batch)


"""
TODO build a multitask collator, especially seeing how close the chem Mask Tokenizer is
class MultiTaskCollator:
"""


class NewChemMaskCollator:
    def __init__(self, max_length=1024, pad_token_id=3, targets=False):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.targets = targets

    def __call__(self, examples):
        batch_input_ids = [example["input_ids"].squeeze() for example in examples]
        # Pad inputs ids
        batch_labels = [example["labels"].squeeze() for example in examples]
        padded_inputs, padded_labels = pad_complex_sequence(batch_input_ids, batch_labels, self.max_length,
                                                            self.max_length, self.pad_token_id)
        padded_labels[padded_labels == self.pad_token_id] = -100
        attention_mask = (padded_inputs != self.pad_token_id).long()
        if self.targets:
            targets = torch.tensor([example["target"] for example in examples])
            target_ones = targets.unsqueeze(1) * torch.ones(padded_inputs.shape[0],padded_inputs.shape[1]) 
            processed = BatchEncoding({"input_ids": padded_inputs, "attention_mask": attention_mask,
                                   "labels": padded_labels, "targets": target_ones})
        else:
            processed = BatchEncoding({"input_ids": padded_inputs, "attention_mask": attention_mask,
                                   "labels": padded_labels})  # "decoder_input_ids": padded_labels})
        return processed

class ComplexESMCollator(NewChemMaskCollator):
    def __init__(self, max_length=1024, pad_token_id=3):
        super().__init__(max_length, pad_token_id, True)
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        vocab = "/scratch/tk2801/LightningT5/src/vocab/style2.json"
        self.base_tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab, bos_token="<pad>", eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)

    def __call__(self, examples):
        processed = super().__call__(examples)
        proteins = self.base_tokenizer.batch_decode(processed["input_ids"][:, 1:-1])
        # Replace all <extra_id_#s> with <mask>
        processed_proteins = []
        for protein in proteins:
            protein = re.sub(r"<extra_id_\d+>", "<mask>", protein)
            protein = protein.replace("<P>", "").replace("</s>", "<eos>")
            processed_proteins.append(protein)
        protein_batch = self.esm_tokenizer(processed_proteins, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length+2)
        return {"t5chem": processed, "esm": protein_batch}




class ChemMaskCollator:
    def __init__(self, tokenizer, prefix="", mlm_probability=.15, mean_noise_span_length=3, target_length=1024,
                 input_length=1024, *, add_prefix=False, selfies=False):
        self.tokenizer = tokenizer
        self.CHEM_MASK = "Chem-Mask:"
        if add_prefix:
            self.prefix = self.CHEM_MASK if prefix == "" else self.prefix
        else:
            self.prefix = prefix
        if isinstance(prefix, dict):
            self.prefix = list(prefix.keys())[0] + ":"
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.target_length = target_length  # TODO switch to handle config
        self.input_length = input_length  # TODO switch to handle config
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.bos_token_id
        self.regex = re.compile(fr'(<[^>]+>)|\b{self.CHEM_MASK}|{prefix}\b')
        self.selfies = selfies

    def __call__(self, examples: List[str]) -> BatchEncoding:
        batch_inputs = []
        batch_labels = []
        if isinstance(examples, str):
            examples = [examples]
        for example in examples:
            batch = self.tokenizer(self.prefix + example, return_tensors="pt", padding=True, truncation=True,
                                   max_length=self.input_length)
            input_ids = batch["input_ids"]
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
                if tokens.startswith("<") or self.CHEM_MASK == tokens or self.prefix == tokens:
                    pass
                else:
                    smiles = MolToSmiles(
                        MolFromSequence(tokens))  # RDKIT needs refactored. # TODO Implement a cache here.
                    if self.selfies:
                        smiles = selfies.encoder(smiles)
                    filtered_sample[tokens_index] = smiles  # Potentially unsafe
            decoded_inputs[sample_index] = "".join(filtered_sample)
        return decoded_inputs


# Pretty much copied from FlaxSpanMask from HF
class SpanCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, mean_noise_span_length=3, input_length=512, target_length=512,
                 prefix=""):
        # self.tokenizer = tokenizer
        self.tokenizer_length = len(tokenizer)
        self.mlm_probability = mlm_probability
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length  # TODO switch to handle config
        self.target_length = target_length  # TODO switch to handle config
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.prefix = prefix

    def __call__(self, examples: List[str]) -> BatchEncoding:
        batch_inputs = []
        batch_labels = []
        batch_decoder = []
        if isinstance(examples, str):
            examples = [examples]
        for example in examples:
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
