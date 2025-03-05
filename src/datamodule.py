import json
import os
import warnings
from sys import platform
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import CombinedLoader
from torch.cuda import device_count
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from collator import SpanCollator, BaseCollator, BaseStringCollator, ChemMaskCollator, NewChemMaskCollator, ESMCollator, \
    AltCollator, PosCollator
from data_utils import LineByLineTextDataset, PropertyPretrainDataset, TextPolarsDataset, GOTermDataset, \
    ChemMaskDataset, SpanMaskDataset, ESMPolarsDataset, EsmAlignDataset, ESMGOTermDataset, AltPolarsDataset 

HYPERTHREADING_PATH = "/sys/devices/system/cpu/smt/active"


def get_workers():
    try:
        workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        print(f"slurm scheduler detected, using {workers} workers")
    except:
        workers = 0
        print("Number of workers was not specified and slurm was not detected, using 0")
    if platform == "linux":
        if bool(int(open(HYPERTHREADING_PATH).readlines()[0].strip())):  # HYPERTHREADING : 1, else 0
            workers = workers * 2
    return workers


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)




class BaseStringModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, vocab_file=None, prefix="", workers=None, max_length=600):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 # TODO rethink this initialization.
                                                 eos_token="</s>",
                                                 additional_special_tokens=["<mod>", "</mod>", "Fill-Mask:"],
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True)
        self.collator = BaseStringCollator(tokenizer=self.tokenizer, prefix=prefix, max_length=max_length)

    def get_tokenizer(self):
        return self.tokenizer

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = TextPolarsDataset(self.data_dir, type_path="train")
            self.val_dataset = TextPolarsDataset(self.data_dir, type_path="val")
        elif stage == "validate":
            self.val_dataset = TextPolarsDataset(self.data_dir, "val")
        elif stage == "test":
            self.test_dataset = TextPolarsDataset(self.data_dir, "test")
        else:
            raise ValueError("Stage must be fit, validate,or test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers,
                          collate_fn=self.collator, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.workers,
                          collate_fn=self.collator, shuffle=False, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.workers,
                          collate_fn=self.collator, shuffle=False, pin_memory=True)






class MolecularDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, vocab_file=None, workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 # TODO rethink this initialization.
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True)
        self.collator = SpanCollator(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.save_hyperparameters()

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.train_dataset = LineByLineTextDataset(file_path=os.path.join(self.data_dir, 'train.txt'),
                                                       block_size=1022,
                                                       prefix='Span-Mask:')
            self.val_dataset = LineByLineTextDataset(file_path=os.path.join(self.data_dir, 'val.txt'), block_size=1022,
                                                     prefix='Span-Mask:')
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, shuffle=True,
                          num_workers=self.workers,
                          pin_memory=bool(device_count()))  # TODO add parameters for num workers

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, shuffle=False,
                          num_workers=self.workers, pin_memory=bool(device_count()))


class MolecularDataModuleChemMask(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, vocab_file=None, workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.collator = ChemMaskCollator(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.save_hyperparameters()

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.train_dataset = LineByLineTextDataset(file_path=os.path.join(self.data_dir, 'train.txt'),
                                                       block_size=400,
                                                       prefix='Chem-Mask:')
            self.val_dataset = LineByLineTextDataset(file_path=os.path.join(self.data_dir, 'val.txt'), block_size=400,
                                                     prefix='Chem-Mask:')
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, shuffle=True,
                          num_workers=self.workers,
                          pin_memory=bool(device_count()))  # TODO add parameters for num workers

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, shuffle=False,
                          num_workers=self.workers, pin_memory=bool(device_count()))


class MolecularRegressionData(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, vocab_file, scaler_path=None, workers=None, scale_target=True,
                 hdf5=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 # TODO rethink this initialization. This will make it difficult to add new tokens. and manage the tokenizer if changes are needed.
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True)
        self.scaler_path = scaler_path
        self.collator = BaseCollator(tokenizer=self.tokenizer, prefix="")
        self.save_hyperparameters()
        print("data hparams")
        print(self.hparams)

    def get_tokenizer(self):
        return self.tokenizer

    def setup(self, stage: str) -> None:
        if self.scaler_path:
            scaler = joblib.load(self.scaler_path)
        else:
            scaler = None
        if stage == "fit":
            if self.hparams.hdf5:
                self.train_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                             type_path="train", scaler=scaler)
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.train_dataset = TextPolarsDataset(self.data_dir, type_path="train", has_header=False,
                                                       scaler=scaler, scale_target=self.hparams.scale_target)
                self.val_dataset = TextPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler,
                                                     scale_target=self.hparams.scale_target)
        elif stage == "validate":
            if self.hparams.hdf5:
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.val_dataset = TextPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler,
                                                     scale_target=self.hparams.scale_target)

        elif stage == "test":
            if self.hparams.hdf5:
                self.test_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                            type_path="test", scaler=scaler)
            else:
                self.test_dataset = TextPolarsDataset(self.data_dir, type_path="test", has_header=False, scaler=scaler,
                                                      scale_target=self.hparams.scale_target)
        else:
            raise NotImplementedError("Only fit stage is implemented")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))


class PosBinderData(MolecularRegressionData):
    def __init__(self, data_dir, batch_size, vocab_file, scaler_path=None, workers=None, scale_target=True,
                 hdf5=False):
        super().__init__(data_dir,batch_size, vocab_file, scaler_path, workers, scale_target, hdf5)
        self.collator = PosCollator(tokenizer=self.tokenizer, prefix="")



class JustChemMask(pl.LightningDataModule):  # Refactor the datamodule in to the constituents
    def __init__(self, data_dir_config, batch_size, vocab_file, chemmask=False, workers=None, **kwargs):
        super().__init__()
        with open(data_dir_config, "r") as f:
            self.data_dirs = json.load(f)
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.chemmask = chemmask
        self.max_length = kwargs.get("max_length", 1024)
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.chemmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.spanmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.save_hyperparameters()
        self.kwargs = kwargs
        if kwargs.get("fragment_length", None) is not None:
            if kwargs.get("fragment_length", None) > self.max_length:
                warnings.warn("Fragment length is greater than max length, following max length over fragment length")
        # self.fragment_length = kwargs.get("fragment_length", None)

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.chemmask_train_dataset = ChemMaskDataset(self.data_dirs["chemmask"], tokenizer=self.tokenizer,
                                                          prefix='Chem-Mask:', **self.kwargs)
            self.chem_val_dataset = SpanMaskDataset(self.data_dirs["chem_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
            self.chemmask_val_dataset = ChemMaskDataset(self.data_dirs["chemmask_val"], tokenizer=self.tokenizer,
                                                        prefix='Chem-Mask:', max_length=self.max_length)
            self.prot_val_dataset = SpanMaskDataset(self.data_dirs["prot_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        chemmask = DataLoader(self.chemmask_train_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=True,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()),
                              )

        iterables = {"chemmask": chemmask}
        train_combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        return train_combined_dataloader

    def val_dataloader(self):
        chem_span = DataLoader(self.chem_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        prot_span = DataLoader(self.prot_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        chemmask = DataLoader(self.chemmask_val_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=False,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()))
        iterables = {"chem_span": chem_span, "prot_span": prot_span, "chemmask": chemmask}
        val_combined_dataloader = CombinedLoader(iterables, mode="sequential")
        return val_combined_dataloader


class JustChem(pl.LightningDataModule):
    def __init__(self, data_dir_config, batch_size, vocab_file, chemmask=False, workers=None, **kwargs):
        super().__init__()
        with open(data_dir_config, "r") as f:
            self.data_dirs = json.load(f)
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.chemmask = chemmask
        self.max_length = kwargs.get("max_length", 1024)
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.chemmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.spanmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.save_hyperparameters()
        self.kwargs = kwargs
        if kwargs.get("fragment_length", None) is not None:
            if kwargs.get("fragment_length", None) > self.max_length:
                warnings.warn("Fragment length is greater than max length, following max length over fragment length")
        # self.fragment_length = kwargs.get("fragment_length", None)

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.chem_train_dataset = SpanMaskDataset(data_dir=self.data_dirs["train"], tokenizer=self.tokenizer,
                                                      prefix="Span-Mask:", max_length=self.max_length)
            self.chem_val_dataset = SpanMaskDataset(self.data_dirs["chem_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
            self.chemmask_val_dataset = ChemMaskDataset(self.data_dirs["chemmask_val"], tokenizer=self.tokenizer,
                                                        prefix='Chem-Mask:', max_length=self.max_length)
            self.prot_val_dataset = SpanMaskDataset(self.data_dirs["prot_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        prot_chem_span = DataLoader(self.chem_train_dataset, batch_size=self.batch_size,
                                    collate_fn=self.spanmask_collator, shuffle=True,
                                    num_workers=self.workers,
                                    pin_memory=bool(device_count()),
                                    )

        iterables = {"prot_chem_span": prot_chem_span}
        train_combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        return train_combined_dataloader

    def val_dataloader(self):
        chem_span = DataLoader(self.chem_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        prot_span = DataLoader(self.prot_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        chemmask = DataLoader(self.chemmask_val_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=False,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()))
        iterables = {"chem_span": chem_span, "prot_span": prot_span, "chemmask": chemmask}
        val_combined_dataloader = CombinedLoader(iterables, mode="sequential")
        return val_combined_dataloader


class JustProt(pl.LightningDataModule):
    def __init__(self, data_dir_config, batch_size, vocab_file, chemmask=False, workers=None, **kwargs):
        super().__init__()
        with open(data_dir_config, "r") as f:
            self.data_dirs = json.load(f)
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.chemmask = chemmask
        self.max_length = kwargs.get("max_length", 1024)
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.chemmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.spanmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.save_hyperparameters()
        self.kwargs = kwargs
        if kwargs.get("fragment_length", None) is not None:
            if kwargs.get("fragment_length", None) > self.max_length:
                warnings.warn("Fragment length is greater than max length, following max length over fragment length")
        # self.fragment_length = kwargs.get("fragment_length", None)

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.prot_train_dataset = SpanMaskDataset(data_dir=self.data_dirs["train"], tokenizer=self.tokenizer,
                                                      prefix="Span-Mask:", max_length=self.max_length)
            self.chem_val_dataset = SpanMaskDataset(self.data_dirs["chem_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
            self.chemmask_val_dataset = ChemMaskDataset(self.data_dirs["chemmask_val"], tokenizer=self.tokenizer,
                                                        prefix='Chem-Mask:', max_length=self.max_length)
            self.prot_val_dataset = SpanMaskDataset(self.data_dirs["prot_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        prot_chem_span = DataLoader(self.prot_train_dataset, batch_size=self.batch_size,
                                    collate_fn=self.spanmask_collator, shuffle=True,
                                    num_workers=self.workers,
                                    pin_memory=bool(device_count()),
                                    )
        iterables = {"prot_chem_span": prot_chem_span}
        train_combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        return train_combined_dataloader

    def val_dataloader(self):
        chem_span = DataLoader(self.chem_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        prot_span = DataLoader(self.prot_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        chemmask = DataLoader(self.chemmask_val_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=False,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()))
        iterables = {"chem_span": chem_span, "prot_span": prot_span, "chemmask": chemmask}
        val_combined_dataloader = CombinedLoader(iterables, mode="sequential")
        return val_combined_dataloader


class ChemProt(pl.LightningDataModule):  # Refactor the datamodule in to the constituents
    def __init__(self, data_dir_config, batch_size, vocab_file, chemmask=False, workers=None, **kwargs):
        super().__init__()
        with open(data_dir_config, "r") as f:
            self.data_dirs = json.load(f)
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.chemmask = chemmask
        self.max_length = kwargs.get("max_length", 1024)
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.chemmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.spanmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.save_hyperparameters()
        self.kwargs = kwargs
        if kwargs.get("fragment_length", None) is not None:
            if kwargs.get("fragment_length", None) > self.max_length:
                warnings.warn("Fragment length is greater than max length, following max length over fragment length")
        # self.fragment_length = kwargs.get("fragment_length", None)

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.chem_train_dataset = SpanMaskDataset(data_dir=self.data_dirs["chem_train"], tokenizer=self.tokenizer,
                                                      prefix="Span-Mask:", max_length=self.max_length)
            self.prot_train_dataset = SpanMaskDataset(data_dir=self.data_dirs["prot_train"], tokenizer=self.tokenizer,
                                                      prefix="Span-Mask:", max_length=self.max_length)
            self.chem_val_dataset = SpanMaskDataset(self.data_dirs["chem_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
            self.chemmask_val_dataset = ChemMaskDataset(self.data_dirs["chemmask_val"], tokenizer=self.tokenizer,
                                                        prefix='Chem-Mask:', max_length=self.max_length)
            self.prot_val_dataset = SpanMaskDataset(self.data_dirs["prot_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        chem_span = DataLoader(self.chem_train_dataset, batch_size=self.batch_size,
                               collate_fn=self.spanmask_collator, shuffle=True,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()),
                               )
        prot_span = DataLoader(self.prot_train_dataset, batch_size=self.batch_size,
                               collate_fn=self.chemmask_collator, shuffle=True,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()),
                               )

        iterables = {"chem_span": chem_span, "prot_span": prot_span}
        train_combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        return train_combined_dataloader

    def val_dataloader(self):
        chem_span = DataLoader(self.chem_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        prot_span = DataLoader(self.prot_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        chemmask = DataLoader(self.chemmask_val_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=False,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()))
        iterables = {"chem_span": chem_span, "prot_span": prot_span, "chemmask": chemmask}
        val_combined_dataloader = CombinedLoader(iterables, mode="sequential")
        return val_combined_dataloader


class BigPreTrainTest(pl.LightningDataModule):  # Refactor the datamodule in to the constituents
    def __init__(self, data_dir_config, batch_size, vocab_file, chemmask=True, workers=None, **kwargs):
        super().__init__()
        with open(data_dir_config, "r") as f:
            self.data_dirs = json.load(f)
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.chemmask = chemmask
        self.max_length = kwargs.get("max_length", 1024)
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.chemmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.spanmask_collator = NewChemMaskCollator(max_length=self.max_length,
                                                     pad_token_id=self.tokenizer.pad_token_id)
        self.save_hyperparameters()
        self.kwargs = kwargs
        if kwargs.get("fragment_length", None) is not None:
            if kwargs.get("fragment_length", None) > self.max_length:
                warnings.warn("Fragment length is greater than max length, following max length over fragment length")
        # self.fragment_length = kwargs.get("fragment_length", None)

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            self.prot_chem_train_dataset = SpanMaskDataset(data_dir=self.data_dirs["train"], tokenizer=self.tokenizer,
                                                           prefix="Span-Mask:", max_length=self.max_length)
            self.chemmask_train_dataset = ChemMaskDataset(self.data_dirs["chemmask"], tokenizer=self.tokenizer,
                                                          prefix='Chem-Mask:', **self.kwargs)
            self.chem_val_dataset = SpanMaskDataset(self.data_dirs["chem_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
            self.chemmask_val_dataset = ChemMaskDataset(self.data_dirs["chemmask_val"], tokenizer=self.tokenizer,
                                                        prefix='Chem-Mask:', max_length=self.max_length)
            self.prot_val_dataset = SpanMaskDataset(self.data_dirs["prot_val"], tokenizer=self.tokenizer,
                                                    prefix='Span-Mask:', max_length=self.max_length)
        else:
            raise ValueError("Stage must be fit or validate")

    def train_dataloader(self):
        prot_chem_span = DataLoader(self.prot_chem_train_dataset, batch_size=self.batch_size,
                                    collate_fn=self.spanmask_collator, shuffle=True,
                                    num_workers=self.workers,
                                    pin_memory=bool(device_count()),
                                    )
        chemmask = DataLoader(self.chemmask_train_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=True,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()),
                              )

        if self.chemmask:
            iterables = {"prot_chem_span": prot_chem_span, "chemmask": chemmask}
        else:
            iterables = {"prot_chem_span": prot_chem_span}
        train_combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")
        return train_combined_dataloader

    def val_dataloader(self):
        chem_span = DataLoader(self.chem_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        prot_span = DataLoader(self.prot_val_dataset, batch_size=self.batch_size, collate_fn=self.spanmask_collator,
                               shuffle=False,
                               num_workers=self.workers,
                               pin_memory=bool(device_count()))
        chemmask = DataLoader(self.chemmask_val_dataset, batch_size=self.batch_size,
                              collate_fn=self.chemmask_collator, shuffle=False,
                              num_workers=self.workers,
                              pin_memory=bool(device_count()))
        iterables = {"chem_span": chem_span, "prot_span": prot_span, "chemmask": chemmask}
        val_combined_dataloader = CombinedLoader(iterables, mode="sequential")
        return val_combined_dataloader


class GOTermData(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, vocab_file=None, workers=None, domain="mf", prefix=""):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)
        self.collator = BaseCollator(tokenizer=self.tokenizer, prefix=prefix)
        self.domain = domain
        self.save_hyperparameters()

    def get_tokenizer(self):
        return self.tokenizer

    def get_vocab_file(self):
        return self.vocab_file

    def prepare_data(self):
        mf_lookup = read_json(os.path.join(self.data_dir, "mf_lookup.json"))
        bp_lookup = read_json(os.path.join(self.data_dir, "bp_lookup.json"))
        cc_lookup = read_json(os.path.join(self.data_dir, "cc_lookup.json"))

        df = pd.read_csv(os.path.join(self.data_dir, "pdb_go_info.tsv"), sep="\t", lineterminator="\n")
        self.one_hot_lookup = {}
        for i in range(len(df)):
            key = df.iloc[i][0]
            one_hot_mf = torch.zeros(len(mf_lookup))
            one_hot_bp = torch.zeros(len(bp_lookup))
            one_hot_cc = torch.zeros(len(cc_lookup))
            if isinstance(df.iloc[i][1], str):
                for mf in df.iloc[i][1].split(","):
                    one_hot_mf[mf_lookup[mf]] = 1
            if isinstance(df.iloc[i][2], str):
                for bp in df.iloc[i][2].split(","):
                    one_hot_bp[bp_lookup[bp]] = 1
            if isinstance(df.iloc[i][3], str):
                for cc in df.iloc[i][3].split(","):
                    one_hot_cc[cc_lookup[cc]] = 1
            self.one_hot_lookup[key] = {"mf": one_hot_mf, "bp": one_hot_bp, "cc": one_hot_cc}

    def setup(self, stage):
        self.train_dataset = GOTermDataset(data_dir=self.data_dir, mode="train", lookup=self.one_hot_lookup,
                                           domain=self.domain)
        self.val_dataset = GOTermDataset(data_dir=self.data_dir, mode="val", lookup=self.one_hot_lookup,
                                         domain=self.domain)
        self.test_dataset = GOTermDataset(data_dir=self.data_dir, mode="test", lookup=self.one_hot_lookup,
                                          domain=self.domain)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator,
                          num_workers=self.workers,
                          pin_memory=bool(device_count()))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers,
                          pin_memory=bool(device_count()))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers,
                          pin_memory=bool(device_count()))

class ESMGOTermData(GOTermData):
    def __init__(self, data_dir, batch_size, vocab_file=None, workers=None, domain="mf", prefix="", **kwargs):
        super().__init__(data_dir, batch_size, vocab_file, workers, domain, prefix)
        self.collator = ESMCollator(tokenizer=self.tokenizer, prefix=prefix, **kwargs)

    def setup(self, stage):
        self.train_dataset = ESMGOTermDataset(data_dir=self.data_dir, mode="train", lookup=self.one_hot_lookup,
                                           domain=self.domain)
        self.val_dataset = ESMGOTermDataset(data_dir=self.data_dir, mode="val", lookup=self.one_hot_lookup,
                                         domain=self.domain)
        self.test_dataset = ESMGOTermDataset(data_dir=self.data_dir, mode="test", lookup=self.one_hot_lookup,
                                          domain=self.domain)

class EsmAlign(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, vocab_file, workers=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True, return_token_type_ids=False)

        self.collator = BaseCollator(tokenizer=self.tokenizer, prefix="")
        self.save_hyperparameters()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = EsmAlignDataset(data_dir=self.data_dir, mode="train")
        elif stage == "val":
            self.val_dataset = EsmAlignDataset(data_dir=self.data_dir, mode="val")
        else:
            self.test_dataset = EsmAlignDataset(data_dir=self.data_dir, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
                          pin_memory=bool(device_count()))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                          pin_memory=bool(device_count()))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
                          pin_memory=bool(device_count()))


class ESMRegression(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, vocab_file, scaler_path=None, workers=None, hdf5=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 # TODO rethink this initialization. This will make it difficult to add new tokens. and manage the tokenizer if changes are needed.
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True)
        self.scaler_path = scaler_path
        self.collator = ESMCollator(tokenizer=self.tokenizer, prefix="")
        self.save_hyperparameters()
        print("data hparams")
        print(self.hparams)

    def get_tokenizer(self):
        return self.tokenizer

    def setup(self, stage: str) -> None:
        if self.scaler_path:
            scaler = joblib.load(self.scaler_path)
        else:
            scaler = None
        if stage == "fit":
            if self.hparams.hdf5:
                self.train_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                             type_path="train", scaler=scaler)
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.train_dataset = ESMPolarsDataset(self.data_dir, type_path="train", has_header=False,
                                                      scaler=scaler)
                self.val_dataset = ESMPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler)
        elif stage == "validate":
            if self.hparams.hdf5:
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.val_dataset = ESMPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler)

        elif stage == "test":
            if self.hparams.hdf5:
                self.test_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                            type_path="test", scaler=scaler)
            else:
                self.test_dataset = ESMPolarsDataset(self.data_dir, type_path="test", has_header=False, scaler=scaler)
        else:
            raise NotImplementedError("Only fit stage is implemented")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))
class AltRegression(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, vocab_file, scaler_path=None, workers=None, hdf5=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.workers = get_workers() if workers is None else workers
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.vocab_file, bos_token="<pad>",
                                                 # TODO rethink this initialization. This will make it difficult to add new tokens. and manage the tokenizer if changes are needed.
                                                 eos_token="</s>",
                                                 unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
                                                 return_special_tokens_mask=True)
        self.scaler_path = scaler_path
        self.collator = AltCollator(tokenizer=self.tokenizer, prefix="")
        self.save_hyperparameters()
        print("data hparams")
        print(self.hparams)

    def get_tokenizer(self):
        return self.tokenizer

    def setup(self, stage: str) -> None:
        if self.scaler_path:
            scaler = joblib.load(self.scaler_path)
        else:
            scaler = None
        if stage == "fit":
            if self.hparams.hdf5:
                self.train_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                             type_path="train", scaler=scaler)
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.train_dataset = AltPolarsDataset(self.data_dir, type_path="train", has_header=False,
                                                      scaler=scaler)
                self.val_dataset = AltPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler)
        elif stage == "validate":
            if self.hparams.hdf5:
                self.val_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                           type_path="val", scaler=scaler)
            else:
                self.val_dataset = AltPolarsDataset(self.data_dir, type_path="val", has_header=False, scaler=scaler)

        elif stage == "test":
            if self.hparams.hdf5:
                self.test_dataset = PropertyPretrainDataset(tokenizer=self.tokenizer, data_dir=self.data_dir,
                                                            type_path="test", scaler=scaler)
            else:
                self.test_dataset = AltPolarsDataset(self.data_dir, type_path="test", has_header=False, scaler=scaler)
        else:
            raise NotImplementedError("Only fit stage is implemented")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator,
                          num_workers=self.workers, pin_memory=bool(device_count()))

if __name__ == '__main__':
    raise ValueError("Entered guarded area")
