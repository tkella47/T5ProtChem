import os
import pickle
from copy import deepcopy
from pathlib import Path
import torchmetrics
import matplotlib.pyplot as plt
import einops
import joblib
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.regression import PearsonCorrCoef, R2Score
from torchview import draw_graph
from transformers import EsmModel, EsmForSequenceClassification
from transformers import T5Config, T5ForConditionalGeneration, get_constant_schedule, get_constant_schedule_with_warmup,\
    get_linear_schedule_with_warmup
from metrics import CIndex, Boundary_Accuracy, R2MPaper, R2MDTA, PosAccuracy


def extract_T5_model_pl(path):
    pl_dict = torch.load(path)
    model_dict = pl_dict["state_dict"]
    if "feed_forward_proj" in pl_dict["hyper_parameters"]:
        feed_forward_proj = pl_dict["hyper_parameters"]["feed_forward_proj"]
    else:
        feed_forward_proj = "relu"
    model = T5Chem(learning_rate=0.0001414, feed_forward_proj=feed_forward_proj)
    model.load_state_dict(model_dict)
    return model.model



def extract_T5_model(path, learning_rate, new_out=None, **kwargs):
    model_dict = torch.load(path)
    model = T5Chem(learning_rate, save_hp = False, **kwargs,)
    if new_out is not None and model_dict["model.lm_head.weight"].shape[0] != model.model.lm_head.weight.shape[0]:
        model.model.set_output_embeddings(nn.Linear(model.model.config.d_model, new_out))
    model.load_state_dict(model_dict)
    return model.model


class T5Chem(pl.LightningModule):

    def __init__(self, learning_rate, tokenizer=None,
                 vocab_size=203, pad_token_id=3, eos_token_id=2, output_past=True, num_layers=4, num_head=8,
                 d_model=256, feed_forward_proj="relu", model_path=None, save_hp=True, **kwargs) -> None:
        super().__init__()
        if tokenizer:
            with open(tokenizer, "rb") as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = None
        self.T5Config = T5Config(vocab_size=vocab_size, pad_token_id=pad_token_id, decoder_start_token_id=pad_token_id,
                                 eos_token_id=eos_token_id, output_past=output_past, num_layers=num_layers,
                                 num_heads=num_head, d_model=d_model, feed_forward_proj=feed_forward_proj)
        self.model = T5ForConditionalGeneration(self.T5Config)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.num_wrong = 0
        try:
            self.slurm_job_id = os.environ["SLURM_JOB_ID"]
        except KeyError:
            self.slurm_job_id = "NA"
        if "cov_binder" in kwargs:
            self.cov_binder = kwargs["cov_binder"]
            if str(self.cov_binder) == "True":
                self.cov_binder = "v1"
            assert self.cov_binder in ["v1", "v2"]
        else:
            self.cov_binder = False
        if "legacy" in kwargs:
            self.legacy = kwargs["legacy"]
        else:
            self.legacy = True
        if "do_smaple" in kwargs:
            self.do_sample = kwargs["do_sample"]
        else:
            self.do_sample = True
        if save_hp:
            self.save_hyperparameters()
        print(self.hparams)

    def on_test_start(self):
        sample_data = torch.randint(10, (1, 128))
        sample_attn = torch.ones(1, 128)
        sample_decoder = torch.randint(10, (1, 100))
        model_graph = draw_graph(self.model, input_data=[sample_data, sample_attn, sample_decoder], save_graph=True,
                                 dtypes=[torch.long, torch.long, torch.long], expand_nested=True,
                                 hide_inner_tensors=False, depth=3)
        return None

    def training_step(self, batch, batch_idx):
        if "labels" not in batch.keys():  # To catch the combined dataloader
            loss = 0
            for ind_batch in batch.values():
                outputs = self.model(**ind_batch)
                loss += outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        if batch_idx == 0:
            tensorboard_logger = self.logger.experiment
            sample_input= deepcopy(batch)
            tensorboard_logger.add_graph(self.model, sample_input)
        """
        outputs = self.model(**batch)
        loss = outputs.loss.detach()
        match self.cov_binder:
            case "v1":
                accuracy_smiles, accuracy_start = self.test_cov_binder(batch, batch_idx, legacy=False)
                self.log("accuracy_smiles", accuracy_smiles, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
                self.log("accuracy_start", accuracy_start, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
            case "v2":
                accuracy_smiles, accuracy_pos = self.test_cov_binder_v2(batch, batch_idx, legacy=False)
                self.log("accuracy_smiles", accuracy_smiles, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
                self.log("accuracy_pos", accuracy_pos, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
            case False:
                pass
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        match self.cov_binder:
            case "v1":
                accuracy_smiles, accuracy_start = self.test_cov_binder(batch, batch_idx, self.legacy)
                self.log("accuracy_smiles", accuracy_smiles, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                        sync_dist=True)
                self.log("accuracy_start", accuracy_start, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                        sync_dist=True)
            case "v2":
                accuracy_smiles, accuracy_pos = self.test_cov_binder_v2(batch, batch_idx, self.legacy)
                self.log("accuracy_smiles", accuracy_smiles, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
                self.log("accuracy_pos", accuracy_pos, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)
            case False:
                self.test_fwdrxn_step(batch, batch_idx, self.legacy)


    def test_cov_binder_v2(self, batch, batch_idx, legacy):
        #labels = deepcopy(batch["labels"])
        labels = batch["labels"].clone().detach()
        labels[labels == -100] = 3
        if legacy:
            outputs = self.model.generate(input_ids = batch.input_ids, attention_mask = batch.attention_mask, max_length=2048, do_sample=self.do_sample)
        else:
            # append a start token to the labels
            outputs = self.model(**batch) # teacher forcing just doesn't as well for testing
            _, outputs = torch.max(F.softmax(outputs.logits, dim=-1), dim=-1)
            outputs = self.silence_noise_after_end(outputs.clone(), self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        accuracy_smiles, accuracy_start = self.collect_indices_smiles_v2(outputs.cpu(), labels.cpu(), remove_first_token=legacy)
        return accuracy_smiles, accuracy_start

    def collect_indices_smiles_v2(self, output, labels, starting_token_idx=202, ending_token_idx=201,
                               remove_first_token=True):
        if remove_first_token:
            output = output[:, 1:]

        # Find the indices of the start and end tokens
        start_indices, end_indices = self.validate_starting_ending_index(output, starting_token_idx, ending_token_idx)

        output_extracted_tokens = [output[i, start:end + 1].tolist() for i, (start, end) in
                                   enumerate(zip(start_indices, end_indices))]
        output_extracted_pos = [output[i, :start].tolist() for i, (start, end) in 
                enumerate(zip(start_indices, end_indices))]
        label_start_indices, label_end_indices = self.validate_starting_ending_index(labels, starting_token_idx,
                                                                                ending_token_idx)
        labels_extracted_tokens = [labels[i, start:end + 1].tolist() for i, (start, end) in
                                   enumerate(zip(label_start_indices, label_end_indices))]
        label_extracted_pos = [labels[i, :start].tolist() for i, (start, end) in enumerate(zip(start_indices, end_indices))]
        # Determine the number of starting indices equal to each other
        accurate_start = 0
        for output_pos, label_pos in zip(output_extracted_pos, label_extracted_pos):
            if label_pos == output_pos:
                accurate_start += 1
        accurate_start = accurate_start / output.shape[0]
        count = 0
        for pred_smiles, label_smiles in zip(output_extracted_tokens, labels_extracted_tokens):
            pred_smiles = torch.tensor(pred_smiles[1:-1])
            label_smiles = torch.tensor(label_smiles[1:-1])
            if pred_smiles.shape == label_smiles.shape and torch.eq(pred_smiles, label_smiles).all():
                count += 1
            else:  # Jump down to rdkit to produce canonical smiles
                if len(pred_smiles) == 0 and len(label_smiles) == 0:
                    count += 1

                elif len(pred_smiles) == 0 or len(label_smiles) == 0:  # Empty set on either the label or pred side!
                    pass

                else:
                    pred_smiles = self.tokenizer.decode(pred_smiles).replace(" ", "")
                    label_smiles = self.tokenizer.decode(label_smiles).replace(" ", "")
                    try:
                        if Chem.CanonSmiles(pred_smiles) == Chem.CanonSmiles(label_smiles):
                            count += 1
                    except:
                        pass
        accurate_smiles = count / len(output_extracted_tokens)
        return accurate_smiles, accurate_start

# This process is so messy. It needs to disposed of.
    def test_cov_binder(self, batch, batch_idx, legacy):
        #labels = deepcopy(batch["labels"])
        labels = batch["labels"].clone().detach()
        labels[labels == -100] = 3
        if legacy:
            outputs = self.model.generate(input_ids = batch.input_ids, attention_mask = batch.attention_mask, max_length=2048, do_sample=self.do_sample)
        else:
            # append a start token to the labels
            outputs = self.model(**batch) # teacher forcing just doesn't as well for testing 
            _, outputs = torch.max(F.softmax(outputs.logits, dim=-1), dim=-1)
            outputs = self.silence_noise_after_end(outputs.clone(), self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        accuracy_smiles, accuracy_start = self.collect_indices_smiles(outputs.cpu(), labels.cpu(), remove_first_token=legacy)
        return accuracy_smiles, accuracy_start

    def silence_noise_after_end(self, outputs, eos_token_id=2, pad_token_id=3):
        for i in range(outputs.size(0)):
            tensor = outputs[i]
            indices = (tensor == eos_token_id).nonzero(as_tuple=True)
            if indices[0].nelement() > 0:  # Check if there is at least one occurrence of 2
                first_index = indices[0][0]  # Getting the index of the first occurrence
                # Modifying all values after the first occurrence of 2 to 3
                tensor[first_index + 1:] = pad_token_id
        return outputs

    def validate_starting_ending_index(self, output, starting_token_idx, ending_token_idx):
        possible_starts = torch.where(output == starting_token_idx)
        possible_ends = torch.where(output == ending_token_idx)
        # Case if all is perfect in the world.
        if torch.equal(torch.arange(0, output.shape[0]), possible_starts[0]) and torch.equal(
                torch.arange(0, output.shape[0]), possible_ends[0]):
            return possible_starts[1], possible_ends[1]  # returns the indices, ignores the batches

        elif possible_starts[0].unique().numel() != possible_starts[0].numel() or possible_ends[0].unique().numel() != possible_ends[0].numel():
            # remove duplicates.
            # we have duplicate cases: We will take the first one. This is not ideal, but it is the best idea I have right now.
            possible_starts_batch = list(possible_starts[0])
            possible_ends_batch = list(possible_ends[0])
            starts_for_delete = []
            ends_for_delete = []
            possible_starts_duplicate = {}
            possible_ends_duplicate = {}
            # This is super messy and confusing and need a rewrite.
            # This block marks and remove keys that are duplicates in an individual sample.
            for i in possible_starts[0]:
                if i.item() not in possible_starts_duplicate.keys():
                    index_of_value = possible_starts_batch.index(i.item())
                    possible_starts_duplicate[i.item()] = possible_starts[1][index_of_value]
                else:
                    starts_for_delete.append(i.item())
            for i in possible_ends[0]:
                if i.item() not in possible_ends_duplicate.keys():
                    index_of_value = possible_ends_batch.index(i.item())
                    possible_ends_duplicate[i.item()] = possible_ends[1][index_of_value]
                else:
                    ends_for_delete.append(i.item())

            for i in set(starts_for_delete):
                del possible_starts_duplicate[i]
            for i in set(ends_for_delete):
                del possible_ends_duplicate[i]

            # This block removes keys not present in both dicts.
            # This handles the case where start or end may have a single, but the other has multiple.
            starts_for_delete = []
            ends_for_delete = []
            for end_key in possible_ends_duplicate.keys():
                if end_key not in possible_starts_duplicate.keys():
                    ends_for_delete.append(end_key)
            for start_key in possible_starts_duplicate.keys():
                if start_key not in possible_ends_duplicate.keys():
                    starts_for_delete.append(start_key)

            for i in set(starts_for_delete):
                del possible_starts_duplicate[i]
            for i in set(ends_for_delete):
                del possible_ends_duplicate[i]

            possible_ends_proc = (torch.tensor(list(possible_ends_duplicate.keys())), torch.tensor(list(possible_ends_duplicate.values())))
            possible_starts_proc = (torch.tensor(list(possible_starts_duplicate.keys())), torch.tensor(list(possible_starts_duplicate.values())))
            return self.fix_the_starting_ending_index(output, possible_starts_proc, possible_ends_proc)
        else:
            return self.fix_the_starting_ending_index(output, possible_starts, possible_ends)

    def fix_the_starting_ending_index(self, output, possible_starts, possible_ends):
        r = torch.arange(0, output.shape[0])
        start_mask = torch.isin(r, possible_starts[0])
        end_mask = torch.isin(r, possible_ends[0])
        missing_start_values = r[~start_mask]
        missing_end_values = r[~end_mask]

        new_possible_starts = []
        new_possible_ends = []
        start_offset = 0
        end_offset = 0
        for i in range(output.shape[0]):
            if i in missing_start_values or i in missing_end_values:
                new_possible_starts.append(2)
                new_possible_ends.append(1)
                if i in missing_start_values:
                    start_offset -= 1
                if i in missing_end_values:
                    end_offset -= 1

            else:
                new_possible_starts.append(possible_starts[1][i + start_offset])
                new_possible_ends.append(possible_ends[1][i + end_offset])
        return torch.tensor(new_possible_starts), torch.tensor(new_possible_ends)
    
    def collect_indices_smiles(self, output, labels, starting_token_idx=202, ending_token_idx=201,
                               remove_first_token=True):
        if remove_first_token:
            output = output[:, 1:]

        # Find the indices of the start and end tokens
        start_indices, end_indices = self.validate_starting_ending_index(output, starting_token_idx, ending_token_idx)

        output_extracted_tokens = [output[i, start:end + 1].tolist() for i, (start, end) in
                                   enumerate(zip(start_indices, end_indices))]

        label_start_indices, label_end_indices = self.validate_starting_ending_index(labels, starting_token_idx,
                                                                                ending_token_idx)

        labels_extracted_tokens = [labels[i, start:end + 1].tolist() for i, (start, end) in
                                   enumerate(zip(label_start_indices, label_end_indices))]

        # Determine the number of starting indices equal to each other
        mask = torch.eq(start_indices, label_start_indices)
        accurate_start = mask.sum().item() / mask.shape[0]

        count = 0
        for pred_smiles, label_smiles in zip(output_extracted_tokens, labels_extracted_tokens):
            pred_smiles = torch.tensor(pred_smiles[1:-1])
            label_smiles = torch.tensor(label_smiles[1:-1])
            if pred_smiles.shape == label_smiles.shape and torch.eq(pred_smiles, label_smiles).all():
                count += 1
            else:  # Jump down to rdkit to produce canonical smiles
                if len(pred_smiles) == 0 and len(label_smiles) == 0:
                    count += 1

                elif len(pred_smiles) == 0 or len(label_smiles) == 0:  # Empty set on either the label or pred side!
                    pass

                else:
                    pred_smiles = self.tokenizer.decode(pred_smiles).replace(" ", "")
                    label_smiles = self.tokenizer.decode(label_smiles).replace(" ", "")
                    try:
                        if Chem.CanonSmiles(pred_smiles) == Chem.CanonSmiles(label_smiles):
                            count += 1
                    except:
                        pass
        accurate_smiles = count / len(output_extracted_tokens)
        return accurate_smiles, accurate_start






    def test_fwdrxn_step(self, batch, batch_idx, legacy):
        if legacy:
            temp_outputs = self.model.generate(batch.input_ids,
                                               max_length=300)  # TODO Investigate this to see if this makes sense.
        else:
            temp_outputs = self.model(**batch)
            # Then we need to cut off the tokens after the stop token. But generate makes more sense for this.
        labels, outputs = self.prepare_for_accuracy(batch, temp_outputs)
        accuracy = self.accuracy(outputs, labels)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return accuracy

    def prepare_for_accuracy(self, batch, temp_outputs):
        temp_outputs = temp_outputs[:, 1:]  # remove start token
        #labels = deepcopy(batch["labels"])
        labels = batch["labels"].clone().detach()
        labels[labels == -100] = 3
        outputs = self.standize(temp_outputs)
        # outputs= temp_outputs
        labels = self.standize(labels)
        # pad labels and outputs to same length
        max_length = max(outputs.shape[-1], labels.shape[-1])
        outputs = torch.nn.functional.pad(outputs, (0, max_length - outputs.shape[-1]), mode='constant', value=3)
        labels = torch.nn.functional.pad(labels, (0, max_length - labels.shape[-1]), mode='constant', value=3)
        return labels, outputs

    def standize(self, outputs):
        return_device = outputs.device
        outputs = outputs
        seq_indices = torch.full((outputs.shape[0],), outputs.shape[-1], dtype=torch.long)
        stop_indices = torch.where(outputs == 2)
        transposed_stop_indices = [tensor.unsqueeze(1) for tensor in stop_indices]
        stop_indices = torch.cat(transposed_stop_indices, dim=1)
        for index, stop_index in enumerate(stop_indices):
            seq_indices[stop_index[0]] = stop_index[1]
        new_outputs = []
        for index, output in enumerate(outputs):
            output = output[:seq_indices[index]]
            try:
                mol = Chem.CanonSmiles(self.tokenizer.decode(output).replace(" ", ""))
            except:
                self.num_wrong = self.num_wrong + 1
                mol = "<unk>"
            new_outputs.append(mol + "</s>")
        results = self.tokenizer(new_outputs, padding=True, truncation=True, return_tensors="pt")
        return results.input_ids.to(return_device)

    def accuracy(self, preds, labels):
        match_list = torch.tensor([torch.equal(preds[i], labels[i]) for i in range(preds.shape[0])])
        return match_list.sum() / len(match_list)

    def configure_optimizers(self):
        model = self.model
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule(optimizer)  # TODO switch scheduler to number of training steps
        scheduler = {'scheduler': scheduler, 'interval': 'step', "frequency": 1}
        return [optimizer], [scheduler]


class T5ChemEsmAlign(T5Chem):
    def __init__(self, learning_rate, tokenizer=None,
                 vocab_size=203, pad_token_id=3, eos_token_id=2, output_past=True, num_layers=4, num_head=8,
                 d_model=256, feed_forward_proj="relu", model_path=None, **kwargs):
        super().__init__(learning_rate, tokenizer, vocab_size, pad_token_id, eos_token_id, output_past, num_layers,
                         num_head, d_model, feed_forward_proj, model_path, **kwargs)
        self.esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        for param in self.esm.parameters():
            param.requires_grad = False
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(0.5)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch["t5chem"]).encoder_last_hidden_state.reshape(-1, self.model.config.d_model)
        esm_outputs = self.esm(**batch["esm"]).last_hidden_state.reshape(-1, self.model.config.d_model)
        loss = self.cosine_loss(outputs, esm_outputs, torch.tensor([1], device=outputs.device))
        self.log('cosine_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)




class T5PropertyRegression(pl.LightningModule):
    # eventually I will go back here and set up inheirtance, but today is not that day
    def __init__(self, scaler_path, checkpoint_path: Path = None, learning_rate=5e-4, num_classes=1, lr_free=False, loss="kl",
                 **kwargs) -> None:
        super().__init__()
        # test
        self.save_hyperparameters()
        self.scaler = None
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
            self.scaler.clip = True
            #self.scaler_min = torch.tensor(self.scaler.data_min_).to(torch.float32).cuda()
            #self.scaler_max = torch.tensor(self.scaler.data_max_).to(torch.float32).cuda()
        
        if checkpoint_path is not None:
            self.model = extract_T5_model(checkpoint_path, learning_rate, 2 if loss=="kl" else 1, **kwargs)
        else:
            self.model = T5Chem(learning_rate, **kwargs).model
        self.lr_free = lr_free
        if loss == "kl" and self.model.lm_head.weight.shape[0] != 2:
            self.model.set_output_embeddings(nn.Linear(self.model.config.d_model, 2 * num_classes))  # Method Call (change LM Head)
        elif loss == "mse" and self.model.lm_head.weight.shape[0] != 1:
            self.model.set_output_embeddings(nn.Linear(self.model.config.d_model, num_classes))
        elif loss == "r2" and self.model.lm_head.weight.shape[0] != 1:
            self.model.set_output_embeddings(nn.Linear(self.model.config.d_model, num_classes))
            self.loss_fn = torchmetrics.R2Score()

        if "schedule" in kwargs:
            self.schedule = kwargs["schedule"]
        else:
            self.schedule = None
        print(self.hparams)
        pear_extras = {"compute_on_cpu": True}
        if "deeper" in kwargs:
            self.model.set_output_embeddings(
                nn.Sequential(
                    nn.Linear(self.model.config.d_model, 2 * self.model.config.d_model),
                    nn.ReLU(),
                    nn.Linear(2 * self.model.config.d_model, 2 * num_classes)
                )
            )
        self.pear = PearsonCorrCoef(**pear_extras).to("cuda")
        self.cindex = CIndex()
        self.r2score = R2Score()
        self.r2m = R2MPaper()
        self.r2score_non = R2MDTA()
        if loss == "kl":
            self.loss_type = loss
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            self.loss_type = "mse"
            self.loss_fn = nn.MSELoss()
        if "write" in kwargs:
            self.write = kwargs["write"]
        else:
            self.write = False
        if "graph" in kwargs:
            self.graph = kwargs["graph"]
        else:
            self.graph = False
        if "draw" in kwargs:
            self.draw = kwargs["draw"]
        else:
            self.draw = False
        if "extra" in kwargs:
            self.extra = kwargs["extra"]
        else:
            self.extra = None
        if "save_fig" in kwargs:
            self.save_fig = kwargs["save_fig"]
        else:
            self.save_fig = False
        if "title" in kwargs:
            self.title = kwargs["title"].replace("_", " ")
        else:
            self.title = "T5ProtChem Davis Binding Affinity"
        if "accuracy" in kwargs:
            if "boundary" in kwargs:
                boundary = float(kwargs["boundary"])
            else:
                boundary = 10.0
            self.accuracy = Boundary_Accuracy(boundary)
        else:
            self.accuracy = False
        if "posaccuracy" in kwargs:
            self.pos_accuracy = PosAccuracy().to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.pos_accuracy = False
        if "units" in kwargs:
            self.units = kwargs["units"]
        else:
            self.units = "pKd"
        self.model.config.tie_word_embeddings=False
        self.save_hyperparameters()

    def on_test_start(self) -> None:
        if self.write or self.graph:
            self.logits_coll = torch.empty(0, 1)
            self.labels_coll = torch.empty(0, 1)
        # write a csv file with the model version number

    def forward(self, **inputs):  # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        outputs = self.model.forward(inputs["input_ids"], inputs["attention_mask"], inputs["decoder_input_ids"])
        # make soft labels with inputs["labels"]
        if self.loss_type == "kl":
            soft_labels = nn.LogSoftmax(dim=-1)(
                einops.rearrange(outputs.logits.squeeze(1), "b (c v) -> b c v ", v=2))  # B, 123, 2
            smoothed_labels = torch.stack([1 - inputs["labels"], inputs["labels"]], dim=-1)
            smoothed_labels = einops.rearrange(smoothed_labels, "b (c v) -> b c v", v=2)
            loss = self.loss_fn(soft_labels, smoothed_labels)
            outputs.logits = soft_labels
        else:
            logits = outputs.logits.squeeze(1)
            outputs.logits = logits
            batch_labels = inputs["labels"].to(torch.float32)
            if self.scaler:
                logits, batch_labels = self.unscale(logits, batch_labels, train=True)
            loss = self.loss_fn(logits, batch_labels.to(torch.float32).unsqueeze(1))


        outputs.loss = loss
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        with torch.no_grad():
            if self.loss_type == "kl":
                logits = torch.exp(outputs.logits[:, :, 1])
            elif self.loss_type == "mse" or self.loss_type == "r2":
                if len(outputs.logits.shape) == 0:
                    logits = outputs.logits.unsqueeze(-1).unsqueeze(-1)
                elif len(outputs.logits.shape) == 1:
                    logits = outputs.logits.unsqueeze(-1)
                else:
                    logits = outputs.logits
            if self.scaler:
                logits, batch_labels = self.unscale(logits, batch["labels"].unsqueeze(-1))
            else:
                batch_labels = batch["labels"].unsqueeze(-1)
            mse_loss = F.mse_loss(logits, batch_labels)

        self.log("train_mse", mse_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_mse_epoch", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss



    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss.detach()
        if len(batch["labels"].shape) == 1:
            batch_labels = batch.labels.unsqueeze(-1)
        else:
            batch_labels = batch.labels
        if self.loss_type == "kl":
            logits = torch.exp(outputs.logits[:, :, -1])
        else:
            if len(outputs.logits.shape) == 0:
                logits = outputs.logits.unsqueeze(-1).unsqueeze(-1)
            elif len(outputs.logits.shape) == 1:
                logits = outputs.logits.unsqueeze(-1) 
            else:
                logits = outputs.logits
        if self.scaler:
            logits, batch_labels = self.unscale(logits, batch_labels)

        mse_loss = F.mse_loss(logits, batch_labels)
        if logits.shape[0] > 1:
            self.pear.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.cindex.to(logits.device)(logits, batch_labels)
            self.r2score.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.r2m.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.r2score_non.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.log("r_score", self.pear, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("cindex", self.cindex, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2score", self.r2score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2m_paper", self.r2m, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2m_DTA", self.r2score_non, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.accuracy:
            self.accuracy(logits, batch_labels)
            self.log("val_bound_accuracy", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.pos_accuracy:
            protein_length_shape = batch["protein_length"].shape
            logits = logits.view(protein_length_shape).to(batch["protein_length"].device)
            batch_labels = batch_labels.view(protein_length_shape).to(batch["protein_length"].device)
            self.pos_accuracy(protein_length=batch["protein_length"], preds=logits, labels=batch_labels)
            self.log("val_pos_accuracy", self.pos_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        if len(batch["labels"].shape) == 1:
            batch_labels = batch.labels.unsqueeze(-1)
        else:
            batch_labels = batch.labels
        if self.loss_type == "kl":
            logits = torch.exp(outputs.logits[:, :, -1])
        else:
            logits = outputs.logits
        if self.scaler:
            logits, batch_labels = self.unscale(logits, batch_labels)
        if self.write or self.graph:
            self.collect(logits, batch_labels)
        mse_loss = F.mse_loss(logits, batch_labels)
        if logits.shape[0] > 1:
            self.pear.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.cindex.to(logits.device)(logits, batch_labels)
            self.r2score.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.r2m.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.r2score_non.to(logits.device)(logits.squeeze(), batch_labels.squeeze())
            self.log("r_score", self.pear, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("cindex", self.cindex, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2score", self.r2score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2m_paper", self.r2m, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2m_DTA", self.r2score_non, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.accuracy:
            self.accuracy(logits, batch_labels)
            self.log("bound_accuracy", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.pos_accuracy:
            protein_length_shape = batch["protein_length"].shape
            logits = logits.view(protein_length_shape).to(batch["protein_length"].device)
            batch_labels = batch_labels.view(protein_length_shape).to(batch["protein_length"].device)
            self.pos_accuracy(protein_length=batch["protein_length"], preds=logits, labels=batch_labels)
            self.log("test_pos_accuracy", self.pos_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return mse_loss

    def collect(self, logits, labels):
        if len(self.logits_coll.shape) != len(logits.shape):
            logits = logits.unsqueeze(1)
        self.logits_coll = torch.cat([self.logits_coll, logits.cpu()], dim=0)
        self.labels_coll = torch.cat([self.labels_coll, labels.cpu()], dim=0)

    def on_test_epoch_end(self) -> None:
        # Might want to move this to training end and just run the datamodule.
        if self.graph:
            tensorboard = self.logger.experiment
            figure = self.graph_test()
            tensorboard.add_figure("test_prediction", figure, global_step=self.current_epoch)
            if self.save_fig:
                figure.savefig(f"{self.logger.log_dir}/t5biochem_test_data_{self.extra if self.extra is not None else str()}.png")
        if self.write:
            print(f"mse_loss: {F.mse_loss(self.logits_coll, self.labels_coll)}")
            import pandas as pd
            df = pd.DataFrame(
                {"Pred": self.logits_coll.cpu().numpy().squeeze(), "Target": self.labels_coll.cpu().numpy().squeeze()})
            df.to_csv(f"{self.logger.log_dir}/predictions.csv", index=False)
        # write to csv
    
    def graph_test(self):
        logits = self.logits_coll.squeeze()
        labels = self.labels_coll.squeeze()
        fig, ax = plt.subplots()
        scatter = ax.scatter(logits, labels, label='Test')
        # Add a dashed line with a slope of 1
        ax.plot([min(logits), max(logits)], [min(logits), max(logits)], 'r--', label='Slope 1 Line')

        corr_coefficient = np.corrcoef(logits.squeeze(), labels.squeeze())[0,1]
        squared_diff = (logits - labels) ** 2
        rmse = torch.sqrt(torch.mean(squared_diff))

        # Add the Pearson correlation coefficient as a label

        # Set axis labels
        ax.set_xlabel(f'Prediction {self.units}')
        ax.set_ylabel(f'Measured {self.units}')

        # Set plot title
        ax.set_title(f"{self.title}")
        #ax.set_title(f't5biochem test data {self.extra if self.extra is not None else str()}')

        # Display legend
        legend_text = f'Pearson CC: {corr_coefficient:.2f}\nRMSE: {rmse:.2f}'
        legend_entry = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                                  label=legend_text)
        ax.legend(handles=[scatter, legend_entry])
        plt.tight_layout()
        return fig



    def configure_optimizers(self):
        model = self.model
        optimizer = AdamW(model.parameters(), lr=self.hparams.learning_rate)
        if self.schedule is not None:
            if self.schedule == "warmup":
                scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1446)
            elif self.schedule == "cosine":  # CosineAnnealingLr
                scheduler = CosineAnnealingLR(optimizer, T_max=1000)
            elif self.schedule == "linear":
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                            num_training_steps=50000)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1446, num_training_steps=50000)
        else:
            scheduler = get_constant_schedule(optimizer)  # TODO switch scheduler to number of training steps

        scheduler = {'scheduler': scheduler, 'interval': 'step', "frequency": 1}
        return [optimizer], [scheduler]

    def unscale(self, pred, target, train=False):
        pred = torch.tensor(self.scaler.inverse_transform(pred.detach().cpu().numpy()))
        target = torch.tensor(self.scaler.inverse_transform(target.detach().cpu().numpy()))
        return pred, target





class T5Classification(pl.LightningModule):
    def __init__(self, checkpoint_path, learning_rate=5e-4, num_cycles=3,
                 num_classes=498, max_steps=300000, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = extract_T5_model(checkpoint_path, learning_rate, **kwargs)
        self.model.set_output_embeddings(nn.Linear(self.model.config.d_model, num_classes)) # Change LM Head)
        self.accuracy = Accuracy("binary")
        self.F1Score = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.record_preds = []
        self.model.config.tie_word_embeddings = False

        if "graph" in kwargs:
            self.graph = kwargs["graph"]
        else:
            self.graph = False

        print(self.hparams)


    def forward(self, **batch):
        outputs = self.model(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"])
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs.logits.squeeze(), batch["labels"])
        outputs.logits = torch.nn.functional.sigmoid(outputs.logits)
        outputs.loss = loss
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log('train_loss', outputs.loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss.detach()

        self.accuracy(outputs.logits.squeeze(), batch.labels)
        self.F1Score(outputs.logits.squeeze(), batch.labels)
        self.precision(outputs.logits.squeeze(), batch.labels)
        self.recall(outputs.logits.squeeze(), batch.labels)
        self.log("accuracy_loss", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("F1_score", self.F1Score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("precision", self.precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.record_preds.append((outputs.logits.squeeze(), batch.labels))
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss.detach()
        logits = outputs.logits.squeeze()
        self.accuracy(logits, batch.labels)
        self.F1Score(logits, batch.labels)
        self.precision_values.append(self.precision(logits, batch.labels))
        self.recall_values.append(self.recall(logits, batch.labels))
        self.record_preds.append((logits, batch.labels))
        self.log("accuracy_loss", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("F1_score",self.F1Score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("precision", self.precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("recall", self.recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.recall_values = []
        self.precision_values = []


        self.record_preds = []

    def on_validation_epoch_end(self):
        preds, labels = zip(*self.record_preds)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        thresholds = torch.linspace(0, 1, 100, device=self.device)
        fmax = 0.0
        Fscore = BinaryF1Score().to(self.device)
        for threshold in thresholds:
            threshold_preds = (preds > threshold).to(torch.int)
            fscore = Fscore(threshold_preds, labels)
            if fscore > fmax:
                fmax = fscore
        self.log("fmax_score", fmax, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.graph:
            self.graph_recall_precision()

    def graph_recall_precision(self):
        log_dir = self.logger.log_dir
        fig_, ax_ = self.recall.plot(self.recall_values)
        fig_.savefig(log_dir + "/recall.png")
        fig_, ax_ = self.precision.plot(self.precision_values)
        fig_.savefig(log_dir + "/precision.png")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()


    def configure_optimizers(self):
        #model = self.model
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule(optimizer)  # TODO switch scheduler to number of training steps
        scheduler = {'scheduler': scheduler, 'interval': 'step', "frequency": 1}
        return [optimizer], [scheduler]

class T5ESMClassification(T5Classification):
    def __init__(self, checkpoint_path, learning_rate=5e-4, num_cycles=3,
                 num_classes=498, max_steps=300000, **kwargs) -> None:
        super().__init__(checkpoint_path, learning_rate, num_cycles, num_classes, max_steps, **kwargs)
        self.esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        # freeze all the parameters in esm
        for param in self.esm.parameters():
            param.requires_grad = False

    def forward(self, **batch):
        esm_embeddings = self.esm(batch["esm_input_ids"], batch["esm_attention_mask"]).last_hidden_state
        outputs = self.model(batch["input_ids"], batch["attention_mask"], decoder_inputs_embeds=esm_embeddings, decoder_attention_mask=batch["esm_attention_mask"] )
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs.logits[:,-1,:].squeeze(1), batch["labels"])
        outputs.logits = torch.nn.functional.sigmoid(outputs.logits[:,-1,:])
        outputs.loss = loss
        return outputs

    def validation_step(self, batch, batch_idx):
        batch.labels = batch.labels.squeeze()
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        batch.labels = batch.labels.squeeze()
        return super().test_step(batch,batch_idx)

class ESMGoTerm(pl.LightningModule):
    def __init__(self, learning_rate, num_classes=489, lora=False, **kwargs) -> None:
        super().__init__()
        esm = EsmForSequenceClassification.from_pretrained("facebook/esm2_t30_150M_UR50D", device_map="auto")
        esm.classifier.out_proj = nn.Linear(640, num_classes)
        esm.config.problem_type = "multi_label_classification"
        self.accuracy = Accuracy("binary")
        self.F1Score = BinaryF1Score()
        self.save_hyperparameters()
        self.record_preds = []

    def forward(self, batch):
        outputs = self.lora_esm(input_ids=batch["esm_input_ids"], attention_mask=batch["esm_attention_mask"], labels=batch["labels"])
        #loss_fn = nn.BCELoss()
        #loss = loss_fn(outputs.logits.squeeze(1), batch["labels"])
        #outputs["logits"] = logits.squeeze(1)
        #outputs.loss = loss
        outputs.logits = torch.nn.functional.sigmoid(outputs.logits)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log('train_loss', outputs.loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        return outputs.loss


    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss.detach()
        self.accuracy(outputs.logits, batch.labels)
        self.F1Score(outputs.logits, batch.labels)
        self.log("accuracy_loss", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("F1_score", self.F1Score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss.detach()
        self.record_preds.append((outputs.logits, batch.labels))
        self.accuracy(outputs.logits, batch.labels)
        self.F1Score(outputs.logits, batch.labels)
        self.log("accuracy_loss", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True)
        self.log("F1_score", self.F1Score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        preds, labels = zip(*self.record_preds)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        thresholds = torch.linspace(0, 1, 100, device=self.device)
        fmax = 0.0
        Fscore = BinaryF1Score().to(self.device)
        for threshold in thresholds:
            threshold_preds = (preds > threshold).to(torch.int)
            fscore = Fscore(threshold_preds, labels)
            if fscore > fmax:
                fmax = fscore
        self.log("fmax_score", fmax, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule(optimizer)  # TODO switch scheduler to number of training steps
        scheduler = {'scheduler': scheduler, 'interval': 'step', "frequency": 1}
        return [optimizer], [scheduler]


class T5ESMRegression(T5PropertyRegression):  # eventually I will go back here and set up inheirtance, but today is not that day
    def __init__(self, scaler_path, checkpoint_path: Path = None, learning_rate=5e-4, num_classes=1, lr_free=False,
                 **kwargs) -> None:
        super().__init__(scaler_path, checkpoint_path, learning_rate, num_classes, lr_free, **kwargs)
        self.esm = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
        # freeze all the parameters in esm
        for param in self.esm.parameters():
            param.requires_grad = False
        if self.model.config.d_model != self.esm.config.hidden_size:
            self.projection = nn.Sequential(
            nn.Linear(self.esm.config.hidden_size, self.model.config.d_model),
            nn.ReLU()
             )
        else :
            self.projection = nn.Identity()


    def forward(self, **inputs):  # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # TODO We will rework embeddings to serve input_ids, attention_mask, and esm_input_ids
        esm_embeddings = self.esm(inputs["esm_input_ids"], inputs["esm_attention_mask"]).last_hidden_state
        esm_embeddings = self.projection(esm_embeddings)
        # TODO ensure we don't pull the class tag
        outputs = self.model(inputs["input_ids"], inputs["attention_mask"], decoder_inputs_embeds=esm_embeddings,
                             decoder_attention_mask=inputs["esm_attention_mask"])  # Check on decoder attn mask
        # make soft labels with inputs["labels"]
        outputs.logits = outputs.logits[:, -1, :]
        soft_labels = nn.LogSoftmax(dim=-1)(
            einops.rearrange(outputs.logits.squeeze(1), "b (c v) -> b c v ", v=2))  # B, 123, 2
        smoothed_labels = torch.stack([1 - inputs["labels"], inputs["labels"]], dim=-1)
        smoothed_labels = einops.rearrange(smoothed_labels, "b (c v) -> b c v", v=2)
        loss = self.loss_fn(soft_labels, smoothed_labels)
        outputs.loss = loss
        outputs.logits = soft_labels
        return outputs

class T5AltRegression(T5PropertyRegression):  # eventually I will go back here and set up inheirtance, but today is not that day
    def __init__(self, scaler_path, checkpoint_path: Path = None, learning_rate=5e-4, num_classes=1, lr_free=False,
                 **kwargs) -> None:
        super().__init__(scaler_path, checkpoint_path, learning_rate, num_classes, lr_free, **kwargs)

    def forward(self, **inputs):  # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # TODO We will rework embeddings to serve input_ids, attention_mask, and esm_input_ids
        # TODO ensure we don't pull the class tag
        outputs = self.model(inputs["input_ids"], inputs["attention_mask"], decoder_input_ids=inputs["decoder_input_ids"],
                             decoder_attention_mask=inputs["decoder_attention_mask"])  # Check on decoder attn mask
        # make soft labels with inputs["labels"]
        outputs.logits = outputs.logits[:, -1, :]
        soft_labels = nn.LogSoftmax(dim=-1)(
            einops.rearrange(outputs.logits.squeeze(1), "b (c v) -> b c v ", v=2))  # B, 123, 2
        smoothed_labels = torch.stack([1 - inputs["labels"], inputs["labels"]], dim=-1)
        smoothed_labels = einops.rearrange(smoothed_labels, "b (c v) -> b c v", v=2)
        smoothed_labels = torch.clamp(smoothed_labels, min=0.0, max=1.0)
        loss = self.loss_fn(soft_labels, smoothed_labels)
        outputs.loss = loss
        outputs.logits = soft_labels
        return outputs


class T5AltSwap(T5PropertyRegression):
    def __init__(self, scaler_path, checkpoint_path: Path = None, learning_rate=5e-4, num_classes=1, lr_free=False,
                 **kwargs) -> None:
        super().__init__(scaler_path, checkpoint_path, learning_rate, num_classes, lr_free, **kwargs)

    def forward(self, **inputs):  # input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # TODO We will rework embeddings to serve input_ids, attention_mask, and esm_input_ids
        # TODO ensure we don't pull the class tag
        outputs = self.model(inputs["decoder_input_ids"], inputs["decoder_attention_mask"], decoder_input_ids=inputs["input_ids"],
                             decoder_attention_mask=inputs["attention_mask"])  # Check on decoder attn mask
        # make soft labels with inputs["labels"]
        outputs.logits = outputs.logits[:, -1, :]
        soft_labels = nn.LogSoftmax(dim=-1)(
            einops.rearrange(outputs.logits.squeeze(1), "b (c v) -> b c v ", v=2))  # B, 123, 2
        smoothed_labels = torch.stack([1 - inputs["labels"], inputs["labels"]], dim=-1)
        smoothed_labels = einops.rearrange(smoothed_labels, "b (c v) -> b c v", v=2)
        loss = self.loss_fn(soft_labels, smoothed_labels)
        outputs.loss = loss
        outputs.logits = soft_labels
        return outputs


class Ensemble(pl.LightningModule):

    def __init__(self, model_path_dir, **kwargs) -> None:
        super().__init__()
        import yaml
        import json
        # Read the json file and extract the file model weight paths.
        # Read a json file @ model_path_dir
        self.pear = PearsonCorrCoef()
        self.cindex = CIndex()
        self.r2score = R2Score()
        with open(model_path_dir, "r") as f:
            model_paths = json.load(f)
        self.models = []
        for model_path in model_paths.values():

            with open(f"{model_path}/config.yaml", "r") as f:
                model_config = yaml.safe_load(f)
            # check if model.model.pt exists at model path
            model_state_dict_path = os.path.join(model_path, "checkpoints", "model.model.pt")
            if not os.path.isfile(model_state_dict_path):
                assert os.path.isfile(f"{model_path}/checkpoints/last.ckpt"), f"Model checkpoint not found at {model_path}/checkpoints/last.ckpt"
                torch.save(torch.load(f"{model_path}/checkpoints/last.ckpt", map_location="cpu")["state_dict"], model_state_dict_path)
            model_config["model"]["init_args"]["checkpoint_path"] = model_state_dict_path
            model = T5PropertyRegression(**model_config["model"]["init_args"], **model_config["model"]["dict_kwargs"]).cuda().eval()
            self.models.append(model)
            self.model_paths = model_paths

        if "write" in kwargs:
            self.write = kwargs["write"]
        else:
            self.write = False
        if "graph" in kwargs:
            self.graph = kwargs["graph"]
        else:
            self.graph = False
        if "draw" in kwargs:
            self.draw = kwargs["draw"]
        else:
            self.draw = False
        if "extra" in kwargs:
            self.extra = kwargs["extra"]
        else:
            self.extra = None
        if "save_fig" in kwargs:
            self.save_fig = kwargs["save_fig"]
        else:
            self.save_fig = False
        if "title" in kwargs:
            self.title = kwargs["title"]
        else:
            self.title = "T5BioChem BindingDB Binding Affinity Ensemble Model"
        if "units" in kwargs:
            self.units = kwargs["units"]
        else:
            self.units = "pKd"

    def forward(self, **batch):
        # Empty torch tensor to cat to later
        with torch.no_grad():
            outputs = torch.tensor([])
            for model in self.models:
                unscaled_output = model(**batch).logits[:,:,1].squeeze()
                unscaled_output = torch.exp(unscaled_output)
                preds, _ = model.unscale(unscaled_output.unsqueeze(0), batch["labels"].unsqueeze(0))
                outputs = torch.cat((outputs, preds), dim=0)
        return outputs

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Ensemble model does not support training")

    def shared_val_test(self, batch, batch_idx):
        outputs = self.forward(**batch).to(batch.labels.device)
        prediction = outputs.mean(dim=0)
        mse_loss = F.mse_loss(prediction, batch.labels)
        return mse_loss, prediction, outputs
    
    def shared_routine(self,batch, batch_idx):
        mse_loss, prediction, outputs = self.shared_val_test(batch, batch_idx)
        if prediction.shape[0] > 1:
            self.pear(prediction, batch.labels)
            self.cindex(prediction, batch.labels)
            self.r2score(prediction, batch.labels)
            self.log("r_score", self.pear, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("cindex", self.cindex, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("r2score", self.r2score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.write or self.graph:
            self.collect(prediction, batch["labels"].unsqueeze(1))
        return mse_loss, prediction, outputs
    
    def validation_step(self, batch, batch_idx):
        mse_loss, prediction, outputs = self.shared_routine(batch, batch_idx)
        return mse_loss

    def test_step(self, batch, batch_idx):
        mse_loss, prediction, outputs = self.shared_routine(batch, batch_idx)
        return mse_loss
    
    def on_validation_end(self):
        self.on_test_end()
    
    def on_test_end(self):
        import json
        save_path = self.logger.log_dir
        with open(os.path.join(save_path, "model_paths.json"),"w") as f :
            json.dump(self.model_paths, f)

    def on_test_start(self) -> None:
        if self.write or self.graph:
            self.logits_coll = torch.empty(0, 1)
            self.labels_coll = torch.empty(0, 1)
    
    def collect(self, logits, labels):
        if len(self.logits_coll.shape) != len(logits.shape):
            logits = logits.unsqueeze(1)
        self.logits_coll = torch.cat([self.logits_coll, logits.cpu()], dim=0)
        self.labels_coll = torch.cat([self.labels_coll, labels.cpu()], dim=0)

    def on_test_epoch_end(self) -> None:
        # Might want to move this to training end and just run the datamodule.
        if self.graph:
            tensorboard = self.logger.experiment
            figure = self.graph_test()
            tensorboard.add_figure("test_prediction", figure, global_step=self.current_epoch)
            if self.save_fig:
                figure.savefig(f"{self.logger.log_dir}/t5biochem_test_data_{self.extra if self.extra is not None else str()}.png")
        if self.write:
            print(f"mse_loss: {F.mse_loss(self.logits_coll, self.labels_coll)}")
            import pandas as pd
            df = pd.DataFrame(
                {"Pred": self.logits_coll.cpu().numpy().squeeze(), "Target": self.labels_coll.cpu().numpy().squeeze()})
            df.to_csv(f"{self.logger.log_dir}/predictions.csv", index=False)

    def graph_test(self):
        logits = self.logits_coll.squeeze()
        labels = self.labels_coll.squeeze()
        fig, ax = plt.subplots()
        scatter = ax.scatter(logits, labels, label='Test')
        # Add a dashed line with a slope of 1
        ax.plot([min(logits), max(logits)], [min(logits), max(logits)], 'r--', label='Slope 1 Line')

        corr_coefficient = np.corrcoef(logits.squeeze(), labels.squeeze())[0,1]
        squared_diff = (logits - labels) ** 2
        rmse = torch.sqrt(torch.mean(squared_diff))

        # Add the Pearson correlation coefficient as a label

        # Set axis labels
        ax.set_xlabel(f'Prediction {self.units}')
        ax.set_ylabel(f'Measured {self.units}')

        # Set plot title
        ax.set_title(f"{self.title}")
        #ax.set_title(f't5biochem test data {self.extra if self.extra is not None else str()}')

        # Display legend
        legend_text = f'Pearson CC: {corr_coefficient:.2f}\nRMSE: {rmse:.2f}'
        legend_entry = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                                  label=legend_text)
        ax.legend(handles=[scatter, legend_entry])
        plt.tight_layout()
        return fig

    def configure_optimizers(self):
        raise NotImplementedError("Ensemble model does not support training")
