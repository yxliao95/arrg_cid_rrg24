# use predicted labels as decoder input for generation

import argparse
import json
import logging
import os
import random
import time
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from config_cpu import exp4_vis2lan as CONFIG
from datasets import DatasetDict, load_from_disk
from load_dataset import get_dataset
from nltk.tokenize import wordpunct_tokenize
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
    shift_tokens_right,
)
from utils import StableRandomSampler, set_seed
from vilmedic.blocks.scorers.scores import compute_scores

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOGGER = None
DEVICE = None
PAD_TOKEN_ID = None

TENSORBOARD = None

LABELS_DICT = None


##############################################
# Model
##############################################
class MLC(nn.Module):
    def __init__(self, fc_in_features, label_embedding_dim, num_class_present, num_class_absent, num_class_uncertain):
        super().__init__()

        self.num_class_present = num_class_present
        self.num_class_absent = num_class_absent
        self.num_class_uncertain = num_class_uncertain

        self.classifier_present = nn.Linear(fc_in_features, self.num_class_present)
        self.classifier_absent = nn.Linear(fc_in_features, self.num_class_absent)
        self.classifier_uncertain = nn.Linear(fc_in_features, self.num_class_uncertain)

    def forward(self, pooled_features):

        present_logits = self.classifier_present(pooled_features)
        absent_logits = self.classifier_absent(pooled_features)
        uncertain_logits = self.classifier_uncertain(pooled_features)

        return present_logits, absent_logits, uncertain_logits


class Vision2LanguageModel(VisionEncoderDecoderModel):
    def __init__(self, config=None, encoder=None, decoder=None):

        super().__init__(config=config, encoder=encoder, decoder=decoder)

        # extra components
        self.pooler = nn.AdaptiveAvgPool1d(1)

        self.config.encoder_hidden_size = self.encoder.config.hidden_size
        self.config.decoder_hidden_size = self.decoder.config.hidden_size
        if self.config.encoder_hidden_size != self.config.decoder_hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.config.encoder_hidden_size, self.config.decoder_hidden_size)

        self.num_class_present = CONFIG.img_label_present
        self.num_class_absent = CONFIG.img_label_absent
        self.num_class_uncertain = CONFIG.img_label_uncertain

        self.mlc = MLC(
            fc_in_features=self.config.encoder_hidden_size,
            label_embedding_dim=self.config.decoder_hidden_size,
            num_class_present=self.num_class_present,
            num_class_absent=self.num_class_absent,
            num_class_uncertain=self.num_class_uncertain,
        )

    def image_classification(self, image_features):

        # torch.Size([bsz, 768, 64]) -> torch.Size([bsz, 768, 1]) -> torch.Size([bsz, 768])
        pooled_features = self.pooler(image_features.transpose(1, 2)).squeeze(2)
        present_logits, absent_logits, uncertain_logits = self.mlc(pooled_features)

        return present_logits, absent_logits, uncertain_logits, pooled_features

    def get_label_text(self, present_logits, absent_logits, uncertain_logits, topk=10, threshold=0.5):
        # 获取分类标签
        present_probs = torch.sigmoid(present_logits)
        absent_probs = torch.sigmoid(absent_logits)
        uncertain_probs = torch.sigmoid(uncertain_logits)

        bsz = present_probs.shape[0]
        batch_label_p = [[] for _ in range(bsz)]
        batch_label_a = [[] for _ in range(bsz)]
        batch_label_u = [[] for _ in range(bsz)]

        # 每个batch获取topk个概率大于threshold的label的标签。获取标签的数量可以小于k个，要优先保证threshold
        all_probs = torch.cat([present_probs, absent_probs, uncertain_probs], dim=1)
        top_probs, top_indices = torch.topk(all_probs, k=topk, dim=1)
        for indices in torch.nonzero(top_probs > threshold):
            batch_idx = indices[0]
            label_idx = top_indices[batch_idx, indices[1]]
            if label_idx < self.num_class_present:
                batch_label_p[batch_idx].append(LABELS_DICT["present"][label_idx])
            elif self.num_class_present <= label_idx < (self.num_class_present + self.num_class_absent):
                label_idx = label_idx - self.num_class_present
                batch_label_a[batch_idx].append(LABELS_DICT["absent"][label_idx])
            else:
                label_idx = label_idx - self.num_class_present - self.num_class_absent
                batch_label_u[batch_idx].append(LABELS_DICT["uncertain"][label_idx])

        # 构造label_text
        batch_label_text = get_batch_label_text(batch_label_p, batch_label_a, batch_label_u)

        return batch_label_text

    def forward_train(self, input_dict):
        # Image cls: https://github.com/huggingface/transformers/blob/745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec/src/transformers/models/swinv2/modeling_swinv2.py#L1239

        # extract img features
        encoder_outputs = self.encoder(pixel_values=input_dict["pixel_values"], return_dict=True)
        image_features = encoder_outputs.last_hidden_state  # torch.Size([4, 64, enc_dim])

        # img cls
        present_logits, absent_logits, uncertain_logits, pooled_features = self.image_classification(image_features)

        # cls loss
        bce_loss_fct = nn.BCEWithLogitsLoss(reduction="mean")  # multi_label_classification
        img_loss_p = bce_loss_fct(present_logits, input_dict["img_labels_present"].float())
        img_loss_a = bce_loss_fct(absent_logits, input_dict["img_labels_absent"].float())
        img_loss_u = bce_loss_fct(uncertain_logits, input_dict["img_labels_uncertain"].float())
        img_loss = img_loss_p + img_loss_a + img_loss_u

        # project image features
        if self.config.encoder_hidden_size != self.config.decoder_hidden_size:
            image_features = self.enc_to_dec_proj(image_features)

        # <s>label_text</s></s>report_section</s><pad-100> --> </s><s>label_text</s></s>report_section</s><pad0>
        decoder_input_ids = shift_tokens_right(input_dict["decoder_input_ids"], self.config.pad_token_id, self.config.decoder_start_token_id)

        # text generation
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=input_dict["decoder_attention_masks"], encoder_hidden_states=image_features, return_dict=True)

        # text loss
        text_logits = decoder_outputs.logits
        ce_loss_fct = nn.CrossEntropyLoss()
        text_loss = ce_loss_fct(text_logits.view(-1, self.decoder.config.vocab_size), input_dict["decoder_input_ids"].view(-1))

        return text_loss, img_loss

    def do_generate(self, input_dict, tokenizer):
        """先提取图像特征，然后拼接特征，最后主动调用generate函数。该函数会调用forward方法进行生成"""

        # encode process
        image_features = self.encoder(pixel_values=input_dict["pixel_values"], return_dict=True).last_hidden_state  # torch.Size([4, 64, 768])
        present_logits, absent_logits, uncertain_logits, _ = self.image_classification(image_features)
        batch_label_text = self.get_label_text(present_logits, absent_logits, uncertain_logits, topk=CONFIG.topk, threshold=CONFIG.threshold)

        # manually add special tokens to the input text
        if "bart" in tokenizer.name_or_path:
            prefix_tokens = tokenizer.eos_token + tokenizer.bos_token
            suffix_tokens = tokenizer.eos_token + tokenizer.eos_token
        elif "roberta" in tokenizer.name_or_path:
            prefix_tokens = tokenizer.bos_token + tokenizer.bos_token
            suffix_tokens = tokenizer.eos_token + tokenizer.eos_token
        else:
            raise ValueError("Check tokenizer about how it will add special tokens between two sentence")
        batch_label_text = [prefix_tokens + text + suffix_tokens for text in batch_label_text]

        # left_padding
        tokenizer.padding_side = "left"
        assert tokenizer.padding_side == "left", f"tokenizer.padding_side should be [left] but got: [{tokenizer.padding_side}]"
        tokenized_out = tokenizer(batch_label_text, padding=True, add_special_tokens=False, return_tensors="pt")
        label_text_len = tokenized_out.input_ids.shape[1]

        if self.config.encoder_hidden_size != self.config.decoder_hidden_size:
            image_features = self.enc_to_dec_proj(image_features)
        encoder_outputs = BaseModelOutput(last_hidden_state=image_features)

        # decode process
        # when we pass decoder_input_ids, the generate func will use it as decoder input rather than build a new input
        outputs = self.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=tokenized_out.input_ids.to(DEVICE),
            decoder_attention_mask=tokenized_out.attention_mask.to(DEVICE),
            max_new_tokens=CONFIG.max_generation_len,
            num_beams=CONFIG.num_beam,
            early_stopping=True,
            renormalize_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        # outputs.scores是数组，数组的每个元素表示一次自回归迭代，并为每个句子生成了一个token
        self.config.vocab_size = self.decoder.config.vocab_size
        transition_scores = self.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
        transition_logits = self.compute_transition_scores(outputs.sequences, outputs.logits, outputs.beam_indices, normalize_logits=False)
        print(transition_scores)

        return {
            "generated_ids": generated_ids,
            "generated_ids_without_label_text": generated_ids[:, label_text_len:],
            "present_logits": present_logits,
            "absent_logits": absent_logits,
            "uncertain_logits": uncertain_logits,
            "decoder_input_ids": tokenized_out.input_ids,
        }

    def forward(self, pixel_values: Optional[torch.FloatTensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.BoolTensor] = None, encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs):
        """Copy from the transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder.VisionEncoderDecoderModel.forward
        This is expected to be called only by the generate(). For training, use the forward_train().
        We remove the projection operation as we have done the projection before passing the encoder_outputs into here."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")}

        if encoder_outputs is None:
            raise ValueError("You have to provide encoder_outputs")
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # We should have done the projection before passing the encoder_outputs in this function.
        # optionally project encoder_hidden_states
        # if self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
        #     encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        """What we added is to pass the attention_mask to the decoder.prepare_inputs_for_generation func."""
        # 传入attention_mask=kwargs["decoder_attention_mask"]，则原样返回。若不传，则按照input_ids的形状，构造全1的tensor
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, attention_mask=kwargs["decoder_attention_mask"], past_key_values=past_key_values)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None

        # generate函数中会自动在decoder_attention_mask末尾加1，保证与decoder_input_ids的形状一致

        # # 将原始decoder_attention_mask中有效部分的值复制到新attention mask中
        # if decoder_attention_mask is not None:
        #     init_attention_mask = kwargs["decoder_attention_mask"]
        #     initial_length = init_attention_mask.size(1)
        #     decoder_attention_mask[:, :initial_length] = init_attention_mask

        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict


class ImageTextDataset(Dataset):

    def __init__(self, hf_dataset, img_processor, tokenizer, target_section):
        filtered_dataset, num_removed_data = self._process_text(hf_dataset, target_section)
        self.num_removed_data = num_removed_data
        self.samples = filtered_dataset

        custom_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3)])

        def preprocess_img_text(examples):
            examples["pixel_values"] = []
            for images in examples["images"]:
                # Each sample may have multiple images
                row_data = []
                for img in images:
                    # 将图像转换为灰度图像并指定输出通道数为3
                    # TODO 暂时使用pretrained model的默认图像预处理方法，似乎是用于ImageNet的：https://huggingface.co/docs/transformers/v4.40.0/en/model_doc/vit#transformers.ViTImageProcessor
                    # TODO 对于医疗图像应该需要不同的预处理方式
                    piexl_values = img_processor(custom_transform(img), return_tensors="pt").pixel_values  # (bsz=1, c, h, w)
                    piexl_values = torch.squeeze(piexl_values)  # (c, h, w)
                    row_data.append(piexl_values)
                examples["pixel_values"].append(row_data)

            # 这里的key是表示列; value是iterable (list,tensor都行)，最外层的每个元素都会被视为一行
            return examples

        # The transform function is applied on-the-fly on batches when hf_dataset.__getitem__ is called.
        filtered_dataset.set_transform(transform=preprocess_img_text)

    # 返回数据集大小
    def __len__(self):
        return len(self.samples)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.samples[index]

    def _process_text(self, hf_dataset, target_section):
        # Add id to each row
        hf_dataset = hf_dataset.add_column(name="id", column=list(range(len(hf_dataset))))

        # Remove empty string
        non_empty_section_indices = [idx for idx, txt in enumerate(hf_dataset[target_section]) if txt != ""]
        filtered_dataset = hf_dataset.select(non_empty_section_indices)
        num_removed_data = len(hf_dataset) - len(non_empty_section_indices)

        return filtered_dataset, num_removed_data


def collate_fn(batch_data, tokenizer, inference_only=False):
    """Padding the batch data, and convert list to tensor"""

    pixel_val_tensors = torch.stack([i["pixel_values"][0] for i in batch_data])

    if not inference_only:
        if "label_present" in batch_data[0]:  # train and dev
            # Create img cls gold labels (bsz, num_cls)
            label_p_tensor = batch_label_ids_to_multi_hot_tensor([i["label_present"] for i in batch_data], CONFIG.img_label_present)
            label_a_tensor = batch_label_ids_to_multi_hot_tensor([i["label_absent"] for i in batch_data], CONFIG.img_label_absent)
            label_u_tensor = batch_label_ids_to_multi_hot_tensor([i["label_uncertain"] for i in batch_data], CONFIG.img_label_uncertain)

            # get label string
            batch_label_p = [[LABELS_DICT["present"][label_idx] for label_idx in data["label_present"]] for data in batch_data]
            batch_label_a = [[LABELS_DICT["absent"][label_idx] for label_idx in data["label_absent"]] for data in batch_data]
            batch_label_u = [[LABELS_DICT["uncertain"][label_idx] for label_idx in data["label_uncertain"]] for data in batch_data]
            batch_label_text = get_batch_label_text(batch_label_p, batch_label_a, batch_label_u)

            # text longer than `max_seq_len` will be truncated to `max_seq_len`
            # Indices should be in [-100, 0, ..., config.vocab_size] (see input_ids docstring) Tokens with indices set to -100 are ignored (masked), the loss is only computed for the tokens with labels in [0, ..., config.vocab_size]
            # <s>label_text</s></s>report_section</s><pad>...
            tokenizer.padding_side = "right"
            assert tokenizer.padding_side == "right", f"tokenizer.padding_side should be [right] but got: [{tokenizer.padding_side}]"
            tokenized_out = tokenizer(batch_label_text, [i[CONFIG.target_report_section] for i in batch_data], padding=True, truncation="only_second", max_length=CONFIG.max_seq_len, return_tensors="pt")

            input_ids_tensor = tokenized_out.input_ids
            input_ids_tensor[input_ids_tensor == PAD_TOKEN_ID] = -100

            attention_mask_tensor = tokenized_out.attention_mask

            return {
                "pixel_values": pixel_val_tensors.to(DEVICE),
                "img_labels_present": label_p_tensor.long().to(DEVICE),
                "img_labels_absent": label_a_tensor.long().to(DEVICE),
                "img_labels_uncertain": label_u_tensor.long().to(DEVICE),
                "decoder_input_ids": input_ids_tensor.long().to(DEVICE),  # <s>label_text</s></s>report_section</s><pad>...
                "decoder_attention_masks": attention_mask_tensor.long().to(DEVICE),
                "gold_seq_text_list": [i[CONFIG.target_report_section] for i in batch_data],
                "label_text_list": batch_label_text,
                "data_id_list": [i["id"] for i in batch_data],
            }
        else:  # test-public
            return {
                "pixel_values": pixel_val_tensors.to(DEVICE),
                "gold_seq_text_list": [i[CONFIG.target_report_section] for i in batch_data],
                "data_id_list": [i["id"] for i in batch_data],
            }
    else:  # test-hidden
        return {
            "pixel_values": pixel_val_tensors.to(DEVICE),
            "data_id_list": [i["id"] for i in batch_data],
        }


def get_batch_label_text(batch_label_p, batch_label_a, batch_label_u):
    batch_label_text = []
    for labels_p, labels_a, labels_u in zip(batch_label_p, batch_label_a, batch_label_u):
        rm_label_str = "HIDDEN_LABEL"

        if rm_label_str in labels_p:
            labels_p.remove(rm_label_str)
            labels_p.append("others")
        p_text = ", ".join(labels_p)

        if rm_label_str in labels_a:
            labels_a.remove(rm_label_str)
            labels_a.append("others")
        a_text = ", ".join(labels_a)

        if rm_label_str in labels_u:
            labels_u.remove(rm_label_str)
            labels_u.append("others")
        u_text = ", ".join(labels_u)

        label_text = f"Observation present: [{p_text}]; absent: [{a_text}]; uncertain: [{u_text}]. Describe them in details: "
        batch_label_text.append(label_text)
    return batch_label_text


##############################################
# Train, eval
##############################################


def train(model, img_processor, tokenizer, datasets, resume_training=False):
    LOGGER.info("****************************** Loading data ******************************")
    LOGGER.info("Target report section: [%s]", CONFIG.target_report_section)
    LOGGER.info("Original train dataset size: %d, dev dataset size: %d, test dataset size: %d", len(datasets["train"]), len(datasets["validation"]), len(datasets["test"]))
    train_dataset = ImageTextDataset(hf_dataset=datasets["train"], img_processor=img_processor, tokenizer=tokenizer, target_section=CONFIG.target_report_section)
    train_sampler = StableRandomSampler(train_dataset, CONFIG.num_epoch)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=lambda batch: collate_fn(batch, tokenizer), batch_size=CONFIG.train_batch_size, drop_last=True)

    dev_dataset = ImageTextDataset(hf_dataset=datasets["validation"], img_processor=img_processor, tokenizer=tokenizer, target_section=CONFIG.target_report_section)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, collate_fn=lambda batch: collate_fn(batch, tokenizer), batch_size=CONFIG.eval_batch_size)

    test_dataset = ImageTextDataset(hf_dataset=datasets["test"], img_processor=img_processor, tokenizer=tokenizer, target_section=CONFIG.target_report_section)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=lambda batch: collate_fn(batch, tokenizer), batch_size=CONFIG.eval_batch_size)
    LOGGER.info("Num empty data removed: train: %d, dev: %d, test: %d", train_dataset.num_removed_data, dev_dataset.num_removed_data, test_dataset.num_removed_data)
    if CONFIG.max_seq_len:
        LOGGER.info("Text are truncated to max_seq_len = %d", CONFIG.max_seq_len)
    LOGGER.info("Final train samples = %d, dev samples = %d, test samples = %d", len(train_dataset), len(dev_dataset), len(test_dataset))

    log_info = {
        "train_estimated_sec": 0,
        "eval_estimated_sec": 0,
        "batch_trained_examples": 0,
        "global_steps": 0,
        "batch_loss": 0,
        "batch_text_loss": 0,
        "batch_image_loss": 0,
        "checkpoint_saved_at": 0,
        "best_model_saved_at": 0,
        "dev": {
            "best_text_scores": 0.0,
            "best_at": 0.0,
        },
        "test": {
            "best_text_scores": 0.0,
            "best_at": 0.0,
        },
    }

    model_params = list(model.named_parameters())
    assert model_params[0][0].startswith("encoder")  # check the layer name
    assert model_params[-10][0].startswith("decoder")
    assert model_params[-1][0].startswith("mlc")
    enc_doc_proj_params = [(n, p) for n, p in model_params if not n.startswith("mlc")]
    classifier_params = [(n, p) for n, p in model_params if n.startswith("mlc")]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in enc_doc_proj_params if any(nd in n for nd in no_decay)], "lr": CONFIG.lr, "weight_decay": 0.0},
        {"params": [p for n, p in enc_doc_proj_params if all(nd not in n for nd in no_decay)], "lr": CONFIG.lr, "weight_decay": CONFIG.weight_decay},
        {"params": [p for n, p in classifier_params], "lr": CONFIG.mlc_lr, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
    total_num_steps = len(train_dataloader) // CONFIG.grad_accum_steps * CONFIG.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_num_steps * CONFIG.warmup_proportion), num_training_steps=total_num_steps)

    LOGGER.info("****************************** Training ******************************")
    LOGGER.info("Total samples = %d, batch size = %d", len(train_dataset), CONFIG.train_batch_size)
    LOGGER.info("Total epochs = %d, iterations per epoch = %d", CONFIG.num_epoch, len(train_dataloader))
    LOGGER.info("Gradient accumulation steps = %d", CONFIG.grad_accum_steps)
    LOGGER.info("Total optimization steps = %d, eval per steps = %d, log per steps = %d, save per epoch = %d, ", total_num_steps, CONFIG.eval_steps, CONFIG.logging_steps, CONFIG.save_epoch)

    if resume_training:
        LOGGER.info("---------------------- Resume training from checkpoint ----------------------")
        LOGGER.info("Load checkpoint from: %s", CONFIG.output_checkpoint_dir)
        checkpoint = load_checkpoint(CONFIG.output_checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log_info = checkpoint["log_info"]
        resume_epoch = checkpoint["curr_epoch"]
        # Resume the random sampler inner-seed to the current epoch
        train_sampler.resume_to_epoch(resume_epoch)
        LOGGER.info("Resume to epoch: %d", resume_epoch)

    set_seed(CONFIG.seed)
    model.zero_grad()
    for curr_epoch in range(CONFIG.num_epoch):
        # Resuming to the next of checkpoint epoch
        if resume_training and curr_epoch <= resume_epoch:
            continue

        log_start = time.time()
        for curr_iter, batch_inputs_dict in enumerate(train_dataloader):

            model.train()
            text_loss, img_loss = model.forward_train(batch_inputs_dict)
            text_loss = text_loss * CONFIG.text_loss_weight
            img_loss = img_loss * CONFIG.image_loss_weight
            loss = text_loss + img_loss

            if CONFIG.grad_accum_steps > 1:
                loss = loss / CONFIG.grad_accum_steps
            loss.backward()

            # meta info for logging
            curr_batch_size = len(batch_inputs_dict["gold_seq_text_list"])
            log_info["batch_trained_examples"] += curr_batch_size
            log_info["batch_loss"] += loss.item() * curr_batch_size  # grad accum has been done above
            log_info["batch_text_loss"] += text_loss.item() / CONFIG.grad_accum_steps * curr_batch_size
            log_info["batch_image_loss"] += img_loss.item() / CONFIG.grad_accum_steps * curr_batch_size

            # Update model parameters
            if (curr_iter + 1) % CONFIG.grad_accum_steps == 0:
                if CONFIG.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_info["global_steps"] += 1
                TENSORBOARD.add_scalar(f"{CONFIG.output_name}/lr_enc", scheduler._last_lr[0], log_info["global_steps"])

                # logging
                if log_info["global_steps"] == 1 or log_info["global_steps"] % CONFIG.logging_steps == 0:
                    log_end = time.time()
                    sec_per_step = (log_end - log_start) / CONFIG.logging_steps
                    rest_steps = total_num_steps - log_info["global_steps"]
                    log_info["train_estimated_sec"] = sec_per_step * rest_steps
                    print_loss(curr_epoch, curr_iter, log_info)
                    log_start = time.time()

                # Eval at specific steps:
                if CONFIG.eval_steps != 0 and log_info["global_steps"] % CONFIG.eval_steps == 0:
                    eval_start = time.time()
                    LOGGER.info("--------------- Eval model at step: %d ---------------", log_info["global_steps"])
                    # dev
                    LOGGER.info("--- Dev split ---")
                    dev_out = evaluate(model, dev_dataloader, tokenizer, do_classification=True)
                    achieved_best = compare_and_print_eval_results(eval_out=dev_out, curr_epoch_or_step=log_info["global_steps"], log_info=log_info, split="dev")
                    if achieved_best:
                        LOGGER.info("[Steps=%d] Saving model to %s", log_info["global_steps"], CONFIG.output_model_dir)
                        save_model(model, CONFIG.output_model_dir)
                        log_info["best_model_saved_at"] = log_info["global_steps"]
                    # test-public
                    LOGGER.info("--- Test-public split ---")
                    test_out = evaluate(model, test_dataloader, tokenizer, do_classification=False)
                    compare_and_print_eval_results(test_out, curr_epoch_or_step=log_info["global_steps"], log_info=log_info, split="test")
                    LOGGER.info("-" * 60)
                    eval_end = time.time()
                    sec_per_eval = eval_end - eval_start
                    rest_eval = ((total_num_steps - log_info["global_steps"]) // CONFIG.eval_steps) + ((CONFIG.num_epoch - curr_epoch) // CONFIG.eval_epoch) - 1
                    log_info["eval_estimated_sec"] = sec_per_eval * rest_eval

        # Save checkpoint, we can resume it later
        if CONFIG.save_epoch != 0 and (curr_epoch + 1) % CONFIG.save_epoch == 0:
            LOGGER.info("[Epoch=%d] Saving checkpoint to: %s", curr_epoch, CONFIG.output_checkpoint_dir)
            log_info["checkpoint_saved_at"] = log_info["global_steps"]
            save_checkpoint(model, optimizer, scheduler, log_info, curr_epoch, checkpoint_dir=CONFIG.output_checkpoint_dir)

        # Eval at the end of each epoch:
        if CONFIG.eval_epoch != 0 and (curr_epoch + 1) % CONFIG.eval_epoch == 0:
            eval_start = time.time()
            LOGGER.info("--------------- Eval model at epoch: %d ---------------", curr_epoch)
            LOGGER.info("--- Dev split ---")
            dev_out = evaluate(model, dev_dataloader, tokenizer, do_classification=True)
            achieved_best = compare_and_print_eval_results(eval_out=dev_out, curr_epoch_or_step=curr_epoch, log_info=log_info, split="dev")
            if achieved_best:
                LOGGER.info("[Steps=%d] Saving model to %s", curr_epoch, CONFIG.output_model_dir)
                save_model(model, CONFIG.output_model_dir)
                log_info["best_model_saved_at"] = curr_epoch
            LOGGER.info("--- Test-public split ---")
            test_out = evaluate(model, test_dataloader, tokenizer, do_classification=False)
            compare_and_print_eval_results(test_out, curr_epoch_or_step=curr_epoch, log_info=log_info, split="test")
            LOGGER.info("-" * 60)
            eval_end = time.time()
            sec_per_eval = eval_end - eval_start
            rest_eval = ((total_num_steps - log_info["global_steps"]) // CONFIG.eval_steps) + ((CONFIG.num_epoch - curr_epoch) // CONFIG.eval_epoch) - 1
            log_info["eval_estimated_sec"] = sec_per_eval * rest_eval

    LOGGER.info("Best model saved at: %s", log_info["best_model_saved_at"])


def print_loss(curr_epoch, curr_iter, log_info):
    batch_lost_per_samples = log_info["batch_loss"] / log_info["batch_trained_examples"]
    batch_img_lost_per_samples = log_info["batch_text_loss"] / log_info["batch_trained_examples"]
    batch_text_lost_per_samples = log_info["batch_image_loss"] / log_info["batch_trained_examples"]

    LOGGER.info(
        "Epoch=%d, iter=%d, steps=%d, loss=%.9f (text=%.6f, img=%.6f) [end in %.1f hours]",
        curr_epoch,
        curr_iter,
        log_info["global_steps"],
        batch_lost_per_samples,
        batch_img_lost_per_samples,
        batch_text_lost_per_samples,
        (log_info["train_estimated_sec"] + log_info["eval_estimated_sec"]) / 3600,
    )

    ### comment this out if not using tensorboard ###
    TENSORBOARD.add_scalar(f"{CONFIG.output_name}/loss", batch_lost_per_samples, log_info["global_steps"])
    TENSORBOARD.add_scalar(f"{CONFIG.output_name}/text_loss", batch_img_lost_per_samples, log_info["global_steps"])
    TENSORBOARD.add_scalar(f"{CONFIG.output_name}/image_loss", batch_text_lost_per_samples, log_info["global_steps"])
    ###########

    log_info["batch_trained_examples"] = 0
    log_info["batch_loss"] = 0
    log_info["batch_text_loss"] = 0
    log_info["batch_image_loss"] = 0


def compare_and_print_eval_results(eval_out, curr_epoch_or_step, log_info, split="dev"):
    """Check the baseline script for the metrics chosen https://vilmedic.app/misc/bionlp24/leaderboard#anchor-baseline"""
    # Text scores
    text_scores = 0
    num_metrics = 0
    for metric_key in ["BLEU", "ROUGEL", "chexbert-all_micro avg_f1-score", "radgraph_partial", "bertscore"]:
        if metric_key in eval_out:
            TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}/{metric_key}", eval_out[metric_key], log_info["global_steps"])
            text_scores += eval_out[metric_key]
            num_metrics += 1
    text_scores = text_scores / num_metrics
    LOGGER.info(
        "(%s, curr_epoch_or_step=%d) Current TextAvgScore: %.4f, (Best: %.4f, at: %d)",
        split,
        curr_epoch_or_step,
        text_scores,
        log_info[split]["best_text_scores"],
        log_info[split]["best_at"],
    )
    TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}/TextAvgScore", text_scores, log_info["global_steps"])

    achieved_best = True if text_scores > log_info[split]["best_text_scores"] else False
    if achieved_best:
        LOGGER.info("!!! Achieved the best [%s] text avg scores!", split)
        log_info[split]["best_text_scores"] = text_scores
        log_info[split]["best_at"] = curr_epoch_or_step

    if split == "dev":
        # Img scores
        img_f1 = eval_out["present"] + eval_out["absent"] + eval_out["uncertain"]
        img_f1 = img_f1 / 3
        LOGGER.info("(%s, curr_epoch_or_step=%d) Current ImgAvgF1: %.4f", split, curr_epoch_or_step, img_f1)
        TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}-img/f1_present", eval_out["present"], log_info["global_steps"])
        TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}-img/f1_absent", eval_out["absent"], log_info["global_steps"])
        TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}-img/f1_uncertain", eval_out["uncertain"], log_info["global_steps"])
        TENSORBOARD.add_scalar(f"{CONFIG.output_name}-{split}-img/f1_avg", img_f1, log_info["global_steps"])

        return achieved_best
    else:
        return None


def evaluate(model, dataloader, tokenizer, do_classification):
    img_f1_dict = {}
    eval_results = {
        "present": {
            "num_gold": 0,
            "num_pred": 0,
            "num_corr": 0,
        },
        "absent": {
            "num_gold": 0,
            "num_pred": 0,
            "num_corr": 0,
        },
        "uncertain": {
            "num_gold": 0,
            "num_pred": 0,
            "num_corr": 0,
        },
    }

    LOGGER.info("Batch size = %d", CONFIG.eval_batch_size)
    LOGGER.info("Num samples = %d", len(dataloader.dataset))

    gold_texts = []
    generated_texts = []

    model.eval()
    with torch.no_grad():
        for batch_inputs_dict in dataloader:
            output = model.do_generate(batch_inputs_dict, tokenizer=tokenizer)

            # text generation
            gold_texts.extend(batch_inputs_dict["gold_seq_text_list"])
            generated_ids = output["generated_ids_without_label_text"]
            generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

            # img cls
            if do_classification:
                for label_type in ["present", "absent", "uncertain"]:
                    threshold = CONFIG.threshold
                    probs = torch.sigmoid(output[f"{label_type}_logits"])
                    batch_pred_labels = torch.where(probs > threshold, 1, 0).cpu().numpy()
                    batch_gold_labels = batch_inputs_dict[f"img_labels_{label_type}"].cpu().numpy()
                    for gold_labels, pred_labels in zip(batch_gold_labels, batch_pred_labels):
                        for gold, pred in zip(gold_labels, pred_labels):
                            if gold == 1:
                                eval_results[label_type]["num_gold"] += 1
                            if pred == 1:
                                eval_results[label_type]["num_pred"] += 1
                            if gold == 1 and pred == 1:
                                eval_results[label_type]["num_corr"] += 1

    # Eval the text generation results
    text_scores_dict = compute_generation_score(gold_text_list=gold_texts, generated_text_list=generated_texts)
    LOGGER.info("[TextGen]: %s", json.dumps(text_scores_dict))

    if do_classification:
        # Evaluate the img classification results
        for eval_field, result_dict in eval_results.items():
            num_corr = result_dict["num_corr"]
            num_pred = result_dict["num_pred"]
            num_gold = result_dict["num_gold"]
            p = num_corr / num_pred if num_corr > 0 else 0.0
            r = num_corr / num_gold if num_corr > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if num_corr > 0 else 0.0
            LOGGER.info("[ImgCls %s]: P: %.5f, R: %.5f, 【F1: %.3f】", eval_field, p, r, f1 * 100)
            img_f1_dict[eval_field] = f1
        return {**text_scores_dict, **img_f1_dict}
    else:
        return text_scores_dict


def compute_generation_score(gold_text_list, generated_text_list):
    """Based on the script from https://vilmedic.app/misc/bionlp24/leaderboard#anchor-baseline"""
    if DEVICE.type == "cpu":
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert"]
    else:
        use_metrics = ["BLEU", "ROUGEL", "radgraph", "chexbert", "bertscore"]

    # https://github.com/jbdel/vilmedic/blob/main/vilmedic/blocks/scorers/scores.py
    out_dict = compute_scores(use_metrics, refs=[" ".join(wordpunct_tokenize(s.lower())) for s in gold_text_list], hyps=generated_text_list, split=None, seed=None, config=None, epoch=None, logger=LOGGER, dump=False)
    return out_dict


def inference(model, dataloader, tokenizer, num_data, output_name):

    # generated_text_dict = {}
    generated_text_list = []

    model.eval()
    with torch.no_grad():
        for batch_inputs_dict in tqdm(dataloader):
            output = model.do_generate(batch_inputs_dict, tokenizer=tokenizer)

            # text generation
            batch_generated_ids = output["generated_ids_without_label_text"]
            batch_generated_texts = tokenizer.batch_decode(batch_generated_ids, skip_special_tokens=True)
            batch_data_id_list = batch_inputs_dict["data_id_list"]

            generated_text_list.extend(batch_generated_texts)

            # for idx, generated_text in zip(batch_data_id_list, batch_generated_texts):
            #     generated_text_dict[idx] = generated_text

    output_path = os.path.join(CONFIG.output_result_dir, f"{output_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(generated_text_list))
    LOGGER.info("Output file path: %s", output_path)


##############################################
# settings & utils
##############################################


def batch_label_ids_to_multi_hot_tensor(batch_label_ids, num_classes):
    bsz = len(batch_label_ids)
    multi_hot_tensor = torch.zeros(bsz, num_classes)
    for row_idx, label_indices in enumerate(batch_label_ids):
        multi_hot_tensor[row_idx, label_indices] = 1
    return multi_hot_tensor  # (bsz, num_classes)


def init_model(encoder_path, decoder_path, tokenizer):
    # # init encoder
    # encoder_config = AutoConfig.from_pretrained(encoder_path)
    # encoder_config.is_decoder = False
    # encoder_config.add_cross_attention = False
    # encoder = Swinv2Model.from_pretrained(encoder_path, config=encoder_config, add_pooling_layer=False)

    # # init decoder
    # decoder_config = AutoConfig.from_pretrained(decoder_path)
    # decoder_config.is_decoder = True
    # decoder_config.add_cross_attention = True
    # decoder = T5ForConditionalGeneration.from_pretrained(decoder_path, config=decoder_config)

    # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    # vis_enc_dec = Vision2LanguageModel(encoder=encoder, decoder=decoder, config=config)

    # vis_enc_dec.config.decoder_start_token_id = tokenizer.cls_token_id
    # vis_enc_dec.config.pad_token_id = tokenizer.pad_token_id
    # -------------

    # model.from_encoder_decoder_pretrained(**kwargs): use the prefix *encoder_*, *decoder_* for each model configuration parameter; to update the parent model configuration, do not use a prefix for each configuration parameter.
    # Doc here: https://github.com/huggingface/transformers/blob/745bbfe4bb2b61491dedd56e1e8ee4af8ef1a9ec/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L362
    # self.vis_enc_dec_model.config.tie_encoder_decoder is set to false in the baseline model
    # the encoder_last_hidden_state is irrelevant to the pooling layer whether or not it is added.
    vis_enc_dec = Vision2LanguageModel.from_encoder_decoder_pretrained(encoder_path, decoder_path, encoder_add_pooling_layer=False)

    # Bart uses the eos_token_id as the starting token for decoder_input_ids generation.
    if "bart" in vis_enc_dec.decoder.name_or_path:
        vis_enc_dec.config.decoder_start_token_id = tokenizer.eos_token_id
    elif "roberta" in vis_enc_dec.decoder.name_or_path:
        vis_enc_dec.config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        vis_enc_dec.config.decoder_start_token_id = tokenizer.bos_token_id
    vis_enc_dec.config.pad_token_id = tokenizer.pad_token_id

    return vis_enc_dec


def load_model(model_path):
    # load the fine-tuned vis_enc_dec
    vis_enc_dec = Vision2LanguageModel.from_pretrained(model_path)
    # Create a new model with the fine-tuned vis_enc_dec;
    # model = Vision2LanguageModel(vis_enc_dec)
    # # update the model state dict with other components' state dict
    # sd = model.state_dict()
    # auxiliary_layers_sd_path = os.path.join(model_path, "auxiliary_layers.pth")
    # if not os.path.exists(auxiliary_layers_sd_path):
    #     raise FileNotFoundError(f"The required `auxiliary_layers.pth` is not found in {auxiliary_layers_sd_path}")
    # sd.update(torch.load(auxiliary_layers_sd_path))
    # model.load_state_dict(sd)
    return vis_enc_dec


def save_model(model, output_dir):
    """
    Save the model to the output directory; the vis2lan model and the auxiliary classifiers are saved separately
    """

    # saving encoder to the same model as tokenizer
    model.save_pretrained(output_dir)


def save_checkpoint(model, optimizer, scheduler, log_info, curr_epoch, checkpoint_dir):
    """https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "log_info": log_info,
        "curr_epoch": curr_epoch,
    }
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.tar")
    torch.save(checkpoint, checkpoint_path)

    with open(os.path.join(checkpoint_dir, "checkpoint.log"), "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "curr_epoch": curr_epoch,
                    **log_info,
                },
                indent=4,
            )
        )


def load_checkpoint(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.tar")
    return torch.load(checkpoint_path)


##############################################
# Script arguments
##############################################


def init_logger():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.WARNING)

    # Set logging file_handler
    if CONFIG.do_train:
        log_file_mode = "w"
        if CONFIG.resume_training:
            log_file_mode = "a"
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_result_dir, "train.log"), log_file_mode)
    # elif CONFIG.do_pred:
    #     file_handler = logging.FileHandler(os.path.join(CONFIG.output_result_dir, "pred.log"), "w")
    elif CONFIG.do_inference:
        file_handler = logging.FileHandler(os.path.join(CONFIG.output_result_dir, "inference.log"), "w")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # logger for curr module
    logger = logging.getLogger("vis2lan")
    logger.setLevel(logging.DEBUG)  # Curr file logger is set to debug
    logger.addHandler(file_handler)

    # Set other module logger
    util_logger = logging.getLogger("utils")
    util_logger.setLevel(logging.DEBUG)
    util_logger.addHandler(file_handler)

    # Set other module logger
    util_logger = logging.getLogger("load_dataset")
    util_logger.setLevel(logging.DEBUG)
    util_logger.addHandler(file_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume_training", action="store_true", default=False)

    parser.add_argument("--from_bash", action="store_true")
    parser.add_argument("--output_name", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.debug:
        CONFIG.resume_training = args.resume_training
    if args.from_bash:
        CONFIG.output_name = args.output_name

    # 1. Reproducibility
    set_seed(CONFIG.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Create output dir
    CONFIG.output_result_dir = os.path.join(CONFIG.output_result_dir, CONFIG.output_name)
    CONFIG.output_model_dir = os.path.join(CONFIG.output_model_dir, CONFIG.output_name)
    CONFIG.output_checkpoint_dir = os.path.join(CONFIG.output_checkpoint_dir, CONFIG.output_name)
    os.makedirs(CONFIG.output_result_dir, exist_ok=True)
    os.makedirs(CONFIG.output_model_dir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)

    # 3. Set logger
    LOGGER = init_logger()
    LOGGER.info([i for i in vars(CONFIG).items() if i[0][0] != "_"])
    TENSORBOARD = SummaryWriter(os.path.join(CONFIG.output_result_dir, "log_tensorboard"))

    # 4. Load label strings
    with open(CONFIG.data_path["img_label_string"], "r", encoding="utf-8") as f:
        LABELS_DICT = json.loads(f.readline())

    # 4. Main process
    start = time.time()

    if CONFIG.do_train:
        # 4. Initial model
        LOGGER.info("****************************** Initialize model ******************************")
        LOGGER.info("Loading ImageProcessor from: %s", CONFIG.vision_model_path)
        img_processor = ViTImageProcessor.from_pretrained(CONFIG.vision_model_path)
        LOGGER.info("Loading Tokenizer from: %s", CONFIG.language_model_path)
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.language_model_path)
        PAD_TOKEN_ID = tokenizer.pad_token_id

        if CONFIG.load_from_pretrained:
            LOGGER.info("Loading pre-trained vis-enc-dec model from: %s", CONFIG.load_from_pretrained)
            v2l_model = load_model(CONFIG.load_from_pretrained)
        else:
            LOGGER.info("Loading Encoder from: %s", CONFIG.vision_model_path)
            LOGGER.info("Loading Decoder from: %s", CONFIG.language_model_path)
            v2l_model = init_model(encoder_path=CONFIG.vision_model_path, decoder_path=CONFIG.language_model_path, tokenizer=tokenizer)

        v2l_model.to(DEVICE)
        LOGGER.info("Device: %s", DEVICE)

        LOGGER.info("Loading source data from: %s, %s. Labels from: %s", CONFIG.data_path["interpret"], CONFIG.data_path["mimic"], CONFIG.data_path["img_labels"])
        train_dev_ds = get_dataset(interpret_cxr_dir=CONFIG.data_path["interpret"], mimic_cxr_dir=CONFIG.data_path["mimic"], label_file_path=CONFIG.data_path["img_labels"])

        test_ds = load_from_disk(CONFIG.data_path["interpret-test-public"])
        hf_datasets = DatasetDict({"train": train_dev_ds["train"], "validation": train_dev_ds["validation"], "test": test_ds["test"]})

        train(model=v2l_model, img_processor=img_processor, tokenizer=tokenizer, datasets=hf_datasets, resume_training=CONFIG.resume_training)

    if CONFIG.do_inference:
        img_processor = ViTImageProcessor.from_pretrained(CONFIG.vision_model_path)
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.language_model_path)
        PAD_TOKEN_ID = tokenizer.pad_token_id

        if CONFIG.load_from_pretrained:
            LOGGER.info("Loading pre-trained vis-enc-dec model from: %s", CONFIG.load_from_pretrained)
            v2l_model = load_model(CONFIG.load_from_pretrained)
        else:
            LOGGER.info("Loading pre-trained vis-enc-dec model from: %s", CONFIG.output_model_dir)
            v2l_model = load_model(CONFIG.output_model_dir)

        v2l_model.config.decoder_start_token_id = tokenizer.cls_token_id
        v2l_model.config.pad_token_id = tokenizer.pad_token_id
        v2l_model.to(DEVICE)

        LOGGER.info("Loading source data from: %s", CONFIG.data_path["interpret-test-public"])
        test_public = load_from_disk(CONFIG.data_path["interpret-test-public"])
        test_dataset = ImageTextDataset(hf_dataset=test_public["test"], img_processor=img_processor, tokenizer=tokenizer, target_section=CONFIG.target_report_section)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=lambda batch: collate_fn(batch, tokenizer), batch_size=CONFIG.eval_batch_size)
        LOGGER.info("Doing inference...")
        inference(v2l_model, test_dataloader, tokenizer, num_data=len(test_dataset), output_name="inference_test_public")

        LOGGER.info("Loading source data from: %s", CONFIG.data_path["interpret-test-hidden"])
        test_hidden = load_from_disk(CONFIG.data_path["interpret-test-hidden"])
        test_dataset = ImageTextDataset(hf_dataset=test_hidden["test"], img_processor=img_processor, tokenizer=tokenizer, target_section=CONFIG.target_report_section)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=lambda batch: collate_fn(batch, tokenizer), batch_size=CONFIG.eval_batch_size)
        LOGGER.info("Doing inference...")
        inference(v2l_model, test_dataloader, tokenizer, num_data=len(test_dataset), output_name="inference_test_hidden")

    end = time.time()
    LOGGER.info("Time: %d hours", (end - start) / 360)
    TENSORBOARD.close()
