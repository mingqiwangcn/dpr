#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List

import torch
import transformers
from torch import Tensor as T
from torch import nn


if transformers.__version__.startswith("4"):
    from transformers import BertConfig, BertModel
    from transformers import AdamW
    from transformers import BertTokenizerFast
    from transformers import RobertaTokenizer
else:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizerFast
    from transformers.tokenization_roberta import RobertaTokenizer

from dpr.utils.data_utils import Tensorizer
from dpr.models.student_biencoder import StudentBiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)


def get_student_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    teacher = kwargs.get('teacher', None)
    if teacher is not None:
        question_teacher_encoder = teacher.model.question_model
        ctx_teacher_encoder = teacher.model.ctx_model
    else:
        teacher = None # the student is a TA
        question_teacher_encoder = None
        ctx_teacher_encoder = None

    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = StudentHFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        teacher_encoder=question_teacher_encoder,
        encoder_name='question',
        **kwargs
    )
    student_layers = None
    if cfg.student_layers is not None:
        student_layers = [a for a in cfg.student_layers] 
    ctx_encoder = StudentHFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        teacher_encoder=ctx_teacher_encoder,
        student_layers=student_layers,
        encoder_num_layers=kwargs.get('student_num_layers', None),
        encoder_name='passage',
        **kwargs
    )

    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    biencoder = StudentBiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder, teacher=teacher)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


# TODO: unify tensorizer init methods
def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


def get_bert_tensorizer_p(
    pretrained_model_cfg: str, sequence_length: int, do_lower_case: bool = True, special_tokens: List[str] = []
):
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    if special_tokens:
        _add_special_tokens(tokenizer, special_tokens)
    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("Tokenizer: %s", type(tokenizer))
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code

    assert special_tokens_num < 500
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)


def get_roberta_tensorizer(pretrained_model_cfg: str, do_lower_case: bool, sequence_length: int):
    tokenizer = get_roberta_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    return RobertaTensorizer(tokenizer, sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizerFast.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


class StudentHFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config, add_pooling_layer=False)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()
    
    @staticmethod
    def prune_layers(model, student_layers):
        assert (student_layers is not None) and (len(student_layers) > 0)
        updated_layer = nn.ModuleList()
        for idx, module in enumerate(model.encoder.layer):
            if idx in student_layers:
                updated_layer.append(module)
        model.encoder.layer = updated_layer
    
    @staticmethod
    def set_layers(model, num_layers):
        updated_layer = nn.ModuleList()         
        for idx, module in enumerate(model.encoder.layer):
            if idx < num_layers:
                updated_layer.append(module)
        model.encoder.layer = updated_layer

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, 
        teacher_encoder=None, student_layers=None, encoder_num_layers=None, encoder_name=None,  **kwargs
    ) -> BertModel:
        logger.info("Initializing Student HF BERT Encoder. cfg_name=%s", cfg_name)
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # right now, student has one layer less than the teacher.
        #if pretrained:
        #    model = cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        #else:
        #    model = StudentHFBertEncoder(cfg, project_dim=projection_dim)
        
        model = StudentHFBertEncoder(cfg, project_dim=projection_dim)
        if teacher_encoder is None:
            # This is when the model is in evaluation mode.
            if encoder_num_layers is not None: 
                StudentHFBertEncoder.set_layers(model, encoder_num_layers)
        else:
            # Need to update model layers as the same as the teacher to copy weights
            num_teacher_layer = len(teacher_encoder.encoder.layer)
            num_bert_layer = len(model.encoder.layer)
            if num_bert_layer > num_teacher_layer:
                StudentHFBertEncoder.set_layers(model, num_teacher_layer)
            # Always copy teacher model weights and then prune 
            model.load_state_dict(teacher_encoder.state_dict()) 
            if student_layers is not None:
                StudentHFBertEncoder.prune_layers(model, student_layers)
       
        if student_layers is None: 
            logger.info('%s %s encoder layers = %d' % (kwargs['tag'], encoder_name, len(model.encoder.layer)) )
        else:
            if kwargs['tag'] == 'TA':
                logger.info('%s %s encoder layers = %d' % (kwargs['tag'], encoder_name, len(model.encoder.layer))) 
            else:
                logger.info('%s %s encoder layers = %d, %s' % (kwargs['tag'], encoder_name, len(model.encoder.layer), 
                        str(student_layers)))
        return model

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizerFast, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)
