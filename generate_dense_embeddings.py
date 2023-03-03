#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import threading
import queue

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components, get_bert_tensorizer
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
    get_ctx_model_layers,
)

logger = logging.getLogger()
setup_logger(logger)

queue_token_tensor = queue.Queue()
queue_output = queue.Queue()
queue_output_stat = queue.Queue()

def tok_worker(start_idx, end_idx):
    worker_tensorizer = get_bert_tensorizer(cfg)
    idx = start_idx
    bsz = cfg.batch_size
    insert_title = True
    count = 0
    for batch_start in range(start_idx, end_idx, bsz):
        batch_end = min(batch_start + bsz, end_idx)
        batch = ctx_data[batch_start : batch_end]
        batch_token_tensors = [
            worker_tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]
        ctx_ids = [r[0] for r in batch]
        
        out_item = [ctx_ids, batch_token_tensors]
        queue_token_tensor.put(out_item)
        count += len(ctx_ids)


def start_tok_threading():
    num_rows = len(ctx_data)
    num_workers = cfg.num_tok_workers
    part_size = num_rows // num_workers

    start_idx = 0
    for w_idx in range(num_workers):
        if w_idx < (num_workers - 1):
            end_idx = start_idx + part_size
        else:
            end_idx = num_rows
        
        threading.Thread(target=tok_worker, args=(start_idx, end_idx, )).start()
        start_idx = end_idx

def output_worker(part_idx):
    data = queue_output.get()
    file = cfg.out_file + "_" + str(cfg.shard_id) + "_part_" + str(part_idx) 
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info("Writing part %d with size %d results to %s" % (part_idx, len(data), file))
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    queue_output_stat.put([part_idx, len(data), file])

def start_output_threading(results, part_idx):
    queue_output.put(results)
    threading.Thread(target=output_worker, args=(part_idx, )).start()
     

def gen_ctx_vectors(
    #cfg: DictConfig,
    #ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    #insert_title: bool = True,
):
    
    start_tok_threading()

    num_ctx_rows = len(ctx_data)
    total = 0
    results = []
    output_part_idx = 0
    while True:
        item = queue_token_tensor.get()
        batch_token_tensors = item[1] 

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()
        
        ctx_ids = item[0]
        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        if total % 100 == 0:
            logger.info("Encoded passages %d", total)
        
        if len(results) >= cfg.output_batch_size:
            start_output_threading(results, output_part_idx)
            output_part_idx += 1
            results = []

        if total == num_ctx_rows:
            break 

    if len(results) > 0:
        start_output_threading(results, output_part_idx)
        output_part_idx += 1
        results = []
    
    return output_part_idx

@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg_data: DictConfig):

    assert cfg_data.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg_data.ctx_src, "Please specify passages source as ctx_src param"

    global cfg
    cfg = setup_cfg_gpu(cfg_data)

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    student_num_layers = None
    tag = ''
    if cfg.is_teacher:
        model_type = cfg.encoder.teacher_encoder_model_type
    else:
        model_type = cfg.encoder.student_encoder_model_type
        student_num_layers = get_ctx_model_layers(saved_state.model_dict)
        tag = 'Student'
         
    tensorizer, encoder, _ = init_biencoder_components(model_type, cfg, 
                                                       student_num_layers=student_num_layers, 
                                                       tag=tag, inference_only=True)

    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    logger.info("reading data source: %s", cfg.ctx_src)

    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    global ctx_data
    ctx_data = all_passages[start_idx:end_idx]
    num_output_parts = gen_ctx_vectors(encoder, tensorizer) #, True)
    show_output_stat(num_output_parts) 


def show_output_stat(num_output_parts):
    num_part = 0
    output_size = 0
    while True:
        out_stat = queue_output_stat.get()
        part_idx = out_stat[0]
        part_size = out_stat[1]
        out_file = out_stat[2]
        logger.info("Passages part %d processed %d. Written to %s", part_idx, part_size, out_file)
        num_part += 1
        output_size += out_stat[1]
        if num_part == num_output_parts:
            break
    logger.info("Total passages processed %d.", output_size)  

if __name__ == "__main__":
    main()
