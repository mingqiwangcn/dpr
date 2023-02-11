python train_dense_encoder.py \
train=biencoder_nq \
model_file=/home/cc/code/catalog/dpr/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
dev_datasets=[nq_dev] \
output_dir=teacher_eval \
ignore_checkpoint_optimizer=True \
