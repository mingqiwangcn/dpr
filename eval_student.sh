python train_student_encoder.py \
train=biencoder_nq \
eval_type=rank \
model_file=/home/cc/code/catalog/dpr/outputs/2023-02-14/16-49-38/student_encoder_test/dpr_student_biencoder.26 \
dev_datasets=[nq_dev] \
output_dir=student_eval \
ignore_checkpoint_optimizer=True \
