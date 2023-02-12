python train_student_encoder.py \
train=biencoder_nq \
ta_layers=1 \
model_file=/home/cc/code/catalog/dpr/trained_models/student_8/dpr_student_biencoder.12 \
dev_datasets=[nq_dev] \
output_dir=student_eval \
ignore_checkpoint_optimizer=True \
