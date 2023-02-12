python train_student_encoder.py \
train=biencoder_nq \
ta_layers=3 \
model_file=/home/cc/code/catalog/dpr/trained_models/ta_student_4/dpr_student_biencoder.18 \
dev_datasets=[nq_dev] \
output_dir=ta_eval \
ignore_checkpoint_optimizer=True \
