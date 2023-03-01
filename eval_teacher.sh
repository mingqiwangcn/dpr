if [ "$#" -ne 1 ]; then
    echo "Usage: ./eval_teacher.sh <eval_mode>"
    exit
fi
eval_mode=$1
python train_dense_encoder.py \
train=biencoder_nq \
eval_type=${eval_mode} \
model_file=/home/cc/code/catalog/dpr/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
dev_datasets=[nq_dev] \
output_dir=teacher_eval \
ignore_checkpoint_optimizer=True \
do_log_score=True \
