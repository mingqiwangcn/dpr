if [ "$#" -ne 2 ]; then
    echo "Usage: ./eval_student.sh <eval_mode> <model_file>"
    exit
fi
eval_mode=$1
model=$2
python train_student_encoder.py \
train=biencoder_nq \
eval_type=${eval_mode} \
model_file=${model} \
dev_datasets=[nq_dev] \
output_dir=student_eval \
ignore_checkpoint_optimizer=True \
