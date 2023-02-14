if [ "$#" -ne 3 ]; then
    echo "Usage: ./train_student.sh <tag> <teacher_model_file> <student layers>"
    exit
fi
tag=$1
teacher_model=$2
layers=$3
python train_student_encoder.py \
teacher_is_ta=false \
teacher_model_file=${teacher_model} \
student_layers=${layers} \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=student_encoder_${tag} \
score_temperature=6 \
