if [ "$#" -ne 4 ]; then
    echo "Usage: ./train_student.sh <tag> <teacher_is_ta> <teacher_model_file> <student layers>"
    exit
fi
tag=$1
is_ta=$2
teacher_model=$3
layers=$4
python train_student_encoder.py \
teacher_is_ta=${is_ta} \
teacher_model_file=${teacher_model} \
student_layers=${layers} \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=student_encoder_${tag} \
score_temperature=6 \
