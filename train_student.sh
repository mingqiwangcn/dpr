if [ "$#" -ne 5 ]; then
    echo "Usage: ./train_student.sh <teacher_name> <teacher_is_ta> <teacher_model_file> <student_name> <student layers>"
    exit
fi

teacher_name=$1
is_ta=$2
teacher_model=$3
student_name=$4
layers=$5

python train_student_encoder.py \
teacher_is_ta=${is_ta} \
teacher_name=${teacher_name} \
teacher_model_file=${teacher_model} \
student_layers=${layers} \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=encoder_${teacher_name}_${student_name} \
score_temperature=6 \
