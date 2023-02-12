if [ "$#" -ne 1 ]; then
    echo "Usage: ./train_student.sh <tag>"
    exit
fi
tag=$1
python train_student_encoder.py \
teacher_is_ta=true \
ta_layers=3 \
teacher_model_file=/home/cc/code/catalog/dpr/outputs/2023-02-10/22-48-10/student_encoder_keep_0_6_11/dpr_student_biencoder.18 \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=student_encoder_${tag} \

