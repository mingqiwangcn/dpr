if [ "$#" -ne 1 ]; then
    echo "Usage: ./train_student.sh <tag>"
    exit
fi
tag=$1
python train_student_encoder.py \
teacher_is_ta=false \
teacher_model_file=/home/cc/code/catalog/dpr/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
student_layers=[1,2,3,4,5,6,7,8,9,10,11] \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=student_encoder_${tag} \
score_temperature=6 \
