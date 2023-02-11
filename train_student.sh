if [ "$#" -ne 1 ]; then
    echo "Usage: ./train_student.sh <tag>"
    exit
fi
tag=$1
python train_student_encoder.py \
teacher_model_file=/home/cc/code/catalog/dpr/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp \
train=biencoder_nq \
train_datasets=[nq_train_hn1] \
dev_datasets=[nq_dev] \
output_dir=student_encoder_${tag} \

