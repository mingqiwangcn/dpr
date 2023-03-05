if [ "$#" -ne 3 ]; then
    echo "Usage: ./retrieve.sh <indexer> <factory string> <tag>"
    exit
fi

indexer=$1
fac_str=$2
tag=$3
is_teacher=false
model=/home/cc/code/catalog/dpr/trained_models/dpr_student_precom_16_biencoder.71
python dense_retriever.py \
is_teacher=${is_teacher} \
model_file=${model} \
indexer=${indexer} \
factory_string=\"${fac_str}\" \
qa_dataset=nq_test \
ctx_datatsets=[dpr_wiki] \
encoded_ctx_files=[\"/home/cc/code/catalog/dpr/passage_embs/student_1_layer_16_neg_embs/wiki_emb_0_part_*\"] \
out_file=retrieval_${tag}.json \

