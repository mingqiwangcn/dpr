if [ "$#" -ne 3 ]; then
    echo "Usage: ./retrieve.sh <indexer> <factory string> <tag>"
    exit
fi

indexer=$1
fac_str=$2
tag=$3
is_teacher=true
model=/home/cc/code/catalog/dpr/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp
python dense_retriever.py \
is_teacher=${is_teacher} \
model_file=${model} \
indexer=${indexer} \
factory_string=\"${fac_str}\" \
qa_dataset=nq_test \
ctx_datatsets=[dpr_wiki] \
encoded_ctx_files=[\"/home/cc/code/catalog/dpr/passage_embs/teacher_emb_recomputed/wiki_emb_0_part_*\"] \
out_file=retrieval_${tag}.json \

