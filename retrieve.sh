if [ "$#" -ne 3 ]; then
    echo "Usage: ./retrieve.sh <is_teacher> <model_file> <indexer>"
    exit
fi
is_teacher=$1
model=$2
indexer=$3
python dense_retriever.py \
is_teacher=${is_teacher} \
model_file=${model} \
indexer=${indexer} \
qa_dataset=nq_test \
ctx_datatsets=[dpr_wiki] \
encoded_ctx_files=[\"/home/cc/code/dpr/passage_embs/student_1_layer_16_neg_embs/wiki_emb_0_part_*\"] \
out_file=output_retrieval.json \
