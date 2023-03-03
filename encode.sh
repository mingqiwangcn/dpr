if [ "$#" -ne 2 ]; then
    echo "Usage: ./encode.sh <is_teacher> <model_file>"
    exit
fi
is_teacher=$1
model=$2
python generate_dense_embeddings.py \
is_teacher=${is_teacher} \
model_file=${model} \
ctx_src=dpr_wiki \
batch_size=32 \
out_file=wiki_emb \
num_tok_workers=1 \
