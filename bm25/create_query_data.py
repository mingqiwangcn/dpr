import json
from tqdm import tqdm

def main():
    f_o_questions = open('./nq_bm25_questions.jsonl', 'w')
    f_o_passages = open('./nq_bm25_passages.jsonl', 'w')
    with open('./nq_rank_question_passages.jsonl') as f:
        for line in tqdm(f):
            item = json.loads(line)
            qid = item['qid']
            question = item['question']
            passage_lst = item['passages'] 
            p_id_lst = item['passage_ids']
            assert(len(passage_lst) == len(p_id_lst)) 
            for offset, passage in enumerate(passage_lst):
                out_passage_item = {
                    'p_id':p_id_lst[offset],
                    'passage':passage,
                }
                f_o_passages.write(json.dumps(out_passage_item) + '\n')
            
            out_question_item = {
                'qid':qid,
                'question':question,
                'gold_passage_id':p_id_lst[0],
            }
            f_o_questions.write(json.dumps(out_question_item) + '\n')

    f_o_questions.close()
    f_o_passages.close()

if __name__ == '__main__':
    main()
