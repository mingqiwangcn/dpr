from fabric_qa.ir_ranker import IRRanker
from tqdm import tqdm
import json
import numpy as np

def main():
    eshost = dict()
    eshost['host'] = '127.0.0.1'
    eshost['port'] = 9200
    eshost_lst = [eshost]
    ir_ranker = IRRanker(hosts=eshost_lst)
    
    top_k_lst = [20, 50, 100, 150, 200]
    metric_dict = {}
    for top_k in top_k_lst:
        metric_dict[top_k] = []

    with open('./nq_bm25_questions.jsonl') as f:
        for line in tqdm(f):
            item = json.loads(line)
            gold_p_id = item['gold_passage_id']
            res = ir_ranker.search(
                index_name='nq_rank',
                question=item['question'],
                k=200,
                ret_src=True, 
            )
            p_id_lst = [a['_source']['p_id'] for a in res] 
            for top_k in top_k_lst:
                found = int(gold_p_id in p_id_lst[:top_k])
                metric_dict[top_k].append(found)
    
    report = 'BM25' 
    for top_k in top_k_lst:
        top_k_acc = np.mean(metric_dict[top_k]) * 100
        report += '  R@%d = %.2f' % (top_k, top_k_acc)
    print(report) 

if __name__ == '__main__':
    main() 
