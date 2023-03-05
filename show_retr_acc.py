import json
import numpy as np
from tqdm import tqdm
import pandas as pd

def show_acc(tag, rank_file, top_k_lst):
    res_dict = {}
    for top_k in top_k_lst:
        res_dict[top_k] = []

    with open(rank_file) as f:
        for line in f:
            item = json.loads(line)
            rank = item['rank']
            for top_k in top_k_lst:
                res_dict[top_k].append(int(rank < top_k)) # rank is 0 based 
    report = '%9s' % tag 
    for top_k in top_k_lst:
        top_k_acc = np.mean(res_dict[top_k]) * 100
        report += '  R@%d = %.2f' % (top_k, top_k_acc)
    print(report)  


def read_nll_score(data_file):
    num_correct = 0
    num_total = 0
    correct_set = set()
    score_dict = {}
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            ctxs = item['ctxs']
            query_lst = item['queries']
            for query in query_lst:
                query['ctxs'] = ctxs
                score_dict[query['index']] = query
                num_total += 1
                pos_index = query['pos_ctx_index']
                scores = query['scores']
                best_idx = np.argmax(scores)
                if best_idx == pos_index:
                    num_correct += 1
                    correct_set.add(query['index'])
                
    print('correct ratio = %d/%d = %f' % (num_correct, num_total, num_correct / num_total))
    return correct_set, score_dict


def cmp_teacher_student():
    correct_teacher_set, teacher_score_dict = read_nll_score('./outputs/2023-02-28/21-32-27/scores.jsonl') 
    correct_student_set, student_score_dict = read_nll_score('./outputs/2023-02-28/21-21-08/scores.jsonl') 
    
    teacher_1_student_0 = correct_teacher_set - correct_student_set  
    teacher_0_student_1 = correct_student_set - correct_teacher_set
    
    print('teacher_1_student_0 = %d' % len(teacher_1_student_0))
    print('teacher_0_student_1 = %d' % len(teacher_0_student_1))

    write_queries('teacher_1_student_0.csv', teacher_1_student_0, teacher_score_dict, student_score_dict)
    write_queries('teacher_0_student_1.csv', teacher_0_student_1, teacher_score_dict, student_score_dict)

def get_ranks(arg_sort):
    rank_dict = {}
    for rank in range(len(arg_sort)):
        rank_dict[arg_sort[rank]] = rank
    return rank_dict

def write_queries(out_file, index_lst, teacher_score_dict, student_score_dict):
    col_names = ['query_index', 'query', 'pos_ctx_index', 'ctx_index', 'ctx', 
                 'teacher_score', 'teacher_rank', 'student_score', 'student_rank']
    out_data = []
    log_qry_count = 0
    for index in index_lst:
        query = student_score_dict[index]
        pos_ctx_index = query['pos_ctx_index']
        out_qry_item = [query['index'], query['query'], pos_ctx_index]
        student_scores = query['scores']
        student_ranks = get_ranks(query['arg_sort'])
        
        teacher_query = teacher_score_dict[index]
        teacher_scores = teacher_query['scores']
        teacher_ranks = get_ranks(teacher_query['arg_sort'])
        
        rank_diff = abs(teacher_ranks[pos_ctx_index] - student_ranks[pos_ctx_index])
        
        if rank_diff < 5:
            continue
        log_qry_count += 1
        out_data.append(out_qry_item)
        ctx_lst = query['ctxs']
        for ctx_index, ctx in enumerate(ctx_lst):
            out_ctx_item = [
                '', '', '', ctx_index, ctx,
                teacher_scores[ctx_index], teacher_ranks[ctx_index],
                student_scores[ctx_index], student_ranks[ctx_index],
            ] 
            out_data.append(out_ctx_item) 
        
    print('log_qry_count =', log_qry_count) 
    df = pd.DataFrame(out_data, columns=col_names)
    df.to_csv(out_file)

def main():
    #cmp_teacher_student() 
    show_rank_accuracy()

def show_rank_accuracy():
    top_k_lst = [20, 50, 100, 150, 200]
   
    student_rank_file = './outputs/2023-03-04/00-49-01/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)

if __name__ == '__main__':
    main()
