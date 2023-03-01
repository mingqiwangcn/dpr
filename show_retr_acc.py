import json
import numpy as np
from tqdm import tqdm

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
    with open(data_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            ctxs = item['ctxs']
            query_lst = item['queries']
            for query in query_lst:
                num_total += 1
                pos_index = query['pos_ctx_index']
                scores = query['scores']
                best_idx = np.argmax(scores)
                if best_idx == pos_index:
                    num_correct += 1
                    correct_set.add(query['index'])

    print('correct ratio = %d/%d = %f' % (num_correct, num_total, num_correct / num_total))
    return correct_set


def cmp_teacher_student():
    correct_teacher_set = read_nll_score('./outputs/2023-02-28/18-42-29/scores.jsonl') 
    correct_student_set = read_nll_score('./outputs/2023-02-28/18-11-59/scores.jsonl') 
    
    teacher_1_student_0 = correct_teacher_set - correct_student_set  
    teacher_0_student_1 = correct_student_set - correct_teacher_set
    
    print('teacher_1_student_0 = %d' % len(teacher_1_student_0))
    print('teacher_0_student_1 = %d' % len(teacher_0_student_1))

def main():
    cmp_teacher_student()    
       

def show_rank_accuracy():
    top_k_lst = [20, 50, 100, 150, 200]
   
    student_rank_file = './outputs/2023-02-26/13-37-08/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)
    
    student_rank_file = './outputs/2023-02-26/17-16-33/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)

    student_rank_file = './outputs/2023-02-27/00-45-45/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)
    
    student_rank_file = './outputs/2023-02-27/12-01-55/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)
    
    student_rank_file = './outputs/2023-02-27/17-00-36/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)
    
    student_rank_file = './outputs/2023-02-28/00-00-47/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)
    
    student_rank_file = './outputs/2023-02-28/10-01-20/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)

if __name__ == '__main__':
    main()
