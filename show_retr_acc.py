import json
import numpy as np

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

def main():
    top_k_lst = [20, 50, 100, 150, 200]
   
    teacher_rank_file = './outputs/2023-02-11/23-09-57/rank.jsonl'
    show_acc('Teacher', teacher_rank_file, top_k_lst)
     
    ta_rank_file = './outputs/2023-02-11/23-10-59/rank.jsonl'
    show_acc('TA', ta_rank_file, top_k_lst)
     
    student_rank_file = './outputs/2023-02-14/12-44-31/rank.jsonl'
    show_acc('Student', student_rank_file, top_k_lst)

if __name__ == '__main__':
    main()
