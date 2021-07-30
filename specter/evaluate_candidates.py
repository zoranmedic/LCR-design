import argparse

from utils import mrr, ndcg, recall

def main(args):
    context_candidates = json.load(open(args.candidates_file))

    ranks = []
    for cid in context_candidates:
        citing, cited = cid.split('_')[:2]
        rank = 1001
        if cited in context_candidates[cid]:
            rank = context_candidates[cid].index(cited) + 1
        ranks.append(rank)

    print(f'Recall@k: {recall(ranks, k=10):.5f}')
    print(f'MRR: {mrr(ranks, k=10):.5f}')
    print(f'NDCG: {ndcg(ranks, k=10):.5f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidates_file')
    args = parser.parse_args()

    main(args)
