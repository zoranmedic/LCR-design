
import argparse
import json
import pysolr
import re
import sys

from utils import mrr, recall, ndcg


def main(args):

    papers = json.load(open(args.papers_path))
    contexts = json.load(open(args.contexts_path))
    cids = set([i['context_id'] for i in json.load(open(args.pairs_path))])
    exclude_cids = set(json.load(open(args.exclude_cids_path)))
    print(f'Total contexts: {len(cids)}')

    # Assumes Solr is started at port 8983
    solr = pysolr.Solr(f'http://localhost:8983/solr/{args.solr_core}/')

    ranks = []
    cid_candidates_map = {}
    for i, cid in enumerate(cids):
        if i % 1000 == 0:
            print(i)

        if cid in exclude_cids: 
            continue

        citing, cited = cid.split('_')[:2]
        citing_title, citing_abstract = papers[citing]['title'], papers[citing]['abstract']
        citing_title = citing_title.replace('\n', ' ')
        query = ' '.join([contexts[cid]['masked_text'].strip(), citing_title, citing_abstract])
        
        query = query.replace('\\', '\\\\')
        special_chars = ['+', '-', '&&', '||', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '/']
        for c in special_chars:
            query = query.replace(c, "\\" + c)
        query = query.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')

        query = query[:4000]
        if query[-1] == '\\' and query[-2] != '\\':
            query += '\\'
        try:
            results = solr.search(query, df='text', rows=1000, fl=['pid'])
        except:
            print(sys.exc_info())
            print(query)
            break

        cid_candidates = [result['pid'] for result in results][:1000]
        cid_candidates_map[cid] = cid_candidates
        if cited in cid_candidates:
            ranks.append(cid_candidates.index(cited) + 1)
        else:
            ranks.append(1001)

    if args.output_file != 'none':
        json.dump(cid_candidates_map, open(args.output_file, 'wt'))

    for k in [10, 1000]:
        print(f'k={k}')
        print(f'R@{k}: {recall(ranks, k=k)}')
        print(f'MRR: {mrr(ranks, k=k)}')
        print(f'NDCG: {ndcg(ranks, k=k)}')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('solr_core')
    parser.add_argument('papers_path')
    parser.add_argument('contexts_path')
    parser.add_argument('pairs_path')
    parser.add_argument('exclude_cids_path')
    parser.add_argument('-output_file', default='none')
    args = parser.parse_args()

    main(args)
