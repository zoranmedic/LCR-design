import argparse
import jsonlines
import faiss
import numpy as np
import random
import json

from math import log2

from utils import year_from_id

def get_candidates(paper_embeddings, query_embeddings, dataset, year_cids=None):

    # Load paper embeddings
    paper_ids = []
    paper_embs = []
    with jsonlines.open(paper_embeddings) as reader:
        for line in reader:
            if dataset == 'acl' and year_from_id(line['paper_id']) <= year or dataset == 'refseer':
                paper_ids.append(line['paper_id'])
                paper_embs.append(np.array(line['embedding']).astype('float32'))
    print(f'Loaded {len(paper_ids)} paper embeddings.')

    # Load context embeddings
    context_ids = []
    context_embs = []
    with jsonlines.open(query_embeddings) as reader:
        for line in reader:
            if dataset == 'acl' and line['paper_id'] in year_cids or dataset == 'refseer':
                context_ids.append(line['paper_id'])
                context_embs.append(np.array(line['embedding']).astype('float32'))
    print(f'Loaded {len(context_ids)} context embeddings.')

    # Normalize embeddings before creating index
    paper_embs = np.array(paper_embs)
    context_embs = np.array(context_embs)

    # Setup index on GPU
    res = faiss.StandardGpuResources()  # use a single GPU
    d = paper_embs.shape[1]
    index_flat = faiss.IndexFlatL2(d) # IndexFlatIP(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    # Index paper embeddings                          
    gpu_index_flat.add(paper_embs)         # add vectors to the index
    
    k = 1024                                       # get 1024 nearest neighbors (max for GPU)
    D, I = gpu_index_flat.search(context_embs, k)  # actual search
    
    for cid, scores, neighbours in zip(context_ids, D, I):
        citing, cited = cid.split('_')[:2]
        neighbours_pids = [paper_ids[i] for i in neighbours]
        candidates = [i for i in neighbours_pids if i != citing][:1000] # store top 1000 candidates
        context_candidates[cid] = candidates

    return context_candidates

def main(args):

    cids = set([i['context_id'] for i in json.load(open(args.train_pairs))])
    context_candidates = {}

    if args.dataset == 'acl':
        years = set([year_from_id(i.split('_')[0]) for i in cids])
        for year in years:
            year_cids = set(cid for cid in cids if year_from_id(cid.split('_')[0]) == year)
            year_candidates = get_candidates(args.paper_embeddings, args.query_embeddings, args.dataset, year_cids)
            for k in year_candidates:
                context_candidates[k] = year_candidates[k]
    else:
        context_candidates = get_candidates(args.paper_embeddings, args.query_embeddings, args.dataset)

    json.dump(context_candidates, open(args.output_file, 'wt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_pairs')
    parser.add_argument('paper_embeddings')
    parser.add_argument('query_embeddings')
    parser.add_argument('output_file')
    parser.add_argument('-dataset', default='acl')
    args = parser.parse_args()
    main(args)