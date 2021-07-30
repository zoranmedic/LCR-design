import argparse
import json
import os

import spacy
nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]


def main(args):
    contexts = json.load(open(args.contexts))
    papers = json.load(open(args.papers))
    queries = json.load(open(args.pairs))

    # Generate files for paper embeddings
    metadata = {}
    for pid in papers:
        metadata[pid] = {
            'title': papers[pid]['title'],
            'abstract': papers[pid]['abstract'],
            'paper_id': pid
        }
    json.dump(metadata, open(os.path.join(args.data_dir, 'papers-metadata.json'), 'wt'))

    with open(os.path.join(args.data_dir, 'papers.ids', 'wt')) as f:
        for pid in metadata.keys():
            f.write('%s\n' % pid)

    # Generate files for citation context embeddings
    metadata_per_variant = {
        'contextpaper': {},
        'papercontext': {},
        'contextonly': {},
        'paperonly': {}
    }

    cids = set([i['context_id'] for i in queries])
    for cid in cids:
        citing = cid.split('_')[0]
        paper = ' '.join(tokenize(papers[citing]['abstract'])[:200])
        
        metadata_per_variant['paperonly'][cid] = {
            'title': papers[citing]['title'],
            'abstract': papers[citing]['abstract'],
            'paper_id': cid
        }
        metadata_per_variant['contextpaper'][cid] = {
            'title': papers[citing]['title'],
            'abstract': contexts[cid]['masked_text'] + ' ' + paper,
            'paper_id': cid
        }
        metadata_per_variant['papercontext'][cid] = {
            'title': papers[citing]['title'],
            'abstract': paper + ' ' + contexts[cid]['masked_text'],
            'paper_id': cid
        }
        metadata_per_variant['contextonly'][cid] = {
            'title': papers[citing]['title'],
            'abstract': contexts[cid]['masked_text'],
            'paper_id': cid
        }

    for k in metadata_per_variant:
        json.dump(metadata_per_variant[k], open(os.path.join(args.data_dir, f'{k}-metadata.json', 'wt'))
        with open(os.path.join(args.data_dir, f'{k}.ids', 'wt') as f:
            for cid in metadata_per_variant[k].keys():
                f.write('%s\n' % cid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('contexts')
    parser.add_argument('papers')
    parser.add_argument('pairs')
    parser.add_argument('output_data_dir')
    args = parser.parse_args()
    main(args)