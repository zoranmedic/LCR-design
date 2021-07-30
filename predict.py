import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import PairsDataset
from datasets.factory import create_dataloader
from datasets.collate import collate_pairs_predict
from model import CitRecModel
from utils import Config


def main(args):
    config = Config(args.config_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'

    model = CitRecModel(
        embeddings_path=config.embeddings_path,
        rnn_num_layers=config.rnn_num_layers,
        rnn_hidden_size=config.rnn_hidden_size,
        author_embedding_dim=config.author_embedding_dim,
        num_author_embeddings=config.author_embedding_num,
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_out_channels=config.cnn_out_channels,
        citations_dim=config.citations_dim,
        pad_author_id=config.pad_author_id,
        dual=args.dual,
        global_info=args.global_info,
        device=device
    )
    model.load_state_dict(
        torch.load(f'{args.model_path}', map_location=torch.device(device))
    )
    model = model.to(device)
    print(f'Loaded model: {args.model_path}')

    for input_file in args.input_file:
        dataloader = create_dataloader(
            file_path=input_file,
            items_type='pairs',
            mode='test',
            pad_token_id=config.pad_token_id,
            pad_author_id=config.pad_author_id,
            batch_size=config.triplets_batch_size
        )

        context_scores = defaultdict(list)
        with torch.no_grad():
            with tqdm(total=len(dataloader)) as pbar:
                for i, (data, ids) in enumerate(dataloader):
                    preds = model.predict(data)
                    score_preds = preds[2] if args.dual else preds
                    score_preds = score_preds.squeeze(1).cpu().tolist()
                    for j, (cid, pid) in enumerate(zip(ids['context_ids'], ids['ref_ids'])):
                        context_scores[cid].append((pid, score_preds[j]))
                    pbar.update(1)
        print(f'Prediction on file {input_file} done.')
        
        output_filename = f'{args.model_path.split("/")[-1].split(".")[0]}_{input_file.split("/")[-1]}'
        output_path = f'{config.predictions_folder}/{output_filename}'
        json.dump(context_scores, open(output_path, 'wt'))
        print(f'Predictions stored to: {output_path}')

        del dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('model_path')
    parser.add_argument('--dual', default=False, action='store_true', 
                        help='Whether model should use both modules (Text and Bib). if not set, model uses only Text module')
    parser.add_argument('--global_info', default=False, action='store_true', 
                        help='Whether model should use global information from citing article')  # TODO change to default True
    parser.add_argument('input_file', nargs='+')
    args = parser.parse_args()

    main(args)
