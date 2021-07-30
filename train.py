import argparse
import json
import logging
import math
import random
from datetime import datetime
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset import PairsDataset, TripletsDataset
from datasets.collate import collate_pairs_train, collate_triplets
from datasets.factory import create_dataloader
from model import CitRecModel
from utils import Config, mrr


def custom_triplet_loss(scores, margin=None, device='cpu'):
    margins = (torch.ones(scores[0].shape) * margin).to(device)
    zeros = torch.zeros(scores[0].shape).to(device)
    text_loss = torch.max(scores[1] - scores[0] + margins, zeros)
    if len(scores) == 2:    # Text reranker variant
        return torch.mean(text_loss)
    else:                   # Text+Bib reranker variant
        bib_loss = torch.max(scores[3] - scores[2] + margins, zeros)
        total_loss = torch.max(scores[5] - scores[4] + margins, zeros)
        loss = text_loss + bib_loss + total_loss
        return torch.mean(loss)


def main(args):
    train_start = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    writer = SummaryWriter(log_dir=args.log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'

    config = Config(args.config_file)

    val_model_path = f'{config.models_folder}/{args.log_dir.strip().split("/")[-1]}-{train_start}.pth'
    hard_negatives_path = config.train_hardnegs_path if args.strategy != 'random' else None
    # negative_type = 'mix' if args.hardnegs else 'random'
    
    train_dataloader = create_dataloader(
        file_path=config.train_path,
        items_type='pairs',
        mode='train',
        pad_token_id=config.pad_token_id,
        pad_author_id=config.pad_author_id,
        batch_size=config.train_queries_per_batch,
        triplets_per_context=config.triplets_per_context,
        negative_type=args.hardnegs,
        hard_negatives_path=hard_negatives_path,
        batches_per_epoch=config.batches_per_epoch
    )
    val_dataloader = create_dataloader(
        file_path=config.val_path, 
        items_type='pairs',
        mode='val',
        pad_token_id=config.pad_token_id,
        pad_author_id=config.pad_author_id,
        batch_size=config.triplets_batch_size
    )

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
    if device == 'cuda':
        model = model.to(device)
    optimizer = Adam(model.parameters())

    num_items = len(train_dataloader.dataset.pairs)
    print(f'Number of training pairs: {num_items}')
    
    best_mrr = 0.
    iter_cnt = 0

    for epoch in range(config.num_epochs):

        # Train the model
        with tqdm(total=len(train_dataloader)) as pbar:
            for i, (batch, _) in enumerate(train_dataloader):
                optimizer.zero_grad()
                preds = model(batch)
                batch_loss = custom_triplet_loss(
                    scores=preds, 
                    margin=config.margin,
                    device=device
                )
                batch_loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({'loss': batch_loss.item()})

                iter_cnt += 1
                writer.add_scalars(
                    'Loss/loss_per_batch', 
                    {'batch_loss': batch_loss.item()}, 
                    iter_cnt
                )

        # Run validation after 1st and later after each 5 epochs, skip otherwise
        if (epoch + 1) % 5 != 0 and epoch != 0:     
            continue

        # Run the validation set on the model
        val_context_scores = defaultdict(list)
        with torch.no_grad():
            with tqdm(total=len(val_dataloader)) as val_pbar:
                for data, ids in val_dataloader:
                    preds = model.predict(data)
                    score_preds = preds[2] if args.dual else preds
                    score_preds = score_preds.squeeze(1).cpu().tolist()
                    for j, (cid, pid) in enumerate(zip(ids['context_ids'], ids['ref_ids'])):
                        val_context_scores[cid].append((pid, score_preds[j]))
                    val_pbar.update(1)
        
        # For each validation context get the rank of the correct recommendation
        ranks = []
        for cid in val_context_scores:
            true_pid = cid.split('_')[1]
            sorted_pids = [ 
                i[0] for i in sorted(val_context_scores[cid], key=lambda x: x[1], reverse=True)
            ]
            rank = sorted_pids.index(true_pid) + 1     # rank of correct recommendation
            ranks.append(rank)

        # Calculate MRR from the ranks
        val_mrr = mrr(ranks, k=10)
        writer.add_scalars(
            f'MRR/MRR_per_epoch', {f'val': val_mrr}, epoch + 1
        )
        logging.info(f'Validation MRR in epoch {epoch + 1}: {val_mrr}')

        # If validation MRR is the best so far, save the model
        if val_mrr > best_mrr:
            logging.info(f'Reached best validation MRR in epoch {epoch + 1}. ' \
                        f'Saving model to: {val_model_path}')
            best_mrr = val_mrr
            torch.save(model.state_dict(), val_model_path)
    
    logging.info('DONE.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('log_dir')
    parser.add_argument('-strategy', default='random', type=str, help='Negative sampling strategy. Expected values: [random, prefiltered, graph, most_citations, cited].')
    parser.add_argument('--dual', default=False, action='store_true', 
                        help='Whether model should use both modules (Text and Bib). if not set, model uses only Text module')
    parser.add_argument('--global_info', default=False, action='store_true', 
                        help='Whether model should use global information from citing article')  # TODO change to default True
    parser.add_argument('--hard_negatives_only', default=False, action='store_true', hepl='Whether only hard negatives will be sampled for triplets.')
    args = parser.parse_args()

    main(args)
