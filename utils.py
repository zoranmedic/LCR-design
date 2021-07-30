import json
from math import log2

def year_from_id(paper_id):
    digits = int(paper_id[1:3])
    return 2000 + digits if digits < 60 else 1900 + digits

def mrr(ranks, k):
    rec_ranks = [1./r if r <= k else 0. for r in ranks]
    return sum(rec_ranks) / len(ranks)

def recall(ranks, k):
    return sum(r <= k for r in ranks) / len(ranks)

def ndcg(ranks, k):
    ndcg_per_query = sum(1 / log2(r + 1) for r in ranks if r <= k)
    return ndcg_per_query / len(ranks)


class Config(object):

    # Files and folders
    train_path = "train_true_pairs_json"
    train_hardnegs_path = "train_hardnegs_json"
    val_path = "val_triplets_json"
    val_path_2 = "val_triplets_json"
    contexts_path = "contexts_map_json"
    papers_path = "papers_map_json"
    embeddings_path = "ai2_embeddings.txt"
    models_folder = "path_to_folder_for_storing_models"
    predictions_folder = "path_to_folder_for_storing_predictions"

    # LSTM parameters
    rnn_num_layers = 1
    rnn_hidden_size = 100
    pad_token_id = 504339

    # Author convolution + Bib features parameters
    cnn_kernel_size = [1, 2]
    cnn_out_channels = [100, 100]
    author_embedding_dim = 50
    num_author_embeddings = None
    pad_author_id = None
    citations_dim = None
    
    # Training parameters
    num_epochs = None
    random_negs = None
    margin = None

    # For training: batch_size = train_queries_per_batch * triplets_per_context
    train_queries_per_batch = None
    triplets_per_context = None

    # For validation: batch_size = triplets_batch_size
    triplets_batch_size = None
    
    def __init__(self, config_file):
        config = json.load(open(config_file))
        self.__dict__.update(config)

    def dump(self, output_file):
        json.dump(self.__dict__, open(output_file, 'wt'))
        print(f'Stored CONFIG to: {output_file}')
