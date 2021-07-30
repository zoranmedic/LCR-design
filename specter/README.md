## SPECTER prefiltering

### SPECTER code

For obtaining query and document embeddings, code from SPECTER's public repository is used: https://github.com/allenai/specter

### Generating input for SPECTER

Script `generate_input.py` creates files that should be passed as input to SPECTER's script for obtaining embeddings.
Example of generating input for ACL-ARC's papers and train contexts that stores generated files into `data/acl-specter` folder:
```
python generate_input.py ../data/acl/contexts.json ../data/acl/papers.json ../data/acl/train_pairs_standard.json ../data/acl-specter

```

### Obtaining embeddings for all contexts

Example of running SPECTER's script that outputs embeddings for "contextonly" variant of input representation in the `contextonly-output.jsonl` file.
```
python scripts/embed.py --ids data/acl/contextonly.ids --metadata data/acl-specter/contextonly-metadata.json --model ./model.tar.gz --output-file data/acl/contextonly-output.jsonl --vocab-dir data/vocab/ --batch-size 16 --cuda-device 0

```

### Generating candidates with SPECTER prefiltering

Once the embeddings of papers and citation contexts are generated and stored in `.jsonl` files, script for obtaining candidates can be run to generate a list of candidates for each given context.
Example of running the script for ACL-ARC's train contexts in "contextonly" variant and storing the candidates into "contextonly-candidates.json" file.
```
python generate_candidates.py ../data/acl/train_pairs_standard.json ../data/acl-specter/papers-output.jsonl ../data/acl-specter/contextonly-output.jsonl ../data/acl-specter/contextonly-candidates.json -dataset acl

```

### Evaluating candidates obtained with SPECTER

Candidates for each context stored into `.json` file can be evaluated with `evaluate_candidates.py` script.
Example of evaluating `contextonly-candidates.json` file:
```
python evaluate_candidates.py ../data/acl-specter/contextonly-candidates.json

```