# Local Citation Recommendation - Analysis of Design Choices

This repository includes code and instructions for experiments related to various design choices in constructing local citation recommendation systems. If you encounter any problems or errors feel free to raise an issue or send an email to <zoran.medic@fer.hr>.

## Data and Resources

Here we provide links for downloading preprocessed dataset instances as well as training, validation, and test splits, and files containing negative items for sampling in negative sampling strategies. Both ACL-ARC and RefSeer dataset files are compressed and available on the following links:
* [ACL-ARC](https://takelab.fer.hr/data/scicite/acl.tar.gz)
* [RefSeer](https://takelab.fer.hr/data/scicite/refseer.tar.gz)

Note that compressed `RefSeer` archive is ~23GB large (~70GB uncompressed).

For running the experiments, you will need the file with pretrained word embeddings. While the code should run with whatever embeddings you have (either pretrained or not), you should make sure to include the following tokens in the vocabulary: `'TARGETCIT', 'OTHERCIT', 'PAD', 'UNK'`. 
[Here](https://drive.google.com/file/d/1iiIu1Rz9iGPXs4La5_57CcJVQ8dxEc5j/view?usp=sharing) you can download the pretrained embeddings that we used in our experiments. The embeddings are from the paper [''Content-Based Citation Recommendation"](https://www.aclweb.org/anthology/N18-1022/) by Bhagavatula et al. (2018). In the file that we used, we included the above mentioned tokens in the vocabulary (randomly initialized) and used that in our experiments.

## Requirements

Model is built using PyTorch. For setting up the environment for running the code, make sure you run the following commands:

```
git clone git@github.com:zoranmedic/lcr-designchoices.git

conda env create -f environment.yml

conda activate scicite

python -m spacy download en_core_web_sm

```

## Data preprocessing

If you'd like to preprocess your own data instances (both contexts and articles) for using with the model, check out [these instructions](https://github.com/zoranmedic/lcr-designchoices/sample_data/README.md).

## Training

To train the 'Text+Bib' variant of the model with graph neigbors negative sampling strategy, run this command:

```train
python train.py <path_to_config_json> <log_dir> -strategy graph --dual --global_info
```

Make sure you fill out all the relevant fields in the config JSON file.


## Evaluation

To generate predictions on test files with 'Text+Bib' variant of the model, run the following command:

```eval
python predict.py <path_to_config_json> <path_to_saved_model> --dual --global_info <path_to_input_file>
```
