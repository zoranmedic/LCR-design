### BM25 prefiltering with Solr

## Download and installation

Download Solr from https://solr.apache.org/downloads.html

In our experiments we used version 8.7.0.

Install Solr using the instructions from the webpage.

## Starting core

### Setting Java home environment variable

Before starting, make sure that Java is set to version 8:

```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

```

### Starting Solr's core

Example for starting core with ACL papers:

```
bin/solr start

bin/solr create -c acl

curl -X POST -H 'Content-type:application/json' --data-binary '{"add-field": {"name":"pid", "type":"text_general", "multiValued":false, "stored":true}}' http://localhost:8983/solr/acl/schema

bin/post -c acl example/acl/papers.json

```

If variables $b$ and $k_1$ need to be changed (for validation) that has to be done manually in the configuration file (unfortunately, we didn't find any simpler way to do that).
Here are the steps for doing that:

1. Move to `conf` folder: `cd server/solr/acl/conf/`
2. Rename `managed-schema` file: `mv managed-schema schema.xml`
3. Edit `schema.xml` and change the values of parameters. Also set the similarity to BM25 as in: https://github.com/apache/lucene-solr/blob/663655d6590223573fbc816aa19e9c256fb8d937/solr/core/src/test-files/solr/collection1/conf/schema-bm25.xml#L33
4. Go back to root folder of solr: `cd ~/solr-8.7.0`
5. Start the core: `curl "http://localhost:8983/solr/admin/cores?action=RELOAD&core=acl"`

### Querying Solr's core

Python script `query_solr.py` is used for obtaining candidates for given citation contexts.

Example of running the script for obtaining candidates for ACL-ARC's validation set (from this folder):

```
python query_solr.py acl ../data/acl/papers.json ../data/acl/contexts.json ../data/acl/bm25/val_pairs_strict.json ../data/acl/exclude_acl_val_cids.json -output_file out.json

```
