python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input tmp_wiki/wiki_jsonls \
  --index tmp_wiki/psgs_w100.index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw