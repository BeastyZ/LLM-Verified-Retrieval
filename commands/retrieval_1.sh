# Dataset: eli5; Retriever: bm25
# export CUDA_VISIBLE_DEVICES=0
# retriever=bm25 
# data_file=new_data/eli5_bm25-hyde_top100/retriever_output.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/faiss_index/sparse  # bm25-sphere-index
# output_dir=new_data/eli5_test
# top_k=100
# doc_pool=sphere

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python multi_process/bm25_multi_process.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool
# fi

