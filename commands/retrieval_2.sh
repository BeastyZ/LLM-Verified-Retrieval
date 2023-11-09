# Dataset: qampari; Retriever: bm25
export CUDA_VISIBLE_DEVICES=7
retriever=bm25 
method=lamer
data_file=new_data/asqa_instructor-large-lamer_top100/retriever_output.json
corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
output_dir=new_data/asqa_bm25-lamer_top100
top_k=100 
doc_pool=wiki

mkdir $output_dir -p
output_file=${output_dir}/retriever_output.json

if [ ! -f "${output_file}" ]; then
    python multi_process/bm25_multi_process.py \
        --retriever $retriever \
        --data_file $data_file \
        --corpus_path $corpus_path \
        --output_file $output_file \
        --top_k $top_k \
        --doc_pool $doc_pool \
        --method $method
fi