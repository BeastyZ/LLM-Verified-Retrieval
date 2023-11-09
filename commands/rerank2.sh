# Dataset: asqa; Rerank: rank_gpt
# export CUDA_VISIBLE_DEVICES=5,6,7 
reranker=rank_gpt 
data_file=new_data/qampari_bge-large-en-v1.5_top100/retriever_output.json
output_dir=new_data/qampari_bge-large-en-v1.5_rankgpt
used_doc_field=text
use_title=0
use_sub_questions=0
doc_num=50
openai_model_name=gpt-3.5-turbo-0301

mkdir $output_dir -p
output_file=${output_dir}/reranker_output-text-50_0301.json

if [ ! -f "${output_file}" ]; then
    python rerank.py \
        --reranker $reranker \
        --data_file $data_file \
        --output_file $output_file \
        --use_title $use_title \
        --used_doc_field $used_doc_field \
        --use_sub_questions $use_sub_questions \
        --openai_model_name $openai_model_name \
        --doc_num $doc_num
fi