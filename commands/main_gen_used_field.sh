########### asqa ##################
# Dataset: asqa; Retriever: BAAI/bge-base-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/asqa_bge-base-en-v1.5_top100/retriever_output.json
# output_dir=new_data/asqa_bge-base-en-v1.5_summary_no_sub
# top_k=100
# use_sub_questions=0
# target_used_field=summary_no_sub
# openai_multi_thread=80
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/summary_no_sub_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: asqa; Retriever: gtr
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/asqa_gtr_top100/retriever_output.json
# output_dir=new_data/asqa_gtr_summary_use_sub
# top_k=100
# use_sub_questions=1
# target_used_field=summary_use_sub
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/summary_use_sub_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: asqa; Retriever: BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/asqa_bge-large-en-v1.5_top100/retriever_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_summary_use_sub
# top_k=100
# use_sub_questions=1
# target_used_field=summary_use_sub
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/${target_used_field}_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: asqa; Retriever: HYDE*BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0613
# data_file=new_data/asqa_bge-large-en-v1.5-hyde_top100/retriever_output_0613.json
# output_dir=new_data/asqa_bge-large-en-v1.5-hyde_summary_use_sub
# top_k=100
# use_sub_questions=1
# target_used_field=summary_use_sub
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/${target_used_field}_output_0613.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: asqa; Retriever: bm25
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/asqa_bm25_top100/retriever_output.json
# output_dir=new_data/asqa_bm25_summary_use_sub
# top_k=100
# use_sub_questions=1
# target_used_field=summary_use_sub
# openai_multi_thread=10
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/${target_used_field}_output_0301.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


############# qampari ############
# Dataset: qampari; Retriever: BAAI/bge-base-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=llama2
# data_file=new_data/qampari_bge-base-en-v1.5_top100/retriever_output.json
# output_dir=new_data/qampari_bge-base-en-v1.5_summary
# top_k=50
# use_sub_questions=0
# target_used_field=summary
# openai_multi_thread=5
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/summary_output-top50-llama2.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: qampari; Retriever: BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
model_name=gpt-3.5-turbo-1106
data_file=new_data/qampari_bge-large-en-v1.5_top100/retriever_output.json
output_dir=new_data/qampari_bge-large-en-v1.5_summary
top_k=50
use_sub_questions=0
target_used_field=summary
openai_multi_thread=10
temperature=0
max_tokens=150
prompt_style=summary

mkdir $output_dir -p
output_file=${output_dir}/${target_used_field}_output_1106.json

if [ ! -f "${output_file}" ]; then
    python gen_used_field.py \
        --model_name $model_name \
        --data_file $data_file \
        --output_file $output_file \
        --top_k $top_k \
        --use_sub_questions $use_sub_questions \
        --target_used_field $target_used_field \
        --openai_multi_thread $openai_multi_thread \
        --temperature $temperature \
        --max_tokens $max_tokens \
        --prompt_style $prompt_style
fi


# Dataset: qampari; Retriever: bm25-rest
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/qampari_rest_bm25-query-using-missing-info-from-question-and-psgs_top100/retriever-xnli_output.json
# output_dir=new_data/qampari_rest_bm25-query-using-missing-info-from-question-and-psgs_summary
# top_k=100
# use_sub_questions=0
# target_used_field=summary
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/xnli_summary_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: qampari; Retriever: HYDE*BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0613
# data_file=new_data/qampari_bge-large-en-v1.5-hyde_top100/retriever_output_0613.json
# output_dir=new_data/qampari_bge-large-en-v1.5-hyde_summary
# top_k=100
# use_sub_questions=0
# target_used_field=summary
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=summary

# mkdir $output_dir -p
# output_file=${output_dir}/${target_used_field}_output_0613.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


############# eli5 ############
# Dataset: eli5; Retriever: bm25-rest
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/eli5_rest_bm25-query-using-missing-info-from-question-and-psgs_top100/retriever-prompt13_output.json
# output_dir=new_data/eli5_rest_bm25-query-using-missing-info-from-question-and-psgs_answer
# top_k=100
# use_sub_questions=0
# target_used_field=answer
# openai_multi_thread=20
# temperature=0
# max_tokens=150
# prompt_style=answer

# mkdir $output_dir -p
# output_file=${output_dir}/prompt13_answer_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi


# Dataset: eli5; Retriever: HYDE*bm25
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# model_name=gpt-3.5-turbo-0301
# data_file=new_data/eli5_bm25-hyde_top100/retriever_output.json
# output_dir=new_data/eli5_bm25-hyde_answer
# top_k=100
# use_sub_questions=0
# target_used_field=answer
# openai_multi_thread=15
# temperature=0
# max_tokens=150
# prompt_style=answer

# mkdir $output_dir -p
# output_file=${output_dir}/${target_used_field}_output.json

# if [ ! -f "${output_file}" ]; then
#     python gen_used_field.py \
#         --model_name $model_name \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --target_used_field $target_used_field \
#         --openai_multi_thread $openai_multi_thread \
#         --temperature $temperature \
#         --max_tokens $max_tokens \
#         --prompt_style $prompt_style
# fi