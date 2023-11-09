############### asqa ################
# Dataset: asqa; Rerank: llm_select
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# reranker=llm_select
# data_file=new_data/asqa_eval_gtr_top100_with_summaries/input.json
# output_dir=new_data/asqa_gtr_llm-select_top5
# openai_multi_thread=6
# used_doc_field=summary_use_sub
# use_title=1
# top_k=5
# use_sub_questions=1
# openai_model_name=gpt-3.5-turbo-0301
# window_size=20
# system_prompt_file_name=prompt5_select_no_up_to
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --window_size $window_size \
#         --system_prompt_file_name $system_prompt_file_name \
#         --doc_num $doc_num
# fi


# Dataset: asqa; Rerank: llama2
# export CUDA_VISIBLE_DEVICES=7
# reranker=llama2
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_llama2
# top_k=5 
# openai_multi_thread=5
# used_doc_field=summary_use_sub 
# use_title=1
# use_sub_questions=0
# openai_model_name=llama2
# system_prompt_file_name=prompt5_select_no_up_to
# window_size=20
# doc_num=50
# # position=tail
# # old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-main-question.json

# if [ ! -f "${output_file}" ]; then
#     echo "*****************************"
#     echo "lxn start rerank"
#     echo "*****************************"

#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --used_doc_field $used_doc_field \
#         --output_file $output_file \
#         --top_k $top_k \
#         --openai_multi_thread $openai_multi_thread \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --window_size $window_size \
#         --doc_num $doc_num
#         # --position $position \
#         # --old_data_file $old_data_file

#     echo "*****************************"
#     echo "lxn finish rerank"
#     echo "*****************************"
# fi


# Dataset: asqa; Rerank: rank_gpt
# export CUDA_VISIBLE_DEVICES=5,6,7 
# reranker=rank_gpt 
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_rankgpt
# used_doc_field=text
# use_title=1
# use_sub_questions=1
# openai_model_name=gpt-3.5-turbo-0613

# mkdir $output_dir -p
# output_file=${output_dir}/reranker-text_output_0613.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name
# fi


# Dataset: asqa; Rerank: BAAI/bge-reranker-large
# export CUDA_VISIBLE_DEVICES=1
# reranker=BAAI/bge-reranker-large
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_bge-reranker-large
# openai_multi_thread=6
# used_doc_field=summary_use_sub
# use_title=1
# top_k=5
# use_sub_questions=1
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi


# Dataset: asqa; Rerank: BAAI/bge-reranker-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=BAAI/bge-reranker-base
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_bge-reranker-base
# openai_multi_thread=6
# used_doc_field=summary_use_sub
# use_title=1
# top_k=5
# use_sub_questions=1
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi

# Dataset: asqa; Rerank: monot5-3B
# export CUDA_VISIBLE_DEVICES=1
# reranker=castorini/monot5-3b-msmarco-10k
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_monot5-3b-reranker
# top_k=5 
# use_title=1
# use_sub_questions=1
# used_doc_field=summary_use_sub
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi 


# Dataset: asqa; Rerank: monot5-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=castorini/monot5-base-msmarco-10k
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_monot5-base-reranker
# top_k=5
# use_title=1
# use_sub_questions=1
# used_doc_field=summary_use_sub
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi


# Dataset: asqa; Rerank: castorini/monobert-large
# export CUDA_VISIBLE_DEVICES=3
# reranker=castorini/monobert-large-msmarco
# data_file=new_data/asqa_bge-large-en-v1.5_summary_use_sub/summary_use_sub_output.json
# output_dir=new_data/asqa_bge-large-en-v1.5_monobert-large-reranker
# top_k=5
# use_title=1
# use_sub_questions=1
# used_doc_field=summary_use_sub
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi


################# qampari #################
# Dataset: qampari; Rerank: llm_select
# export CUDA_VISIBLE_DEVICES=0,3
reranker=llm_select
data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output_1106.json
output_dir=new_data/qampari_bge-large-en-v1.5_llm-select
top_k=5 
openai_multi_thread=10
used_doc_field=summary
use_title=0
use_sub_questions=0
openai_model_name=gpt-3.5-turbo-1106
system_prompt_file_name=prompt5_select_no_up_to
window_size=20
doc_num=50
# position=tail
# old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

mkdir $output_dir -p
output_file=${output_dir}/reranker_output_1106.json

if [ ! -f "${output_file}" ]; then
    echo "*****************************"
    echo "lxn start rerank"
    echo "*****************************"

    python rerank.py \
        --reranker $reranker \
        --data_file $data_file \
        --used_doc_field $used_doc_field \
        --output_file $output_file \
        --top_k $top_k \
        --openai_multi_thread $openai_multi_thread \
        --use_title $use_title \
        --use_sub_questions $use_sub_questions \
        --openai_model_name $openai_model_name \
        --system_prompt_file_name $system_prompt_file_name \
        --window_size $window_size \
        --doc_num $doc_num
        # --position $position \
        # --old_data_file $old_data_file

    echo "*****************************"
    echo "lxn finish rerank"
    echo "*****************************"
fi

# # Dataset: qampari; Rerank: llama2
# # export CUDA_VISIBLE_DEVICES=7
# reranker=llama2
# data_file=new_data/qampari_bge-large-en-v1.5_top100/retriever_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_llama2
# top_k=5 
# openai_multi_thread=5
# used_doc_field=text 
# use_title=0
# use_sub_questions=0
# openai_model_name=llama2
# system_prompt_file_name=prompt5_select_no_up_to
# window_size=20
# doc_num=50
# # position=tail
# # old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-top50-text.json

# if [ ! -f "${output_file}" ]; then
#     echo "*****************************"
#     echo "lxn start rerank"
#     echo "*****************************"

#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --used_doc_field $used_doc_field \
#         --output_file $output_file \
#         --top_k $top_k \
#         --openai_multi_thread $openai_multi_thread \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --window_size $window_size \
#         --doc_num $doc_num
#         # --position $position \
#         # --old_data_file $old_data_file

#     echo "*****************************"
#     echo "lxn finish rerank"
#     echo "*****************************"
# fi


# Dataset: qampari; Rerank: xwin
# export CUDA_VISIBLE_DEVICES=7
# reranker=xwin
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_xwin
# top_k=5 
# openai_multi_thread=5
# used_doc_field=summary
# use_title=0
# use_sub_questions=0
# openai_model_name=xwin
# system_prompt_file_name=prompt5_select_no_up_to
# window_size=20
# doc_num=50
# # position=tail
# # old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output.json

# if [ ! -f "${output_file}" ]; then
#     echo "*****************************"
#     echo "lxn start rerank"
#     echo "*****************************"

#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --used_doc_field $used_doc_field \
#         --output_file $output_file \
#         --top_k $top_k \
#         --openai_multi_thread $openai_multi_thread \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --window_size $window_size \
#         --doc_num $doc_num
#         # --position $position \
#         # --old_data_file $old_data_file

#     echo "*****************************"
#     echo "lxn finish rerank"
#     echo "*****************************"
# fi


# Dataset: qampari; Rerank: rank_gpt
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# reranker=rank_gpt 
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_rankgpt
# used_doc_field=text
# use_title=0
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0613

# mkdir $output_dir -p
# output_file=${output_dir}/reranker-text_output_0613.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name
# fi


# Dataset: qampari; Rerank: BAAI/bge-reranker-large
# export CUDA_VISIBLE_DEVICES=1
# reranker=BAAI/bge-reranker-large
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_bge-reranker-large
# openai_multi_thread=6
# used_doc_field=summary
# use_title=0
# top_k=5
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi


# Dataset: qampari; Rerank: BAAI/bge-reranker-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=BAAI/bge-reranker-base
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_bge-reranker-base
# openai_multi_thread=6
# used_doc_field=summary
# use_title=0
# top_k=5
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi


# Dataset: qampari; Rerank: monot5-3B
# export CUDA_VISIBLE_DEVICES=1
# reranker=castorini/monot5-3b-msmarco-10k
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_monot5-3b-reranker
# top_k=5 
# use_title=0
# use_sub_questions=0
# used_doc_field=summary
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi 


# Dataset: qampari; Rerank: monot5-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=castorini/monot5-base-msmarco-10k
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_monot5-base-reranker
# top_k=5
# use_title=0
# use_sub_questions=0
# used_doc_field=summary
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi


# Dataset: qampari; Rerank: castorini/monobert-large
# export CUDA_VISIBLE_DEVICES=3
# reranker=castorini/monobert-large-msmarco
# data_file=new_data/qampari_bge-large-en-v1.5_summary/summary_output.json
# output_dir=new_data/qampari_bge-large-en-v1.5_monobert-large-reranker
# top_k=5
# use_title=0
# use_sub_questions=0
# used_doc_field=summary
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi


################ eli5 #################
# Dataset: eli5; Rerank: llm_select
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# reranker=llm_select
# data_file=new_data/eli5_rest_bm25-query-using-question_answer/answer_output.json
# output_dir=new_data/eli5_rest_bm25-query-using-question_llm-select-tail
# top_k=5 
# openai_multi_thread=20
# used_doc_field=answer 
# use_title=0
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301
# system_prompt_file_name=prompt5_select_no_up_to
# window_size=20
# position=tail
# old_data_file=new_data/eli5_bm25_llm-select_filter_demo/no_filter_output.json

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output.json

# if [ ! -f "${output_file}" ]; then
#     echo "*****************************"
#     echo "lxn start rerank"
#     echo "*****************************"

#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --used_doc_field $used_doc_field \
#         --output_file $output_file \
#         --top_k $top_k \
#         --openai_multi_thread $openai_multi_thread \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --window_size $window_size \
#         --position $position \
#         --old_data_file $old_data_file

#     echo "*****************************"
#     echo "lxn finish rerank"
#     echo "*****************************"
# fi


# Dataset: eli5; Rerank: llama2
# export CUDA_VISIBLE_DEVICES=7
# reranker=llama2
# data_file=new_data/eli5_bm25-hyde_answer/answer_output.json
# output_dir=new_data/eli5_bm25-hyde_llama2
# top_k=5 
# openai_multi_thread=5
# used_doc_field=answer
# use_title=0
# use_sub_questions=0
# openai_model_name=llama2
# system_prompt_file_name=prompt5_select_no_up_to
# window_size=20
# doc_num=50
# # position=tail
# # old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output.json

# if [ ! -f "${output_file}" ]; then
#     echo "*****************************"
#     echo "lxn start rerank"
#     echo "*****************************"

#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --used_doc_field $used_doc_field \
#         --output_file $output_file \
#         --top_k $top_k \
#         --openai_multi_thread $openai_multi_thread \
#         --use_title $use_title \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --window_size $window_size \
#         --doc_num $doc_num
#         # --position $position \
#         # --old_data_file $old_data_file

#     echo "*****************************"
#     echo "lxn finish rerank"
#     echo "*****************************"
# fi


# Dataset: eli5; Rerank: rank_gpt
# export CUDA_VISIBLE_DEVICES=5,6,7 
# reranker=rank_gpt 
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_rankgpt
# used_doc_field=answer
# use_title=0
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output_0301_veri.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name
# fi


# Dataset: eli5; Rerank: BAAI/bge-reranker-large
# export CUDA_VISIBLE_DEVICES=1
# reranker=BAAI/bge-reranker-large
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_bge-reranker-large
# openai_multi_thread=6
# used_doc_field=answer
# use_title=0
# top_k=5
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi


# Dataset: eli5; Rerank: BAAI/bge-reranker-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=BAAI/bge-reranker-base
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_bge-reranker-base
# openai_multi_thread=6
# used_doc_field=answer
# use_title=0
# top_k=5
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301
# batch_size=512
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --openai_multi_thread $openai_multi_thread \
#         --used_doc_field $used_doc_field \
#         --use_title $use_title \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name \
#         --batch_size $batch_size \
#         --doc_num $doc_num
# fi


# Dataset: eli5; Rerank: monot5-3B
# export CUDA_VISIBLE_DEVICES=1
# reranker=castorini/monot5-3b-msmarco-10k
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_monot5-3b-reranker
# top_k=5 
# use_title=0
# used_doc_field=answer
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi 


# Dataset: eli5; Rerank: monot5-base
# export CUDA_VISIBLE_DEVICES=2
# reranker=castorini/monot5-base-msmarco-10k
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_monot5-base-reranker
# top_k=5
# use_title=0
# used_doc_field=answer
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi

# Dataset: eli5; Rerank: castorini/monobert-large
# export CUDA_VISIBLE_DEVICES=3
# reranker=castorini/monobert-large-msmarco
# data_file=new_data/eli5_eval_bm25_top100_with_summaries/input.json
# output_dir=new_data/eli5_bm25_monobert-large-reranker
# top_k=5
# use_title=0
# used_doc_field=answer
# doc_num=50

# mkdir $output_dir -p
# output_file=${output_dir}/reranker_output-50.json

# if [ ! -f "${output_file}" ]; then
#     python rerank.py \
#         --reranker $reranker \
#         --data_file $data_file \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_title $use_title \
#         --used_doc_field $used_doc_field \
#         --doc_num $doc_num
# fi
