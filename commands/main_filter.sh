# asqa 数据集
# export CUDA_VISIBLE_DEVICES=5,6,7 
# openai_multi_thread=9
# openai_model_name=gpt-3.5-turbo-0301
# idf_use_letter=int 
# use_title=1 
# system_prompt_file_name=prompt14_score
# used_doc_field=summary_use_sub 
# k=5 
# use_demo=0
# demo_file=prompts/asqa_demo.json
# data_file=iter_retrieval_50_threshold/asqa_max-4_bge-large-en-v1.5-prompt12-update-using-missing-info-from-question-and-psgs_summary_use_sub_llm-select-head/reranker_output_iter-0.json
# use_sub_questions=1
# threshold=7

# output_dir=iter_retrieval_50_threshold/asqa_max-4_bge-large-en-v1.5-prompt12-update-using-missing-info-from-question-and-psgs_summary_use_sub_llm-select-head_filtration
# mkdir $output_dir -p

# yes_output_file=${output_dir}/filtration_output_iter-0_yes.json
# no_output_file=${output_dir}/filtration_output_iter-0_no.json

# if [ ! -f "${yes_output_file}" ]; then
#     python filter.py \
#         --data_file $data_file \
#         --openai_multi_thread $openai_multi_thread \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --used_doc_field $used_doc_field \
#         --output_dir $output_dir \
#         --k $k \
#         --idf_use_letter $idf_use_letter \
#         --use_title $use_title \
#         --use_demo $use_demo \
#         --demo_file $demo_file \
#         --use_sub_questions $use_sub_questions \
#         --yes_output_file $yes_output_file \
#         --no_output_file $no_output_file \
#         --threshold $threshold
# fi 


# eli5 数据集
# openai_multi_thread=6
# openai_model_name=gpt-3.5-turbo-0613 
# idf_use_letter=int 
# use_title=0
# system_prompt_file_name=prompt14_score
# used_doc_field=answer
# k=5 
# use_demo=0
# demo_file=prompts/eli5_default.json
# data_file=new_data/eli5_bm25-hyde_llm-select/reranker_output_0613_title-0-prompt5.json
# use_sub_questions=0
# threshold=5.5

# output_dir=new_data/eli5_bm25-hyde_llm-select_filter
# mkdir $output_dir -p

# yes_output_file=${output_dir}/yes_filter_output-prompt14.json
# no_output_file=${output_dir}/no_filter_output-prompt14.json

# if [ ! -f "${yes_output_file}" ]; then
#     python filter.py \
#         --data_file $data_file \
#         --openai_multi_thread $openai_multi_thread \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name \
#         --used_doc_field $used_doc_field \
#         --output_dir $output_dir \
#         --k $k \
#         --idf_use_letter $idf_use_letter \
#         --use_title $use_title \
#         --use_demo $use_demo \
#         --demo_file $demo_file \
#         --use_sub_questions $use_sub_questions \
#         --yes_output_file $yes_output_file \
#         --no_output_file $no_output_file \
#         --threshold $threshold
# fi 


# qampari 数据集
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
openai_multi_thread=10
openai_model_name=gpt-3.5-turbo-1106
idf_use_letter=int 
use_title=0
system_prompt_file_name=prompt8_filter_question_with_demo
used_doc_field=summary
k=5 
use_demo=1
demo_file=prompts/qampari_default.json
data_file=new_data/qampari_bge-large-en-v1.5_llm-select/reranker_output_1106.json
use_sub_questions=0
# threshold=7

output_dir=new_data/qampari_bge-large-en-v1.5_llm-select_filter
mkdir $output_dir -p

yes_output_file=${output_dir}/filtration_output_iter-0_yes_1106.json
no_output_file=${output_dir}/filtration_output_iter-0_no_1106.json

if [ ! -f "${yes_output_file}" ]; then
    python filter.py \
        --data_file $data_file \
        --openai_multi_thread $openai_multi_thread \
        --openai_model_name $openai_model_name \
        --system_prompt_file_name $system_prompt_file_name \
        --used_doc_field $used_doc_field \
        --output_dir $output_dir \
        --k $k \
        --idf_use_letter $idf_use_letter \
        --use_title $use_title \
        --use_demo $use_demo \
        --demo_file $demo_file \
        --use_sub_questions $use_sub_questions \
        --yes_output_file $yes_output_file \
        --no_output_file $no_output_file
fi 
