##################### asqa #####################
# Iterative_retrieval
# export CUDA_VISIBLE_DEVICES=7
# max_iteration=4
# dataset_name=asqa
# use_sub_questions=1
# use_title=1
# used_doc_field=summary_use_sub
# openai_model_name=gpt-3.5-turbo-0301
# # Args for retrieval
# input_file=data/asqa_eval_gtr_top100.json
# retriever=instructor-large
# update_prompt_file=prompt12-update-using-missing-info-from-question-and-psgs
# update_query_using_missing_info_from_question_and_psgs=1
# # corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv
# # Args for generating used field.
# prompt_style=summary
# target_used_field=summary_use_sub
# max_tokens=150
# # Args for reranker
# position=head
# reranking_prompt_file=prompt5_select_no_up_to
# doc_num=50
# window_size=20
# # Args for filtration
# # threshold=4
# demo_file=prompts/${dataset_name}_demo.json
# filtration_prompt_file=prompt8_filter_question_with_demo
# filtration_method=judgment

# python Iterative_retrieval.py \
#     --max_iteration $max_iteration \
#     --dataset_name $dataset_name \
#     --use_sub_questions $use_sub_questions \
#     --use_title $use_title \
#     --used_doc_field $used_doc_field \
#     --openai_model_name $openai_model_name \
#     --input_file $input_file \
#     --retriever $retriever \
#     --update_prompt_file $update_prompt_file \
#     --update_query_using_missing_info_from_question_and_psgs $update_query_using_missing_info_from_question_and_psgs \
#     --corpus_path $corpus_path \
#     --prompt_style $prompt_style \
#     --target_used_field $target_used_field \
#     --max_tokens $max_tokens \
#     --position $position \
#     --reranking_prompt_file $reranking_prompt_file \
#     --doc_num $doc_num \
#     --window_size $window_size \
#     --demo_file $demo_file \
#     --filtration_prompt_file $filtration_prompt_file \
#     --filtration_method $filtration_method

# run_eval
# export CUDA_VISIBLE_DEVICES=7
# shot=1 
# openai_api=1 
# num_samples=1 
# data_file=iter_retrieval_50_retriever/asqa_final_data/final_data_instructor-large_max_iteration-4_prompt12-update-using-missing-info-from-question-and-psgs_head.json
# ndoc=5
# openai_multi_thread=10
# model=gpt-3.5-turbo-0301
# quick_test=0 
# seed=42 
# temperature=0 
# eval_metric=default 
# use_sub_questions=1
# # Other args
# dataset_name=asqa 

# prompt_file=prompts/${dataset_name}_default.json
# output_dir=iter_retrieval_50_retriever/${dataset_name}_max-4_instructor-large-prompt12_llm-select-head_run_eval
# mkdir $output_dir -p
# output_file=${output_dir}/run_output.json

# if [ ! -f "${output_file}" ]; then
#   echo "*****************************"
#   echo "lxn start run.py"
#   echo "*****************************"

#   python run.py \
#     --shot $shot \
#     --openai_api $openai_api \
#     --prompt_file $prompt_file \
#     --output_fp $output_file \
#     --dataset_name $dataset_name \
#     --num_samples $num_samples \
#     --data_file $data_file \
#     --ndoc $ndoc \
#     --openai_multi_thread $openai_multi_thread \
#     --model $model \
#     --quick_test $quick_test \
#     --seed $seed \
#     --temperature $temperature \
#     --use_sub_questions $use_sub_questions \
#     --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."

#   echo "*****************************"
#   echo "lxn finish run.py"
#   echo "*****************************"
# fi

# eval_f=${output_file%.json}
# eval_result_fp=${eval_f}.score
# if [ ! -f $eval_result_fp ]; then
#     echo "*****************************"
#     echo "lxn start eval.py"
#     echo "*****************************"
    
#     python eval.py \
#       --f $output_file \
#       --eval_metric $eval_metric

#     echo "*****************************"
#     echo "lxn finish eval.py"
#     echo "*****************************"
# fi


##################### qampari #####################
# # Iterative_retrieval
export CUDA_VISIBLE_DEVICES=2
max_iteration=4
dataset_name=qampari
use_sub_questions=0
use_title=0
used_doc_field=summary
openai_model_name=gpt-3.5-turbo-1106
# Args for retrieval
input_file=data/qampari_eval_gtr_top100.json
retriever=bge-large-en-v1.5
update_prompt_file=prompt12-update-using-missing-info-from-question-and-psgs
update_query_using_missing_info_from_question_and_psgs=1
# corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv
# Args for generating used field.
prompt_style=summary
target_used_field=summary
max_tokens=150
# Args for reranker
position=head
reranking_prompt_file=prompt5_select_no_up_to
doc_num=50
window_size=20
# Args for filtration
demo_file=prompts/${dataset_name}_default.json
filtration_prompt_file=prompt8_filter_question_with_demo
# threshold=9
filtration_method=judgment

python Iterative_retrieval.py \
    --max_iteration $max_iteration \
    --dataset_name $dataset_name \
    --use_sub_questions $use_sub_questions \
    --use_title $use_title \
    --used_doc_field $used_doc_field \
    --openai_model_name $openai_model_name \
    --input_file $input_file \
    --retriever $retriever \
    --update_prompt_file $update_prompt_file \
    --update_query_using_missing_info_from_question_and_psgs $update_query_using_missing_info_from_question_and_psgs \
    --corpus_path $corpus_path \
    --prompt_style $prompt_style \
    --target_used_field $target_used_field \
    --max_tokens $max_tokens \
    --position $position \
    --reranking_prompt_file $reranking_prompt_file \
    --doc_num $doc_num \
    --window_size $window_size \
    --filtration_prompt_file $filtration_prompt_file \
    --demo_file $demo_file \
    --filtration_method $filtration_method

# run_eval
export CUDA_VISIBLE_DEVICES=2
shot=1 
openai_api=1 
num_samples=1 
data_file=iter_retrieval_50/qampari_final_data/final_data_bge-large-en-v1.5_max_iteration-4_prompt12-update-using-missing-info-from-question-and-psgs_head.json
ndoc=5
openai_multi_thread=10
model=gpt-3.5-turbo-0301
quick_test=0 
seed=42 
temperature=0 
eval_metric=default 
# Other args
dataset_name=qampari 

prompt_file=prompts/${dataset_name}_default.json
output_dir=iter_retrieval_50/${dataset_name}_max-4_bge-large-en-v1.5-prompt12_llm-select-head_run_eval
mkdir $output_dir -p
output_file=${output_dir}/run_output_1106.json

if [ ! -f "${output_file}" ]; then
  echo "*****************************"
  echo "lxn start run.py"
  echo "*****************************"

  python run.py \
    --shot $shot \
    --openai_api $openai_api \
    --prompt_file $prompt_file \
    --output_fp $output_file \
    --dataset_name $dataset_name \
    --num_samples $num_samples \
    --data_file $data_file \
    --ndoc $ndoc \
    --openai_multi_thread $openai_multi_thread \
    --model $model \
    --quick_test $quick_test \
    --seed $seed \
    --temperature $temperature \
    --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."

  echo "*****************************"
  echo "lxn finish run.py"
  echo "*****************************"
fi

eval_f=${output_file%.json}
eval_result_fp=${eval_f}.score
if [ ! -f $eval_result_fp ]; then
    echo "*****************************"
    echo "lxn start eval.py"
    echo "*****************************"
    
    python eval.py \
      --f $output_file \
      --eval_metric $eval_metric

    echo "*****************************"
    echo "lxn finish eval.py"
    echo "*****************************"
fi


##################### eli5 #####################
# Iterative_retrieval
# export CUDA_VISIBLE_DEVICES=6
# max_iteration=4
# dataset_name=eli5
# use_sub_questions=0
# use_title=0
# used_doc_field=answer
# openai_model_name=gpt-3.5-turbo-0301
# # Args for retrieval
# input_file=iter_retrieval_50_hyde/eli5_input/eli5_eval_bm25_top100.json
# retriever=bm25
# update_prompt_file=prompt13-update-using-missing-info-from-question-and-psgs
# update_query_using_missing_info_from_question_and_psgs=1
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/faiss_index/sparse  # bm25-sphere-index
# # Args for generating used field.
# prompt_style=answer
# target_used_field=answer
# max_tokens=150
# # Args for reranker
# position=head
# reranking_prompt_file=prompt5_select_no_up_to
# doc_num=50
# # Args for filtration
# demo_file=prompts/${dataset_name}_default.json
# filtration_prompt_file=prompt8_filter_question_with_demo
# # threshold=7
# filtration_method=judgment

# python Iterative_retrieval.py \
#     --max_iteration $max_iteration \
#     --dataset_name $dataset_name \
#     --use_sub_questions $use_sub_questions \
#     --use_title $use_title \
#     --used_doc_field $used_doc_field \
#     --openai_model_name $openai_model_name \
#     --input_file $input_file \
#     --retriever $retriever \
#     --update_prompt_file $update_prompt_file \
#     --update_query_using_missing_info_from_question_and_psgs $update_query_using_missing_info_from_question_and_psgs \
#     --corpus_path $corpus_path \
#     --prompt_style $prompt_style \
#     --target_used_field $target_used_field \
#     --max_tokens $max_tokens \
#     --position $position \
#     --reranking_prompt_file $reranking_prompt_file \
#     --doc_num $doc_num \
#     --demo_file $demo_file \
#     --filtration_prompt_file $filtration_prompt_file \
#     --filtration_method $filtration_method

# # run_eval
# export CUDA_VISIBLE_DEVICES=6
# shot=1 
# openai_api=1 
# num_samples=1 
# data_file=iter_retrieval_50_hyde/eli5_final_data/final_data_bm25_max_iteration-4_prompt13-update-using-missing-info-from-question-and-psgs_head.json
# ndoc=5
# openai_multi_thread=6
# model=gpt-3.5-turbo-0301
# quick_test=0 
# seed=42 
# temperature=0 
# eval_metric=default 
# # Other args
# dataset_name=eli5 

# prompt_file=prompts/${dataset_name}_default.json
# output_dir=iter_retrieval_50_hyde/${dataset_name}_max-4_bm25-prompt13_llm-select-head_run_eval
# mkdir $output_dir -p
# output_file=${output_dir}/run_output.json

# if [ ! -f "${output_file}" ]; then
#   echo "*****************************"
#   echo "lxn start run.py"
#   echo "*****************************"

#   python run.py \
#     --shot $shot \
#     --openai_api $openai_api \
#     --prompt_file $prompt_file \
#     --output_fp $output_file \
#     --dataset_name $dataset_name \
#     --num_samples $num_samples \
#     --data_file $data_file \
#     --ndoc $ndoc \
#     --openai_multi_thread $openai_multi_thread \
#     --model $model \
#     --quick_test $quick_test \
#     --seed $seed \
#     --temperature $temperature \
#     --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."

#   echo "*****************************"
#   echo "lxn finish run.py"
#   echo "*****************************"
# fi

# eval_f=${output_file%.json}
# eval_result_fp=${eval_f}.score
# if [ ! -f $eval_result_fp ]; then
#     echo "*****************************"
#     echo "lxn start eval.py"
#     echo "*****************************"
    
#     python eval.py \
#       --f $output_file \
#       --eval_metric $eval_metric

#     echo "*****************************"
#     echo "lxn finish eval.py"
#     echo "*****************************"
# fi
