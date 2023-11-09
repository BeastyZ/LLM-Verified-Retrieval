################### asqa #####################
# export CUDA_VISIBLE_DEVICES=4,5,6 
# shot=1 
# openai_api=1 
# num_samples=1 
# data_file=new_data/asqa_bge-large-en-v1.5_top100/retriever_output.json
# ndoc=5 
# openai_multi_thread=20 
# model=gpt-3.5-turbo-0301 
# quick_test=0 
# seed=42 
# temperature=0 
# eval_metric=default 
# use_sub_questions=1
# # Other args
# dataset_name=asqa 

# prompt_file=prompts/${dataset_name}_default.json
# output_dir=new_data/${dataset_name}_bge-large-en-v1.5_run_eval
# mkdir $output_dir -p
# output_file=${output_dir}/run_output-use-sub.json

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

# # Remove .json
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


############# qampari #############
# export CUDA_VISIBLE_DEVICES=0,3
# shot=1 
# openai_api=1 
# num_samples=1 
# data_file=new_data/qampari_rest_bm25-query-using-missing-info-from-question-and-psgs_llm-select-tail/prompt12_reranker_output.json
# ndoc=5 
# openai_multi_thread=20 
# model=gpt-3.5-turbo-0301 
# quick_test=0 
# seed=42 
# temperature=0 
# eval_metric=default 
# # Other args
# dataset_name=qampari 

# prompt_file=prompts/${dataset_name}_default.json
# output_dir=new_data/${dataset_name}_rest_bm25-query-using-missing-info-from-question-and-psgs_llm-select-tail_run_eval
# mkdir $output_dir -p
# output_file=${output_dir}/prompt12_run_output.json

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


################### eli5 #####################
# run & eval
export CUDA_VISIBLE_DEVICES=4
shot=1 
openai_api=1 
num_samples=1 
data_file=new_data/eli5_bm25_bge-reranker-large/reranker_output-50.json
ndoc=5 
openai_multi_thread=10
model=gpt-3.5-turbo-0301
quick_test=0 
seed=42 
temperature=0 
eval_metric=default 
use_sub_questions=0
# Other args
dataset_name=eli5

prompt_file=prompts/${dataset_name}_default.json
output_dir=new_data/${dataset_name}_bm25_bge-reranker-large_run_eval
mkdir $output_dir -p
output_file=${output_dir}/run_output-50.json

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
    --use_sub_questions $use_sub_questions \
    --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."

  echo "*****************************"
  echo "lxn finish run.py"
  echo "*****************************"
fi

# Remove .json
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
