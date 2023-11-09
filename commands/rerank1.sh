# Dataset: eli5; Rerank: xwin
# export CUDA_VISIBLE_DEVICES=7
reranker=xwin
data_file=new_data/eli5_bm25-hyde_answer/answer_output.json
output_dir=new_data/eli5_bm25-hyde_xwin
top_k=5 
openai_multi_thread=5
used_doc_field=answer
use_title=0
use_sub_questions=0
openai_model_name=xwin
system_prompt_file_name=prompt5_select_no_up_to
window_size=20
doc_num=50
# position=tail
# old_data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json

mkdir $output_dir -p
output_file=${output_dir}/reranker_output.json

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


# run_eval
export CUDA_VISIBLE_DEVICES=3
shot=1 
openai_api=1 
num_samples=1 
data_file=new_data/eli5_bm25-hyde_xwin/reranker_output.json
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
output_dir=new_data/${dataset_name}_bm25-hyde_xwin_run_eval
mkdir $output_dir -p
output_file=${output_dir}/run_output.json

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