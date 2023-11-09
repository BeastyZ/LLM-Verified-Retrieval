# run_eval
export CUDA_VISIBLE_DEVICES=0
shot=1 
openai_api=1 
num_samples=1 
data_file=new_data/eli5_bm25-hyde_llm-select/reranker_output_0613_title-0-prompt5.json
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
output_dir=new_data/${dataset_name}_bm25-hyde_llm-select_run_eval
mkdir $output_dir -p
output_file=${output_dir}/0613_to_run_output_0301.json

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
