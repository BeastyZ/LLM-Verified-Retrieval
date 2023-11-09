#echo "exp_tag=$exp_tag"
#echo "dataset_name=$dataset_name"
#echo "retriever=$retriever"
#echo "reader_llm=$reader_llm"
#echo "retriever_llm=$retriever_llm"
#echo "use_llm_rerank=$use_llm_rerank"
#echo "quick_test=$quick_test"
#echo "num_samples=$num_samples"
#echo "rerank_method=$rerank_method"
#echo "ndoc=$ndoc"
#export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES


shot=1
openai_multi_thread=20
fitlog_dir=logs_explore

required_args=(\
exp_tag \
dataset_name \
retriever \
reader_llm \
retriever_llm \
quick_test \
num_samples \
rerank_method \
ndoc \
seed \
eval_metric \
shot \
)

pseudo_reader_llm=${reader_llm//-/_}
pseudo_reader_llm=${pseudo_reader_llm//\//_}

pseudo_retriever_llm=${retriever_llm//-/_}
pseudo_retriever_llm=${pseudo_retriever_llm//\//_}

real_reader_llm=$reader_llm
real_retriever_llm=$retriever_llm

reader_llm=$pseudo_reader_llm
retriever_llm=$pseudo_retriever_llm

if [ $num_samples -gt 1 ]; then
  echo "not support num_samples > 1, so exit"
  exit
fi


empty_flag=0

for arg in "${required_args[@]}"; do
  # 使用${!var}获取变量的值
  value="${!arg}"

  if [ -z "$value" ]; then
    # 如果变量为空，则打印变量名，并将empty_flag设置为1
    echo "    ****${arg} is empty ****"
    empty_flag=1
  else
    echo "${arg}=$value"
  fi

done

if [ $empty_flag -eq 0 ]; then
  echo "args are all provided"
else
  echo "args are not all provided, so exit"
  exit
fi

required_args_to_record=""
for arg in "${required_args[@]}"; do
  required_args_to_record="$required_args_to_record --${arg} ${!arg}"

done

echo "required_args_to_record="
echo "$required_args_to_record"

#python tools/fitlog_record.py \
#--fitlog_dir $fitlog_dir \
#--raw_retrieval_data $raw_retrieval_data \
#--eval_result_f $eval_result_f \
#--final_retrieval_result_data $final_retrieval_result_data \
#$required_args_to_record
#exit

#python tools/fitlog_record.py \
#--fitlog_dir $fitlog_dir \
#--raw_retrieval_data 2 \
#--eval_result_f 2 \
#--final_retrieval_result_data 2 \
#$required_args_to_record
#exit



prompt_file=prompts/${dataset_name}_default.json

raw_retrieval_dir="new_data/${dataset_name}_eval_${retriever}_top100"
mkdir $raw_retrieval_dir -p

raw_retrieval_data="${raw_retrieval_dir}/input.json"


echo "*****************************"
echo "lxn start rerank retrieval"
echo "*****************************"
if [ "$rerank_method" = "no" ]; then
  echo "rerank_method='no'"
  final_retrieval_result_data="$raw_retrieval_data"
  final_retrieval_result_dir="$raw_retrieval_dir"

elif [ "$rerank_method" = "rank_gpt" ]; then

  rerank_retrieval_result_dir="${raw_retrieval_dir}_rank_gpt_${retriever_llm}"
  mkdir $rerank_retrieval_result_dir -p

  rerank_retrieval_result_data="${rerank_retrieval_result_dir}/input.json"

  if [ -e $rerank_retrieval_result_data ]; then
    echo "rank_gpt result exists, so skip running rank_gpt"
  else
    python rerank.py \
  --data_file $raw_retrieval_data \
  --output_file $rerank_retrieval_result_data \
  --openai_model_name $real_retriever_llm \
  --openai_multi_thread $openai_multi_thread
  fi

  final_retrieval_result_data="$rerank_retrieval_result_data"
  final_retrieval_result_dir="$rerank_retrieval_result_dir"

else
  echo "now not support rerank_method=$rerank_method"
  exit
fi

echo "*****************************"
echo "lxn finish rerank retrieval"
echo "*****************************"


run_result_fp="${final_retrieval_result_dir}/${exp_tag}-${pseudo_reader_llm}-shot${shot}-ndoc${ndoc}-seed${seed}-quick_test$quick_test"

if [ ! -f "$run_result_fp" ]; then


echo "*****************************"
echo "lxn start run.py"
echo "*****************************"
python run.py \
--shot $shot \
--openai_api 1 \
--prompt_file $prompt_file \
--output_fp $run_result_fp \
--dataset_name $dataset_name \
--num_samples $num_samples \
--eval_file $final_retrieval_result_data \
--ndoc $ndoc \
--openai_multi_thread $openai_multi_thread \
--model $real_reader_llm \
--quick_test $quick_test \
--turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."

echo "*****************************"
echo "lxn finish run.py"
echo "*****************************"

fi






echo "*****************************"
echo "lxn start eval.py"
echo "*****************************"
eval_f=$run_result_fp
eval_result_fp=${eval_f}.score
#eval_f=result/eli5-gpt-3.5-turbo-0301-bm25_rank_gpt-shot2-ndoc

if [ ! -f $eval_result_fp ]; then
python eval.py \
--f $eval_f \
--eval_metric $eval_metric

fi

echo "*****************************"
echo "lxn finish eval.py"
echo "*****************************"




python tools/fitlog_record.py \
--fitlog_dir $fitlog_dir \
--raw_retrieval_data $raw_retrieval_data \
--eval_result_fp $eval_result_fp \
--final_retrieval_result_data $final_retrieval_result_data \
$required_args_to_record

