


#shot=1
openai_multi_thread=20
fitlog_dir=logs_lm_select_8_4

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
llm_select_from \
k \
idf_use_letter \
use_title \
reverse_doc_order \
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
  echo "not support rank_gpt in this file, you need use another bash script, so exit"
  exit

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

elif [ "$rerank_method" = "tmp_llm_select" ]; then

  if [ "$llm_select_from" = "raw_retriever" ]; then
    echo "lxn in llm_select_from='raw_retriever'"
    echo "lxn llm_select_from=$llm_select_from"
    llm_select_input_dir=$raw_retrieval_dir
  elif [ "$llm_select_from" = "rank_gpt" ]; then
    echo "lxn in llm_select_from='rank_gpt'"
    echo "lxn llm_select_from=$llm_select_from"
    rerank_retrieval_result_dir="${raw_retrieval_dir}_rank_gpt_gpt_3.5_turbo_0301"
    rerank_retrieval_result_data="${rerank_retrieval_result_dir}/input.json"
    llm_select_input_dir=$rerank_retrieval_result_dir
  else
    echo "not support llm_select_from=$llm_select_from, so exit"
    exit
  fi

  llm_select_input_data=$llm_select_input_dir/input.json
  llm_select_output_dir="${llm_select_input_dir}_select_${retriever_llm}_${k}_${idf_use_letter}_${use_title}_${reverse_doc_order}"
  mkdir -p $llm_select_output_dir
  llm_select_output_data=$llm_select_output_dir/input.json

  final_retrieval_result_dir=$llm_select_output_dir
  final_retrieval_result_data=$llm_select_output_data

  if [ -e $llm_select_output_data ]; then
    echo "llm_select result exists, so skip running llm_select"
  else

#  echo "llm_select_from=$llm_select_from"
#  echo "llm_select_input_data=$llm_select_input_data"
#  echo "llm_select_output_data=$llm_select_output_data"
#  exit

  python llm_retrieval_related/main_only_select_supporting_docs.py \
    --k $k \
    --idf_use_letter $idf_use_letter \
    --use_title $use_title \
    --reverse_doc_order $reverse_doc_order \
    --openai_model_name $real_retriever_llm \
    --data_file $llm_select_input_data \
    --output_file $llm_select_output_data \
    --openai_multi_thread $openai_multi_thread
  fi

else
  echo "now not support rerank_method=$rerank_method"
  exit
fi

echo "*****************************"
echo "lxn finish rerank retrieval"
echo "*****************************"


run_result_fp="${final_retrieval_result_dir}/${exp_tag}-${reader_llm}-shot${shot}-ndoc${ndoc}-seed${seed}-quick_test$quick_test"

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




python tools/fitlog_record_llm_select.py \
--fitlog_dir $fitlog_dir \
--raw_retrieval_data $raw_retrieval_data \
--eval_result_fp $eval_result_fp \
--final_retrieval_result_data $final_retrieval_result_data \
$required_args_to_record

