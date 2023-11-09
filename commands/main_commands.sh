python eval.py --f {path/to/result/file} --citations --qa --mauve
python eval.py --f {path/to/result/file} --citations
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve

python run.py \
  --config configs/asqa_turbo_shot2_ndoc5_gtr_default.yaml \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."


python eval.py --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42.json --mauve

python run.py \
  --config configs/qampari_turbo_shot2_ndoc5_gtr_default.yaml \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations."



python eval.py --f result/qampari-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42.json --citations

python run.py \
  --config configs/asqa_turbo_shot2_ndoc5_gtr_default.yaml \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 4

python tools/rerank_outputs.py \
  --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json \
  --rerank_mode random

python eval.py --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json.rerank --mauve --citation

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/rerank_outputs.py \
  --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json \
  --rerank_mode prob

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/rerank_outputs.py \
  --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json \
  --rerank_mode discrete

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python eval.py --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json.rerank.prob --citations  --qa --mauve

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python eval.py --f result/asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json.rerank.discrete --citations  --qa --mauve

python retrieval.py --data {path/to/data} --retriever {bm25/gtr} --output_file {path/to/output}

python retrieval.py \
  --data_file data/asqa_eval_gtr_top100.json \
  --retriever bm25 \
  --output_file data/asqa_eval_bm25_top100_2.json \
  --doc_pool wiki

python retrieval.py \
  --data_file data/qampari_eval_gtr_top100.json \
  --retriever bm25 \
  --output_file data/qampari_eval_bm25_top100_2.json \
  --doc_pool wiki

python rerank.py \
  --data_file data/asqa_eval_gtr_top100.json \
  --output_file data/asqa_eval_gtr_top100_rank_gpt.json \
  --openai_model_name gpt-3.5-turbo-0301 \
  --openai_multi_thread 20



for config in \
asqa_turbo_shot2_ndoc5_gtr_default.yaml asqa_turbo_shot2_ndoc5_oracle_default.yaml \
qampari_turbo_shot2_ndoc5_gtr_default.yaml qampari_turbo_shot2_ndoc5_oracle_default.yaml \
eli5_turbo_shot2_ndoc5_bm25_default.yaml eli5_turbo_shot2_ndoc5_oracle_default.yaml \
asqa_turbo_shot2_ndoc5_bm25_default.yaml qampari_turbo_shot2_ndoc5_bm25_default.yaml
do

  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 4

done


export CUDA_VISIBLE_DEVICES=3,4,5,6
for result_fp in \
asqa-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42-sample4.json \
asqa-gpt-3.5-turbo-0301-oracle-shot2-ndoc5-42-sample4.json \
qampari-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json \
qampari-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42-sample4.json \
qampari-gpt-3.5-turbo-0301-oracle-shot2-ndoc5-42-sample4.json \
eli5-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42-sample4.json \
eli5-gpt-3.5-turbo-0301-oracle-shot2-ndoc5-42-sample4.json
do
for rerank_mode in discrete prob
do

python tools/rerank_outputs.py \
  --f result/$result_fp \
  --rerank_mode $rerank_mode

eval_f="result/${result_fp}.rerank.${rerank_mode}"
echo "lxn start eval $eval_f"
python eval.py --f $eval_f --default_eval_metric 1
echo "lxn finish eval $eval_f"

done
done


export CUDA_VISIBLE_DEVICES=0,1,2,3
for result_fp in \
asqa-gpt-3.5-turbo-0301-gtr-shot2-ndoc5-42-sample4.json
do
for rerank_mode in discrete
do

#python tools/rerank_outputs.py \
#  --f result/$result_fp \
#  --rerank_mode $rerank_mode
eval_f="result/${result_fp}.rerank.${rerank_mode}"
echo "lxn start eval $eval_f"
python eval.py --f $eval_f --default_eval_metric 1
echo "lxn finish eval $eval_f"
done
done





for config in asqa_turbo_shot2_ndoc5_bm25_default.yaml qampari_turbo_shot2_ndoc5_bm25_default.yaml
do

  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 1

done

export CUDA_VISIBLE_DEVICES=2,3,4,5
python eval.py --f result/asqa-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json --citations --qa --mauve --default_eval_metric 1
python eval.py --f result/qampari-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json --citations --default_eval_metric 1

vim result/asqa-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json.score
vim result/qampari-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json.score

#python eval.py --f eli5-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json --citations --claims_nli --mauve



python run.py \
  --config configs/qampari_turbo_shot2_ndoc5_gtr_default.yaml \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 1


export CUDA_VISIBLE_DEVICES=0,1,2,3
for result_fp in \
qampari-gpt-3.5-turbo-0301-oracle-shot2-ndoc5-42-sample4.json \
eli5-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42-sample4.json \
eli5-gpt-3.5-turbo-0301-oracle-shot2-ndoc5-42-sample4.json
do
for rerank_mode in discrete prob
do

eval_f="result/${result_fp}.rerank.${rerank_mode}"

if [ -e "$eval_f" ]; then
    echo "$result_fp has been reranked, so just eval it"
else
    echo "lxn start rerank $result_fp"
    python tools/rerank_outputs.py \
      --f result/$result_fp \
      --rerank_mode $rerank_mode
    echo "lxn finish rerank $result_fp"

fi


echo "lxn start eval $eval_f"
python eval.py --f $eval_f --default_eval_metric 1
echo "lxn finish eval $eval_f"

done
done


python rerank.py \
  --data_file data/asqa_eval_gtr_top100.json \
  --output_file data/asqa_eval_gtr_top100_rank_gpt.json \
  --openai_model_name gpt-3.5-turbo-0301 \
  --openai_multi_thread 20

python rerank.py \
  --data_file data/qampari_eval_gtr_top100.json \
  --retriever bm25 \
  --output_file data/qampari_eval_gtr_top100_rank_gpt.json \
  --doc_pool wiki


#for data_file in asqa_eval_gtr_top100 asqa_eval_bm25_top100_2 qampari_eval_gtr_top100 qampari_eval_bm25_top100_2
for data_file in asqa_eval_bm25_top100_2 asqa_eval_gtr_top100 qampari_eval_gtr_top100 qampari_eval_bm25_top100_2
do

python rerank.py \
  --data_file data/${data_file}.json \
  --output_file data/${data_file}_rank_gpt.json \
  --openai_model_name gpt-3.5-turbo-0301 \
  --openai_multi_thread 20

done

for config in \
asqa_turbo_shot2_ndoc5_bm25_default.yaml qampari_turbo_shot2_ndoc5_bm25_default.yaml
do

  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 1

done


asqa-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json
qampari-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json


export CUDA_VISIBLE_DEVICES=0,1,2,3
python eval.py --f result/asqa-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json --default_eval_metric 1
python eval.py --f result/qampari-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42.json --default_eval_metric 1

python retrieval.py \
  --data_file data/eli5_eval_bm25_top100.json \
  --retriever bm25 \
  --output_file data/eli5_eval_bm25_top100_2.json.json \
  --doc_pool sphere


for retriever in gtr bm25
do
for dataset in asqa qampari
do
export CUDA_VISIBLE_DEVICES=0,1,2,3
config=${dataset}_turbo_shot2_ndoc5_${retriever}_rank_gpt_default.yaml
eval_f=result/${dataset}-gpt-3.5-turbo-0301-${retriever}_rank_gpt-shot2-ndoc5-42.json
#eli5-gpt-3.5-turbo-0301-bm25-shot2-ndoc5-42-sample4.json \

if [ -e "$eval_f" ]; then
    echo "$result_fp has been reranked, so just eval it"
else
  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 1
fi

python eval.py --f $eval_f --default_eval_metric 1

done
done


for config in \
asqa_turbo_shot2_ndoc5_gtr_default.yaml asqa_turbo_shot2_ndoc5_oracle_default.yaml \
qampari_turbo_shot2_ndoc5_gtr_default.yaml qampari_turbo_shot2_ndoc5_oracle_default.yaml \
eli5_turbo_shot2_ndoc5_bm25_default.yaml eli5_turbo_shot2_ndoc5_oracle_default.yaml \
asqa_turbo_shot2_ndoc5_bm25_default.yaml qampari_turbo_shot2_ndoc5_bm25_default.yaml
do

  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 4

done

for data_file in eli5_eval_bm25_top100_2
do

python rerank.py \
  --data_file data/${data_file}.json \
  --output_file data/${data_file}_rank_gpt.json \
  --openai_model_name gpt-3.5-turbo-0301 \
  --openai_multi_thread 20

done

  config=eli5_turbo_shot2_ndoc5_bm25_rank_gpt_default.yaml
  python run.py \
  --config used_configs/$config \
  --openai_multi_thread 20 \
  --turbo_system_message "You are a helpful assistant that answers the following questions with proper citations." \
  --num_samples 1


  export CUDA_VISIBLE_DEVICES=1,2,3,45-42.json
  python eval.py --f $eval_f --default_eval_metric 1
  eval_f=result/eli5-gpt-3.5-turbo-0301-bm25_rank_gpt-shot2-ndoc

123

  export CUDA_VISIBLE_DEVICES=1
  eval_f=result/eli5-gpt-3.5-turbo-0301-bm25_rank_gpt-shot2-ndoc5-42.json
  python eval.py --f $eval_f --default_eval_metric 1


python retrieval.py \
  --data_file new_data/asqa_eval_gtr_top100.json \
  --retriever bm25 \
  --output_file data/asqa_eval_bm25_top500.json \
  --doc_pool wiki \
  --topk 500

python retrieval.py \
  --data_file new_data/asqa_eval_gtr_top100.json \
  --retriever gtr \
  --output_file data/asqa_eval_gtr_top500.json \
  --doc_pool wiki \
  --topk 500



python retrieval.py \
  --data_file new_data/qampari_eval_gtr_top100.json \
  --retriever bm25 \
  --output_file data/qampari_eval_bm25_top500.json \
  --doc_pool wiki \
  --topk 500

python retrieval.py \
  --data_file new_data/qampari_eval_gtr_top100.json \
  --retriever gtr \
  --output_file data/qampari_eval_gtr_top500.json \
  --doc_pool wiki \
  --topk 500


python retrieval.py \
  --data_file data/eli5_eval_bm25_top100.json \
  --retriever bm25 \
  --output_file data/eli5_eval_bm25_top500.json. \
  --doc_pool sphere \
  --topk 500


export CUDA_VISIBLE_DEVICES=1
exp_tag=debug \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0613 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
ndoc=5 \
seed=1 \
eval_metric=correctness \
bash commands/entire_pipeline.sh

export CUDA_VISIBLE_DEVICES=1
exp_tag=debug \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0613 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=5 \
seed=1 \
eval_metric=correctness \
bash commands/entire_pipeline.sh

123

4567


#以下为跑没有rerank和用两种turbo跑rank gpt和作为reader的组合的baseline


for retriever in bm25 gtr
do

for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in asqa qampari
do

export CUDA_VISIBLE_DEVICES=1
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh

for retriever_llm in gpt-3.5-turbo-0301 gpt-3.5-turbo-0613
do

export CUDA_VISIBLE_DEVICES=1
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh
done

done
done
done




retriever=bm25
for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in eli5
do

export CUDA_VISIBLE_DEVICES=1
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh

for retriever_llm in gpt-3.5-turbo-0301 gpt-3.5-turbo-0613
do

export CUDA_VISIBLE_DEVICES=1
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh
done

done
done

# 以下为测试ndoc=10相比于5的baseline

for retriever in bm25 gtr
do

for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in asqa qampari
do

export CUDA_VISIBLE_DEVICES=0
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh

done
done
done




retriever=bm25
for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in eli5
do

export CUDA_VISIBLE_DEVICES=0
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=10 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline.sh

done
done


#开始跑单组+retrieval noise的实验：


export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=asqa \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=5 \
ndoc_top_bottom=0 \
ndoc_top_neighbor=5 \
ndoc_random=0 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_noise_explore.sh

#狂跑加retrieval noise的baseline


export ndoc_top_bottom=0
export ndoc_top_neighbor=5
export ndoc_random=0

for retriever in bm25 gtr
do

for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in asqa qampari
do

export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_noise_explore.sh

for retriever_llm in gpt-3.5-turbo-0301
do

export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_noise_explore.sh
done

done
done
done




retriever=bm25
for reader_llm in gpt-3.5-turbo-0301
do

for dataset_name in eli5
do

export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=none \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_noise_explore.sh

for retriever_llm in gpt-3.5-turbo-0301
do

export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=$reader_llm \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_noise_explore.sh
done

done
done

#单独跑一组llm_select
export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=qampari \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=rank_gpt \
k=5 \
idf_use_letter=int \
use_title=1 \
reverse_doc_order=1 \
bash commands/entire_pipeline_llm_select.sh


llm_select_from=raw_retriever


#狂跑llm_select，观察各个超参数，k，idf_use_letter，use_title，reverse_doc_order

for llm_select_from in rank_gpt
do
for retriever in bm25 gtr
do

for dataset_name in asqa
do

for idf_use_letter in int upper lower
do

for use_title in 0 1
do

for reverse_doc_order in 0 1
do

export CUDA_VISIBLE_DEVICES=0
exp_tag=llm_select \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0613 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=$llm_select_from \
k=5 \
idf_use_letter=$idf_use_letter \
use_title=$use_title \
reverse_doc_order=$reverse_doc_order \
bash commands/entire_pipeline_llm_select.sh

done
done
done
done
done
done


for llm_select_from in rank_gpt
do
for retriever in bm25 gtr
do

for dataset_name in qampari
do

for idf_use_letter in int upper lower
do

for use_title in 0 1
do

for reverse_doc_order in 0 1
do

export CUDA_VISIBLE_DEVICES=1
exp_tag=llm_select \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0613 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=$llm_select_from \
k=5 \
idf_use_letter=$idf_use_letter \
use_title=$use_title \
reverse_doc_order=$reverse_doc_order \
bash commands/entire_pipeline_llm_select.sh

done
done
done
done
done
done


for llm_select_from in rank_gpt
do
for retriever in bm25
do

for dataset_name in eli5
do

for idf_use_letter in int upper lower
do

for use_title in 0 1
do

for reverse_doc_order in 0 1
do

export CUDA_VISIBLE_DEVICES=2
exp_tag=llm_select \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0613 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=$llm_select_from \
k=5 \
idf_use_letter=$idf_use_letter \
use_title=$use_title \
reverse_doc_order=$reverse_doc_order \
bash commands/entire_pipeline_llm_select.sh

done
done
done
done
done
done


for retriever in bm25 gtr
do

for retriever_llm in gpt-3.5-turbo-0301
do

for dataset_name in asqa qampari eli5
do

export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
llm_rerank_from=raw_retriever \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_rerank_from_rank_gpt.sh

export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=$retriever_llm \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
llm_rerank_from=rank_gpt \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_rerank_from_rank_gpt.sh

done
done
done


export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0613 \
quick_test=0 \
num_samples=1 \
rerank_method=rank_gpt \
llm_rerank_from=rank_gpt \
ndoc=5 \
seed=1 \
eval_metric=default \
bash commands/entire_pipeline_rerank_from_rank_gpt.sh


for reversed_browse_order in 0 1
do
for selected_doc_first in 1 0
do
for use_title in 0 1
do
for window_size in 15 20
do
#for use_title in 0 1
#do
export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh

done
done
done
done
#done


for reversed_browse_order in 0 1
do
for selected_doc_first in 1 0
do
for use_title in 0 1
do
for window_size in 15 20
do
#for use_title in 0 1
#do
export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=qampari \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh

done
done
done
done


for reversed_browse_order in 0 1
do
for selected_doc_first in 1 0
do

for use_title in 0 1
do
for window_size in 15 20
do
#for pad_with_raw_retrieval_result in 0 1
#do


export CUDA_VISIBLE_DEVICES=2
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh

done
done
done
done

export CUDA_VISIBLE_DEVICES=0
exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=1 \
window_size=20 \
reversed_browse_order=0 \
selected_doc_first=1 \
pad_with_raw_retrieval_result=1 \
bash commands/entire_pipeline_iterative_llm_select.sh


1233


for retriever in bm25 gtr
do

for rerank_method in no rank_gpt
do
for dataset_name in eli5 qampari asqa
do

export CUDA_VISIBLE_DEVICES=0
exp_tag=baseline \
dataset_name=$dataset_name \
retriever=$retriever \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=$rerank_method \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=1 \
window_size=20 \
reversed_browse_order=0 \
selected_doc_first=1 \
pad_with_raw_retrieval_result=0 \
reader_temperature=0.5 \
bash commands/entire_pipeline_iterative_llm_select.sh

done
done
done


for dataset_name in eli5 qampari
do
for retriever in bm25 gtr
do
for prompt_style in summary answer
do

python tools/main_gen_summary_multi_thread.py \
  --fp new_data/${dataset_name}_eval_${retriever}_top100/input.json \
  --fp_out new_data/${dataset_name}_eval_${retriever}_top100_${prompt_style}/input.json \
  --model_name gpt-3.5-turbo-0301 \
  --temperature 0 \
  --prompt_style $prompt_style \
  --max_tokens 150 \
  --ndoc 100 \
  --target_doc_field $prompt_style \
  --asqa_use_sub_questions 0 \
  --num_threads 40

done
done
done

for dataset_name in asqa
do
for retriever in bm25 gtr
do
for prompt_style in summary answer
do

python tools/main_gen_summary_multi_thread.py \
  --fp new_data/${dataset_name}_eval_${retriever}_top100/input.json \
  --fp_out new_data/${dataset_name}_eval_${retriever}_top100_${prompt_style}_no_sub/input.json \
  --model_name gpt-3.5-turbo-0301 \
  --temperature 0 \
  --prompt_style $prompt_style \
  --max_tokens 150 \
  --ndoc 100 \
  --target_doc_field $prompt_style \
  --asqa_use_sub_questions 0 \
  --num_threads 40

python tools/main_gen_summary_multi_thread.py \
  --fp new_data/${dataset_name}_eval_${retriever}_top100/input.json \
  --fp_out new_data/${dataset_name}_eval_${retriever}_top100_${prompt_style}_use_sub/input.json \
  --model_name gpt-3.5-turbo-0301 \
  --temperature 0 \
  --prompt_style $prompt_style \
  --max_tokens 150 \
  --ndoc 100 \
  --target_doc_field $prompt_style \
  --asqa_use_sub_questions 1 \
  --num_threads 40

done
done
done


reversed_browse_order=0
selected_doc_first=1
window_size=20

for use_title in 0 1
do

export CUDA_VISIBLE_DEVICES=4
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh

export CUDA_VISIBLE_DEVICES=4
exp_tag=baseline \
dataset_name=qampari \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh

export CUDA_VISIBLE_DEVICES=4
exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=$use_title \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
bash commands/entire_pipeline_iterative_llm_select.sh


done


asqa_eval_bm25_top100_answer_no_sub
asqa_eval_bm25_top100_answer_use_sub
asqa_eval_bm25_top100_summary_no_sub
asqa_eval_bm25_top100_summary_use_sub

asqa_eval_gtr_top100_answer_no_sub
asqa_eval_gtr_top100_answer_use_sub
asqa_eval_gtr_top100_summary_no_sub
asqa_eval_gtr_top100_summary_use_sub


qampari_eval_bm25_top100_answer
qampari_eval_bm25_top100_summary

qampari_eval_gtr_top100_answer
qampari_eval_gtr_top100_summary

eli5_eval_bm25_top100_answer
eli5_eval_bm25_top100_summary


reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=4
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=1 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=text \
bash commands/entire_pipeline_iterative_llm_select.sh


exp_tag=baseline \
dataset_name=qampari \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=0 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=text \
bash commands/entire_pipeline_iterative_llm_select.sh




exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=0 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=text \
bash commands/entire_pipeline_iterative_llm_select.sh

#试一下用summary检索


reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=0 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=answer \
bash commands/entire_pipeline_iterative_llm_select.sh


reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=qampari \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=0 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=summary \
bash commands/entire_pipeline_iterative_llm_select.sh


reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=3
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=1 \
eval_metric=default \
shot=1 \
llm_select_from=raw_retriever \
k=5 \
idf_use_letter=int \
use_title=1 \
window_size=$window_size \
reversed_browse_order=$reversed_browse_order \
selected_doc_first=$selected_doc_first \
pad_with_raw_retrieval_result=0 \
reader_temperature=0 \
system_prompt_file_name=$system_prompt_file_name \
used_doc_field_in_retrieval=summary_use_sub \
bash commands/entire_pipeline_iterative_llm_select.sh


CUDA_VISIBLE_DEVICES=4,5,6