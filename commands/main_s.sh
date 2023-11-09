#asqa数据集的命令
reversed_browse_order=0
selected_doc_first=1
window_size=20

system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=0,1,2,3 \
exp_tag=baseline \
dataset_name=asqa \
retriever=gtr \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=no \
ndoc=5 \
seed=42 \
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
used_doc_field_in_retrieval=summary_use_sub

bash commands/entire_pipeline_iterative_llm_select.sh


# qampari任务的命令
reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=0,1,2,3 \
exp_tag=baseline \
dataset_name=qampari \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=42 \
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
used_doc_field_in_retrieval=summary 

bash commands/entire_pipeline_run_on_filtration.sh


#eli5数据集的命令
reversed_browse_order=0
selected_doc_first=1
window_size=20

#system_prompt_file_name=prompt4_select_up_to_k
system_prompt_file_name=prompt5_select_no_up_to

export CUDA_VISIBLE_DEVICES=0,1,2,3 \
exp_tag=baseline \
dataset_name=eli5 \
retriever=bm25 \
reader_llm=gpt-3.5-turbo-0301 \
retriever_llm=gpt-3.5-turbo-0301 \
quick_test=0 \
num_samples=1 \
rerank_method=tmp_llm_select \
ndoc=5 \
seed=42 \
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
used_doc_field_in_retrieval=answer 

bash commands/entire_pipeline_run_on_filtration.sh
