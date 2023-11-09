################## asqa ##########################
# Dataset: asqa; Retriever: gtr
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=gtr 
# data_file=/remote-home/ctzhu/paper_repro/ALCE/data/asqa_eval_gtr_top100.json 
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/asqa_gtr_top100 
# top_k=100 
# use_sub_questions=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --use_sub_questions $use_sub_questions
# fi


# Dataset: asqa; Retriever: BAAI/bge-base-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=BAAI/bge-base-en-v1.5 
# data_file=/remote-home/ctzhu/paper_repro/ALCE/data/asqa_eval_gtr_top100.json 
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/asqa_bge-base-en-v1.5_top100 
# top_k=100 
# batch_size=512 
# use_sub_questions=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions
# fi


# Dataset: asqa; Retriever: BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=7
# retriever=BAAI/bge-large-en-v1.5 
# data_file=new_data/asqa_instructor-large-lamer_top100/retriever_output.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/asqa_bge-large-en-v1.5-lamer_top100
# top_k=100 
# batch_size=512 
# use_sub_questions=1
# openai_model_name=gpt-3.5-turbo-0301

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output_0301.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name
# fi


# Dataset: asqa; Retriever: hkunlp/instructor-base
# export CUDA_VISIBLE_DEVICES=0
# retriever=hkunlp/instructor-base
# data_file=/remote-home/ctzhu/paper_repro/ALCE/data/asqa_eval_gtr_top100.json 
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/asqa_instructor-base_top100 
# top_k=100 
# batch_size=256 
# use_sub_questions=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions
# fi


# Dataset: asqa; Retriever: contriever-msmarco
# export CUDA_VISIBLE_DEVICES=7
# retriever=facebook/contriever-msmarco
# data_file=new_data/asqa_instructor-large-hyde_top100/retriever_output.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/asqa_contriever-msmarco-hyde_top100
# top_k=100 
# batch_size=512 
# use_sub_questions=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python contriever_passage_retrieval.py \
#     --model_name_or_path $retriever \
#     --passages $corpus_path \
#     --passages_embeddings "embedding/contriever-msmarco_wikipedia_embeddings/wikipedia_embeddings/*" \
#     --data $data_file \
#     --output_file $output_file \
#     --use_sub_questions $use_sub_questions
# fi


################ qampari #########################
# Dataset: qampari; Retriever: BAAI/bge-base-en-v1.5
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=BAAI/bge-base-en-v1.5 
# data_file=/remote-home/ctzhu/paper_repro/ALCE/data/qampari_eval_gtr_top100.json 
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/qampari_bge-base-en-v1.5_top100 
# top_k=100 
# batch_size=512

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size
# fi


# Dataset: qampari; Retriever: BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=6
# retriever=BAAI/bge-large-en-v1.5 
# data_file=iter_retrieval_50/qampari_max-4_bge-large-en-v1.5-prompt12-update-using-missing-info-from-question-and-psgs_summary_llm-select-head_filtration/filtration_output_iter-0_no.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=iter_retrieval_50/qampari_bge-large-en-v1.5_top50
# top_k=50 
# batch_size=512 
# use_sub_questions=0
# update_query_using_missing_info_from_question_and_psgs=1
# openai_model_name=gpt-3.5-turbo-0301
# system_prompt_file_name=prompt12-update-using-missing-info-from-question-and-psgs

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output-iter-0-no.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions \
#         --update_query_using_missing_info_from_question_and_psgs $update_query_using_missing_info_from_question_and_psgs \
#         --openai_model_name $openai_model_name \
#         --system_prompt_file_name $system_prompt_file_name
# fi

# Dataset: qampari; Retriever: bm25; Residual data for retrieval using new query updated by question.
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=bm25 
# data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json
# corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
# output_dir=new_data/qampari_rest_bm25_top100 
# top_k=100 
# doc_pool=wiki
# update_query_using_question=1
# system_prompt_file_name=prompt9_update_using_question

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --update_query_using_question $update_query_using_question \
#         --system_prompt_file_name $system_prompt_file_name
# fi


# Dataset: qampari; Retriever: bm25; Residual data for retrieval using new query(passage) updated by question.
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=bm25 
# data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json
# corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
# output_dir=new_data/qampari_rest_bm25-query-using-passage_top100 
# top_k=100 
# doc_pool=wiki
# update_query_using_passage=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --update_query_using_passage $update_query_using_passage
# fi


# Dataset: qampari; Retriever: bm25; Residual data for retrieval using psg from question and psgs.
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=bm25 
# data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json
# corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
# output_dir=new_data/qampari_rest_bm25-query-using-psg-from-question-and-psgs_top100 
# top_k=100 
# doc_pool=wiki
# update_query_using_psg_from_question_and_psgs=1

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --update_query_using_psg_from_question_and_psgs $update_query_using_psg_from_question_and_psgs
# fi


# Dataset: qampari; Retriever: bm25; Residual data for retrieval using missing info from question and psgs.
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=bm25 
# data_file=new_data/qampari_bm25_llm-select_filter_demo/no_filter_output.json
# corpus_path=/remote-home/share/xnli/to_zct/psgs_w100.index  # bm25-wiki-index
# output_dir=new_data/qampari_rest_bm25-query-using-missing-info-from-question-and-psgs_top100 
# top_k=100 
# doc_pool=wiki
# update_query_using_missing_info_from_question_and_psgs=1
# system_prompt_file_name=xnli_update_using_missing_info_from_question_and_psgs

# mkdir $output_dir -p
# output_file=${output_dir}/retriever-xnli_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --update_query_using_missing_info_from_question_and_psgs $update_query_using_missing_info_from_question_and_psgs \
#         --system_prompt_file_name $system_prompt_file_name
# fi


# Dataset: qampari; Retriever: contriever
# export CUDA_VISIBLE_DEVICES=3,4
# retriever=facebook/contriever
# data_file=data/qampari_eval_gtr_top100.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/qampari_contriever_top100
# top_k=100 
# batch_size=512 
# use_sub_questions=0

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python contriever_passage_retrieval.py \
#     --model_name_or_path $retriever \
#     --passages $corpus_path \
#     --passages_embeddings "embedding/contriever_wikipedia_embeddings/wikipedia_embeddings/*" \
#     --data $data_file \
#     --output_file $output_file \
#     --use_sub_questions $use_sub_questions
# fi


# Dataset: qampari; Retriever: contriever-msmarco
export CUDA_VISIBLE_DEVICES=7
retriever=facebook/contriever-msmarco
data_file=new_data/qampari_bge-large-en-v1.5-hyde_top100/retriever_output_0301.json
corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
output_dir=new_data/qampari_contriever-msmarco-hyde_top100
top_k=100 
batch_size=512 
use_sub_questions=0

mkdir $output_dir -p
output_file=${output_dir}/retriever_output_0301.json

if [ ! -f "${output_file}" ]; then
    python contriever_passage_retrieval.py \
    --model_name_or_path $retriever \
    --passages $corpus_path \
    --passages_embeddings "embedding/contriever-msmarco_wikipedia_embeddings/wikipedia_embeddings/*" \
    --data $data_file \
    --output_file $output_file \
    --use_sub_questions $use_sub_questions
fi


# Dataset: qampari; Retriever: hkunlp/instructor-base
# export CUDA_VISIBLE_DEVICES=2
# retriever=hkunlp/instructor-base
# data_file=data/qampari_eval_gtr_top100.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/qampari_instructor-base_top100
# top_k=100 
# batch_size=256 
# use_sub_questions=0

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions
# fi


# Dataset: qampari; Retriever: hkunlp/instructor-large
# export CUDA_VISIBLE_DEVICES=3
# retriever=hkunlp/instructor-large
# data_file=data/qampari_eval_gtr_top100.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv 
# output_dir=new_data/qampari_instructor-large_top100
# top_k=100 
# batch_size=256 
# use_sub_questions=0

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --batch_size $batch_size \
#         --use_sub_questions $use_sub_questions
# fi


# Dataset: qampari; Retriever: hyde/lamer BAAI/bge-large-en-v1.5
# export CUDA_VISIBLE_DEVICES=7
# retriever=BAAI/bge-large-en-v1.5
# method=lamer
# data_file=data/qampari_eval_gtr_top100.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/corpus/psgs_w100.tsv
# output_dir=new_data/qampari_bge-large-en-v1.5-lamer_top100
# top_k=100 
# doc_pool=wiki
# use_sub_questions=0
# openai_model_name=gpt-3.5-turbo-0301

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output_0301.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --method $method \
#         --use_sub_questions $use_sub_questions \
#         --openai_model_name $openai_model_name
# fi


################ eli5 #########################
# Dataset: eli5; Retriever: bm25; Residual data for retrieval using new query updated by question.
# export CUDA_VISIBLE_DEVICES=4,5,6,7 
# retriever=bm25 
# data_file=new_data/eli5_bm25_llm-select_filter_demo/no_filter_output.json
# corpus_path=/remote-home/ctzhu/paper_repro/ALCE/faiss_index/sparse
# output_dir=new_data/eli5_rest_bm25_top100 
# top_k=100 
# doc_pool=sphere
# update_query_using_question=1
# system_prompt_file_name=prompt9_update_using_question

# mkdir $output_dir -p
# output_file=${output_dir}/retriever_output.json

# if [ ! -f "${output_file}" ]; then
#     python retrieval.py \
#         --retriever $retriever \
#         --data_file $data_file \
#         --corpus_path $corpus_path \
#         --output_file $output_file \
#         --top_k $top_k \
#         --doc_pool $doc_pool \
#         --update_query_using_question $update_query_using_question \
#         --system_prompt_file_name $system_prompt_file_name
# fi