import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import re
import yaml
from utils import *
from nltk import sent_tokenize
from openai_account_manager import openai_llm_generate_multi_thread
from llm import LLM

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndoc_top_bottom', type=int, default=0)
    parser.add_argument('--ndoc_top_neighbor', type=int, default=0)
    parser.add_argument('--output_fp', default=None, type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument('--openai_multi_thread',type=int,required=True)
    parser.add_argument('--turbo_system_message', required=True)
    parser.add_argument('--use_sub_questions', type=int, default=0)
    # Prompt file is a json file that contains the following fields:
    # - instruction: the instruction, which will appear at the beginning of each demo and the test example
    # - demo_sep: the separator between each demo, for example, "\n\n\n"
    # - demo_prompt: the prompt for the demo, for example, "Instruction: {INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
    #     - {INST}: the instruction
    #     - {D}: the documents
    #     - {Q}: the question
    #     - {A}: the answers
    # - doc_prompt, the prompt for each document, for example, "Document [{ID}](Title: {T}): {P}", where
    #     - {ID}: the document id, staring from 1
    #     - {T}: the document title
    #     - {P}: the document text
    # - demos: a list of demo examples, each of which should have
    #     - question: the question
    #     - docs: the documents ("title" and "text")
    #     - answer: the answer to show in the demo. If it is a list, they will be concatenated by "\n". This is useful when the answer includes interactive components.
    # Note that this python file will sample `--shot` demos from the prompt file given the random seed `--seed`
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    # - question: the question
    # - answer: the answer
    # - docs: the documents, each of which contains "title", "text"
    parser.add_argument("--data_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=0, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, required=True, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    # Interactive
    parser.add_argument("--interactive", type=bool, default=False, help="Whether to run in interactive mode")
    parser.add_argument("--interactive_query", type=str, default=None, help="The query to use in interactive mode, either `doc_id` (corresponding to interact in paper) or `search` (corresponding to inlinesearch in paper).")
    parser.add_argument("--retriever", type=str, default=None, help="When using interactive search mode, which retriever to use. Options: `tfidf`, `gtr-t5-large`")
    parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")
    parser.add_argument("--max_turn", type=int, default=10, help="Max number of all actions")
    parser.add_argument("--max_doc_show", type=int, default=3, help="Max number of documents to show at one time.")
    parser.add_argument("--force_cite_show", type=bool, default=False, help="Force citing the documents that are shown to the model")

    # Load config
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    assert args.openai_api, "only support openai_api now"
    assert not args.azure, "not support azure"
    assert not args.interactive, "not support interactive"
    assert not args.no_doc_in_demo
    assert not args.fewer_doc_in_demo

    # Save args
    args_dict = vars(args)
    directory = os.path.dirname(args.output_fp)
    with open(f'{directory}/args.json', 'a') as f:
        json.dump(args_dict, f, indent=4)

    if args.num_samples > 1:
        assert args.temperature > 0, "when multiple sampling, do not use temperature=0, i.e., greedy decoding"
    # assert args.num_samples == 1, "not support num_samples>1"

    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    if "turbo" in args.model:
        # ChatGPT has a longer max length
        logger.info("Change the max length to 4096 for ChatGPT.")
        args.max_length = 4096

    # Load the model or setup the API
    llm = LLM(args)

    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.data_file))

    logger.info("Generate the demonstration part")
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        if args.no_doc_in_demo:
            ndoc = 0
        elif args.fewer_doc_in_demo:
            assert args.ndoc_in_demo is not None
            ndoc = args.ndoc_in_demo
        # Run here
        head_prompt += make_demo(
            train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter, test=False, use_sub_questions=args.use_sub_questions
        )
        head_prompt += prompt_data["demo_sep"]

    # Sample quick test
    if args.quick_test > 0:  # Don't run
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...")
    incomplete_doc_list = 0 # For some questions there might be less than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        eval_data[idx]['prompt'] = head_prompt + make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter, test=True, use_sub_questions=args.use_sub_questions
        )
        if args.use_shorter is not None:
            doc_list = get_shorter_text(eval_item, eval_item["docs"], args.ndoc, args.use_shorter)
        else:
            doc_list = eval_item["docs"][:args.ndoc]

            if args.ndoc_top_bottom > 0:
                doc_list += eval_item["docs"][-args.ndoc_top_bottom:]
            if args.ndoc_top_neighbor > 0:
                doc_list += eval_item['docs'][30:30+args.ndoc_top_neighbor]

            assert not (args.ndoc_top_bottom > 0 and args.ndoc_top_neighbor > 0), 'not support args.ndoc_top_neighbor and args.ndoc_top_bottom both > 0'

        if not args.retrieve_in_all_docs:
            # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
            # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
            eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    # Load retriever for interactive search 
    if args.interactive and args.interactive_query == "search" and "gtr" in args.retriever:  # Don't run
        from sentence_transformers import SentenceTransformer
        gtr_model = SentenceTransformer(f'sentence-transformers/{args.retriever}', device=args.retriever_device)
        from searcher import SearcherWithinDocs

    eval_data_openai_queries = []

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))
        if idx == 0:
            print(prompt)
        eval_data_openai_queries.append({'input': prompt, 'max_tokens': min(args.max_new_tokens, args.max_length - prompt_len)})

        if "turbo" in args.model and not args.azure:  # Run
            assert args.turbo_system_message != None
            # For OpenAI's ChatGPT API, we need to convert text prompt to chat prompt
            item['prompt'] = [
                {'role': 'system', 'content': args.turbo_system_message},
                {'role': 'user', 'content': prompt}
            ]

    if args.openai_multi_thread > 1:
        eval_data_openai_responses = openai_llm_generate_multi_thread(eval_data_openai_queries,
                                                                    llm,
                                                                    args.openai_multi_thread,
                                                                    1,
                                                                    args.turbo_system_message)
    else:
        raise NotImplementedError

    for idx, item in enumerate(tqdm(eval_data)):
        eval_data_openai_response = eval_data_openai_responses[idx]
        for j, decoded_output in enumerate(eval_data_openai_response):
            decoded_output = decoded_output.replace("<|im_end|>", "").rstrip()
            if decoded_output.endswith("End."):
                decoded_output = decoded_output[:-len("End.")]
            eval_data_openai_response[j] = decoded_output

        logger.info(f"Question: {item['question']}")
        logger.info(f"Gold answer: {item['answer']}")
        logger.info(f"Final model output:")
        for j, decoded_output in enumerate(eval_data_openai_response):
            print('{}: {}'.format(j,decoded_output))
        item['output'] = eval_data_openai_response if len(eval_data_openai_response) > 1 else eval_data_openai_response[0]


    # for idx, item in enumerate(tqdm(eval_data)):
    for idx, item in enumerate([]):

        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            if args.interactive:
                print("============ Interactive =============")
                output_answer = ""
                doc_list = item['docs']

                interactive_prompt = prompt.rstrip() + "\n" # Start a new line
                inline_doc = ""
                num_turn = 0
                
                doc_history = []
                while True:
                    # For each action, it should end at the new line
                    # Three possible actions
                    # - Check: Document [1][2][3] / search query
                    # - Output: output 
                    # - End
                    num_turn += 1
                    new_prompt = interactive_prompt + inline_doc
                    new_prompt_len = len(llm.tokenizer.tokenize(new_prompt))

                    if idx == 0:
                        print(f"-------------- Step {num_turn} prompt --------------")
                        print(new_prompt)
                        print("-----------------------------")

                    output = llm.generate(new_prompt, min(args.max_new_tokens, args.max_length-new_prompt_len), stop=["\n", "\n\n"])

                    if len(inline_doc) > 0:
                        output = "Output: " + output # "Output: " was included in inline_doc
                    inline_doc = "" # Delete inline_doc after use
                    interactive_prompt += output + "\n"
                    logger.info(f"Model output: \"{output}\"")

                    if output.strip().lower()[:3] == "end":
                        # Model decides to end the generation
                        break
                    elif "sorry" in output.lower() and ("relevant document" in output.lower() or "relevant information" in output.lower()) or "none of the documents" in output.lower():
                        # Instruction-tuned model may abstain from answer the question
                        break
                    elif output.strip().lower()[:5] == "check" or output.strip().lower()[:6] == "search":
                        # Checkout or search documents
                        if args.interactive_query == "search":
                            query = output.replace("Search:", "").replace("search:", "").strip()
                            if len(doc_list) == 0:
                                show_doc_ids = []
                            else:
                                searcher = SearcherWithinDocs(doc_list, args.retriever, model=gtr_model, device=args.retriever_device)
                                show_doc_ids = [int(searcher.search(query))]
                        elif args.interactive_query == "doc_id":
                            show_doc_ids = [int(r[1:])-1 for r in re.findall(r"\[\d+", output)] # In text citation id starts from 1
                            show_doc_ids = [doc_id for doc_id in show_doc_ids if doc_id < len(doc_list) and doc_id >= 0]
                            show_doc_ids = show_doc_ids[:args.max_doc_show] # Avoiding showing too many documents
                        else:
                            raise NotImplementedError

                        inline_doc = "".join([make_doc_prompt(doc_list[doc_id], doc_id, prompt_data["doc_prompt"]) for doc_id in show_doc_ids])
                        inline_doc += "Output:" # Force the model to generate output in the next step
                        doc_history.append(show_doc_ids)
                    elif output.strip().lower()[:6] == "output":
                        output = output.strip().replace("Output:", "").strip()
                        if args.force_cite_show:
                            output = remove_citations(output)
                            if len(doc_history) == 0:
                                logger.warn("No doc history??")
                            else:
                                # Just cite whatever documents the model has seen in the last step
                                if "qampari" in args.data_file:
                                    output = ", ".join(["".join([f"[{doc+1}]" for doc in doc_history[-1]]) + " " + entity.strip() for entity in output.rstrip().rstrip(",").split(",")]) + ", "
                                else:
                                    output = " ".join(["".join([f"[{doc+1}]" for doc in doc_history[-1]]) + " " + o for o in sent_tokenize(output)]) + "."
                        output_answer += " " + output 
                    else:
                        # Sometimes model starts to output random things.
                        break
                    
                    if num_turn >= args.max_turn:
                        logger.warning("Reach maximum number of turns. Terminate now.")
                        break
                
                if "qampari" in args.data_file:
                    output_answer = output_answer.rstrip().rstrip(",")
                output_array.append(output_answer)
                item['prompt'] = interactive_prompt
                item['doc_history'] = doc_history
            else: 
                output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                item['prompt'] = prompt
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Prompt length={prompt_len}")
            logger.info(f"Question: {item['question']}")
            logger.info(f"Gold answer: {item['answer']}")
            logger.info(f"Final model output: {output_array[-1]}")
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    # Calculate the price for OpenAI API
    if args.openai_api:
        logger.info(f"Total token used: {llm.total_tokens}")
        if "turbo" in args.model:
            unit_price = 0.002
        else:
            unit_price = 0.02
        logger.info(f"Unit price: {unit_price}")
        logger.info(f"Total cost: %.1f" % (llm.total_tokens / 1000 * unit_price))
    
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    # if "/" in model_name:
    #     model_name = model_name.split("/")[-1]
    os.makedirs('exps',exist_ok=True)
    # name = f"exps/{args.tag}-{args.dataset_name}-{model_name.replace('/','_').replace('-','_')}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"
    if args.azure:
        name += "-azure"
    if args.quick_test > 0:
        name += f"-quick_test{args.quick_test}"
    if args.no_doc_in_demo:
        name += "-no_doc_in_demo"
    if args.fewer_doc_in_demo:
        name += f"-{args.ndoc_in_demo}_doc_in_demo"
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if args.force_cite_show:
        name += f"-forceciteshow"

    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    if args.openai_api:
        eval_data["total_cost"] = llm.total_tokens / 1000 * unit_price
        if args.azure:
            eval_data["azure_filter_fail"] = llm.azure_filter_fail 

    if args.output_fp != None:
        name = args.output_fp
    else:
        if not os.path.exists("result"):
            os.makedirs("result")
            name = "result/" + name + ".json"   

    logger.info('output_fp:{}'.format(name))
    json.dump(eval_data, open(name, "w"), indent=4)

if __name__ == "__main__":
    main()
