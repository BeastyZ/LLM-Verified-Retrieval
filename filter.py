import sys
sys.path.append("..")
sys.path.append(".")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse
import json
from typing import List, Dict, Tuple
import concurrent.futures
import tqdm
import openai
import time
import re
import threading
from datetime import datetime
import os

from llm_retrieval_related.iterative_select_supporting_documents import (
    create_stage2_select_prompt as get_messages,
)
from openai_account_manager import get_account_manager


def repeat_until_success_call_openai_api(func):
    def wrapper(*args, **kwargs):
        while True:
            result = None
            account_manager = kwargs.get('account_manager', None)
            thread_id = kwargs.get('thread_id', None)
            account = account_manager.thread_to_account.get('thread_id', None)  # triad
            if account is None:
                account = account_manager.get_next_account(thread_id)
            openai.api_key = account[-1]
            try:
                result = func(*args, **kwargs)

            except openai.error.APIConnectionError as e:
                logger.info('openai connection error, so retry after sleep 5 seconds')
                logger.info(e)
                time.sleep(5)

            except openai.error.RateLimitError as e:
                logger.info(type(e))
                logger.info(e)
                logger.info('e._message:{}'.format(e._message))
                if 'quota' in e._message:
                    logger.info('now openai account {} runs out. so use next.'.format(account[-1]))
                    logger.info(type(e))
                    logger.info(e)
                    
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('openai rate limit error, so retry after sleep 60 seconds')
                    logger.info(e)
                    time.sleep(60)
                    
            except openai.error.AuthenticationError as e:
                if 'This key is associated with a deactivated account' in e._message:
                    logger.info('the account {} is deactivated. so use next'.format(account[-1]))
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('meet unexpected AuthenticationError, so retry after sleep 5 seconds')
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account

            except openai.error.OpenAIError as e:
                logger.info('meet unexpected openai error, so retry after sleep 5 seconds')
                logger.info(e)
                time.sleep(5)

            except Exception as e:
                raise e

            if result != None:
                return result
            else:
                pass

    return wrapper


@repeat_until_success_call_openai_api
def call_chatgpt(model: str, messages: List[Dict], account_manager: "class", thread_id: int) -> str:
    """Call ChatGPT and get the judgment.
    Args
    ----
    model: str
        ID of the model to use.
    messages: List[Dict]
        A list of messages comprising the conversation so far.
    account_manager: "class"
        OpenAI account manager.
    thread_id: int
        Thread ID

    Returns
    -------
    judgment: str
        Answer to prompt from ChatGPT.
    """
    completion = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=64)
    judgment = completion['choices'][0]['message']['content']
    return judgment


def filter_bad_question(item: Dict, openai_model_name: str, judgment_system_prompt: str, k: int, idf_use_letter: str, use_title: int, used_doc_field: str) -> str:
    """Drop bad data that documents cannot fully support the question.
    Args
    ----
    item: Dict
        Single data.
    openai_model_name: str
        ID of the model to use.
    judgment_system_prompt: str
        System prompt for instruction.
    k: int
        K number of documents to be used.
    idf_use_letter: str
        Use uppercase letters, lowercase letters, or integers to mark the documents.
    use_title: int
        Whether to use title or not.
    used_doc_field: str
        Which filed of document to use.

    Returns
    -------
    judgment(socre_gpt): str
         Answer to prompt from ChatGPT.
    """
    if 'qa_pairs' in item:
        questions = list(map(lambda x: x['question'], list(item['qa_pairs'])))
    else:
        question = item['question']
        questions = [question]
    # Using top-k documents.
    docs = item["docs"][:k]

    thread_id = threading.current_thread().ident
    messages = get_messages(questions, docs, k, idf_use_letter, use_title, judgment_system_prompt, used_doc_field)
    judgment = call_chatgpt(model=openai_model_name, messages=messages, account_manager=account_manager, thread_id=thread_id)
    item["judgment"] = judgment

    return judgment


def get_demonstration(demo_data: Dict) -> str:
    """
    Args
    ----
    demo_data: Dict
        Data for creating demonstration prompt.
    
    Returns
    -------
    demos: str
        Demonstration prompt.
    """
    if 'qa_pairs' in demo_data:
        logger.warning("Load sub questions when getting demostration.")
        questions = list(map(lambda x: x['question'], list(demo_data['qa_pairs'])))
        question = "\n".join(questions)
        answer = demo_data["answer"]
    else:
        doc = demo_data["demos"][0]
        question = doc['question']
        answer = doc["answer"].replace('[1]', '').replace('[2]', '').replace('[3]', '').replace('[4]', '').replace('[5]', '')
    
    demos = f"Question: {question}\nAnswer: {answer}"
    return demos
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question filtering.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file.")
    parser.add_argument('--openai_multi_thread', type=int, required=True, help="Number of thread.")
    parser.add_argument('--openai_model_name', type=str, required=True)
    parser.add_argument('--system_prompt_file_name', type=str, required=True, help="The name of prompt file containing system prompt.")
    parser.add_argument('--used_doc_field', type=str, required=True, help="Which filed of document to use.")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, required=True, help="A specified number of documents for answering the user's specific question(s).")
    parser.add_argument('--idf_use_letter', type=str, required=True, help="Use uppercase letters, lowercase letters, or integers to mark the documents.")
    parser.add_argument('--use_title', type=int, required=True, help="Whether to use title or not.")
    parser.add_argument('--use_demo', type=int, help="Whether to use demonstration or not.")
    parser.add_argument('--demo_file', type=str, help="Path to demonstration file.")
    parser.add_argument('--use_sub_questions', type=int, default=0)
    parser.add_argument('--yes_output_file', type=str, required=True)
    parser.add_argument('--no_output_file', type=str, required=True)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()

    # Save args
    args_dict = vars(args)
    with open(f'{args.output_dir}/args.json', 'a') as f:
        json.dump(args_dict, f, indent=4)

    # Load data for judgment
    with open(args.data_file) as f:
        data = json.load(f)

    # Load system prompt
    judgment_system_prompt = ''.join(open('llm_retrieval_prompt_drafts/{}.md'.format(args.system_prompt_file_name)).readlines())

    # Load demonstration
    if args.use_demo:
        logger.warning("Use demonstration for filtration.")
        assert "with_demo" in args.system_prompt_file_name

        with open(args.demo_file) as f:
            demo_data = json.load(f)
        demos = get_demonstration(demo_data)
        judgment_system_prompt = judgment_system_prompt.replace("{Demo}", demos)

    logger.info('judgment_system_prompt')
    logger.info(judgment_system_prompt)

    logger.info('Start scoring')
    # Get OpenAI account manager
    account_manager = get_account_manager(multi_thread=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.openai_multi_thread) as executor:
        futures = []
        for item in data:
            future = executor.submit(
                filter_bad_question,
                item,
                args.openai_model_name,
                judgment_system_prompt,
                args.k,
                args.idf_use_letter,
                args.use_title,
                args.used_doc_field,
            )
            futures.append(future)
    
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            # Use judgment
            judgment = future.result()
            if "[YES]" in judgment:
                print(judgment)

    logger.info('Finish scoring')

    logger.info('Save wanted data')
    wanted_data = []
    dropped_data = []
    missing_cnt = 0
    dropping_cnt = 0
    for item in data:
        # Use judgment
        if "[YES]" in item["judgment"]:
            wanted_data.append(item)
        else:
            dropped_data.append(item)

    with open(args.yes_output_file, "w") as f:
        json.dump(wanted_data, f, indent=4)
    
    with open(args.no_output_file, "w") as f:
        json.dump(dropped_data, f, indent=4)
