import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import json
from typing import Dict, List
import concurrent.futures
import threading
import tqdm
import openai
import time

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
                    logger.info('openai rate limit error, so retry after sleep 45 seconds')
                    logger.info(e)
                    time.sleep(45)
                    
            except openai.error.AuthenticationError as e:
                if 'This key is associated with a deactivated account' in e._message:
                    logger.info('the account {} is deactivated. so use next'.format(account[-1]))
                    logger.info(e)
                    account = account_manager.get_next_account(thread_id, account)
                    account_manager.thread_to_account[thread_id] = account
                else:
                    logger.info('meet unexpected AuthenticationError, so use next account.')
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
def call_chatgpt(model: str, messages: List[Dict], account_manager: "class", thread_id: int, temperature: float=0, max_tokens: int=150) -> str:
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
    temperature: float=0
        Temperature for decoding
    max_tokens: int=150
        Max tokens to generate.

    Returns
    -------
    ret: str
        Answer to prompt from ChatGPT.
    """
    completion = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    ret = completion['choices'][0]['message']['content']
    return ret


def get_messages(questions: str, prompt_style: str, doc: Dict) -> List[Dict]:
    """
    Args
    ----
    questions: str
        Given questions.
    prompt_style: str
        Set the type of user's content.
    doc: List[Dict]
        One of documents that belong to a problem.
    """
    if prompt_style == 'summary':
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': f"Summarize the following document within 50 words for the given question(s). Return \"irrelevant\" if the document is irrelevant to the question(s). Try to keep all the important dates, numbers, and names.\nQuestion(s):\n{questions}\n\nDocument:\nTitle: {doc['title']}\nText: {doc['text']}\n\nSummary:"}
        ]
    elif prompt_style == 'answer':
        messages = [
            {'role': 'system', 'content': "You are a helpful assistant."},
            {'role': 'user', 'content': f"Answer the given question(s) using the following document. Return \"irrelevant\" if the document is irrelevant to the question(s). Try to keep all the important dates, numbers, and names.\n\nQuestion(s):\n{questions}\n\nDocument:\nTitle: {doc['title']}\nText: {doc['text']}\n\nAnswer:"}
        ]
    else:
        raise NotImplementedError
    
    return messages


def gen_used_field_multi_thread(questions: str, 
                                prompt_style: str, 
                                doc: Dict, 
                                model_name: str="gpt-3.5-turbo-0301", 
                                target_used_field: str=None,
                                # target_used_field: Literal["summary_use_sub", "summary_no_sub", "summary", "answer"]=None,
                                temperature: float=0, 
                                max_tokens: int=150) -> None:
    """Generate used field by multithreading.
    Args
    ----
    questions: str
        Given questions.
    prompt_style: str
        Set the type of user's content.
    doc: Dict
        Single document.
    model_name: str="gpt-3.5-turbo-0301"
        OpenAI model.
    target_used_field: Literal["summary_use_sub", "summary_no_sub", "summary", "answer"]=None
        Which used field to generate
    temperature: float=0
        Temperature for OpenAI model.
    max_tokens int=150.
        Max tokens to generate.
    """
    thread_id = threading.current_thread().ident
    messages = get_messages(questions, prompt_style, doc)
    if model_name in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"]:
        ret = call_chatgpt(model=model_name, messages=messages, account_manager=account_manager, thread_id=thread_id, temperature=temperature, max_tokens=max_tokens)
    else:
        raise NotImplementedError
    doc[target_used_field] = ' '.join(ret.strip().split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate used doc field for passages.")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0301", required=True, help="What model to use")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument('--top_k', type=int, default=None, help="Generate used field for the top-k documents")
    parser.add_argument('--use_sub_questions', type=int, default=0, required=True, help="For asqa, whether to use sub questions")
    parser.add_argument('--target_used_field', type=str, required=True)
    parser.add_argument('--openai_multi_thread', type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for decoding")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--prompt_style", type=str, choices=['summary', 'extraction', 'answer'], required=True)
    args = parser.parse_args()

    # Save args
    args_dict = vars(args)
    directory = os.path.dirname(args.output_file)
    with open(f'{directory}/args.json', 'a') as f:
        json.dump(args_dict, f, indent=4)

    with open(args.data_file) as f:
        data = json.load(f)

    logger.info("Start generating used filed.")
    # Get OpenAI account manager
    account_manager = get_account_manager(multi_thread=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.openai_multi_thread) as executor:
        futures = []
        for d in data:
            if args.use_sub_questions and 'qa_pairs' in d:
                logger.warning("Load sub questions.")
                questions = list(map(lambda x: x['question'], list(d['qa_pairs'])))
            else:
                questions = [d["question"]]
            questions = '\n'.join(questions)

            for doc in d["docs"][:args.top_k]:
                future = executor.submit(
                    gen_used_field_multi_thread,
                    questions,
                    args.prompt_style,
                    doc,
                    args.model_name, 
                    args.target_used_field, 
                    args.temperature, 
                    args.max_tokens
                )
                futures.append(future)
    
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()

    with open(args.output_file, "w") as f:
            json.dump(data, f, indent=4)
    logger.info("Finish generating used filed.")
