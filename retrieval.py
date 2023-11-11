import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import csv
import json
import os
import pickle
from typing import Dict, Union, List, Tuple
import openai
import re
import torch
from tqdm import tqdm
import time
import threading
import concurrent.futures
from sentence_transformers import SentenceTransformer
import numpy as np

from openai_account_manager import get_account_manager


def repeat_until_success_call_openai_api(func):
    def wrapper(*args, **kwargs):
        global account_manager
        while True:
            result = None
            # account_manager = kwargs.get('account_manager', None)
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
def call_chatgpt(model: str, messages: List[Dict], thread_id: int) -> str:
    """Call ChatGPT.
    Args
    ----
    model: str
        ID of the model to use.
    messages: List[Dict]
        A list of messages comprising the conversation so far.
    thread_id: int
        Thread ID

    Returns
    -------
    ret: str
        Answer to prompt from ChatGPT.
    """
    completion = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    ret = completion['choices'][0]['message']['content']
    return ret


def update_query(user_query: str, system_prompt: str, model: str) -> str:
    """Update query by ChatGPT.
    Args
    ----
    user_query: str
        Query for ChatGPT.
    system_prompt: str
        System prompt for ChatGPT.
    model: str
        ID of the model to use.

    Returns
    -------
    query: str
        Updated query.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    while True:
        try:
            ret = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
            break
        except Exception as e:
            logger.info(e)
    query = ret['choices'][0]['message']['content']
    return query


def update_query_multi_thread(user_query: str, system_prompt: str, model: str, d: Dict, field: str) -> None:
    """Update query by ChatGPT using multithreading.
    Args
    ----
    user_query: str
        Query for ChatGPT.
    system_prompt: str
        System prompt for ChatGPT.
    model: str
        ID of the model to use.
    d: Dict
        Single data.
    field: str
        Field for storage.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    thread_id = threading.current_thread().ident
    query = call_chatgpt(model=model, messages=messages, thread_id=thread_id)
    d[field] = query
    

# Save p_embeddings to local position.
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(embeddings, file, protocol=4)


# Load local p_embeddings
def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

