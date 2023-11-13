import logging
from typing import Union
import openai
import fcntl
import threading
import tqdm
from multi_thread_openai_api_call import MyThread

logger = logging.getLogger()


class OpenAI_Account_Manager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance

    def __init__(self, used_account_fp, all_account_fp):
        self.used_account_fp = used_account_fp
        self.all_account_fp = all_account_fp

        used_account_f = open(used_account_fp, 'r')
        used_account = list(map(lambda x: x.strip().split('----'), used_account_f.readlines()))
        used_account_f.close()

        all_account_f = open(all_account_fp, 'r')
        all_account = list(map(lambda x: x.strip().split('----'), all_account_f.readlines()))
        all_account_f.close()

        used_account_key = list(map(lambda x: x[-1], used_account))

        all_account = list(filter(lambda x: x[-1] not in used_account_key, all_account))

        self.used_account = used_account
        self.all_account = all_account

        openai.api_key = self.all_account[0][-1]
        logger.info(
            'successfully build OpenAI_Account_Manager, now the number of available accounts is {} and now api_key is {}'.format(
                len(self.all_account), self.all_account[0][-1]))

    def use_next_account(self):
        self.used_account.append(self.all_account[0])
        del self.all_account[0]
        with open(self.used_account_fp, 'a') as tmp_used_account_f:
            fcntl.fcntl(tmp_used_account_f.fileno(), fcntl.LOCK_EX)
            print('----'.join(self.used_account[-1]), file=tmp_used_account_f)
            logger.info(
                'account:[{}, {}, {}] runs out. so use next.'.format(self.used_account[-1][0], self.used_account[-1][1],
                                                                     self.used_account[-1][2]))
        openai.api_key = self.all_account[0][-1]


class OpenAI_Account_Manager_MultiThread:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance

    def __init__(self, used_account_fp, all_account_fp):
        self.now_account_idx = 0


        self.used_account_fp = used_account_fp
        self.all_account_fp = all_account_fp

        used_account_f = open(used_account_fp, 'r')
        used_account = list(map(lambda x: x.strip().split('----'), used_account_f.readlines()))
        used_account_f.close()

        all_account_f = open(all_account_fp, 'r')
        all_account = list(map(lambda x: x.strip().split('----'), all_account_f.readlines()))
        all_account_f.close()

        used_account_key = list(map(lambda x: x[-1], used_account))

        all_account = list(filter(lambda x: x[-1] not in used_account_key, all_account))

        self.used_account = used_account
        self.all_account = all_account
        self.using_account = []

        # openai.api_key = self.all_account[0][-1]
        logger.info(
            'successfully build OpenAI_Account_Manager, now the number of available accounts is {} and now api_key is {}'.format(
                len(self.all_account), self.all_account[0][-1]))
        self.next_account_lock = threading.Lock()
        self.empty_account_lock = threading.Lock()

    def get_next_account(self, thread_id, last_empty_account=None):
        with self.next_account_lock:
            result = self.all_account[0]
            self.using_account.append(self.all_account[0])
            del self.all_account[0]
            if last_empty_account != None:
                self.record_empty_account(last_empty_account)
                logger.info('Thread {} account: [{}, {}, {}] '
                            'runs out'.format(thread_id,
                                              self.used_account[-1][0],
                                              self.used_account[-1][1],
                                              self.used_account[-1][2]))
                logger.info('Thread {} use next account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            else:
                logger.info('Thread {} first account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            # openai.api_key = self.all_account[0][-1]
            return result

    def record_empty_account(self, empty_account):
        with self.empty_account_lock:
            self.used_account.append(empty_account)
            with open(self.used_account_fp, 'a') as tmp_used_account_f:
                fcntl.fcntl(tmp_used_account_f.fileno(), fcntl.LOCK_EX)
                print('----'.join(self.used_account[-1]), file=tmp_used_account_f)


class OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used:
    '''
    OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used: when OpenAI_Account_Manager_MultiThread uses one account for one thread,
    so the number of threads is limited by the number of accounts.
    OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used support multiple threads using one account.
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

        return cls._instance


    def __init__(self, used_account_fp: str, all_account_fp: str, limit_account_num: int=-1) -> None:
        """Class init
        Args
        ----
        used_account_fp: str
            Path to file containing used OpenAI accounts.
        all_account_fp: str
            Path to file containing all OpenAI accounts.
        limit_account_num: int=-1
            Number of available accounts.
        """
        if hasattr(self, 'inited'):
            return
        self.inited = 1
        self.now_account_idx = 0

        self.used_account_fp = used_account_fp
        self.all_account_fp = all_account_fp

        used_account_f = open(used_account_fp, 'r')
        used_account = list(map(lambda x: x.strip().split('----'), used_account_f.readlines()))
        used_account_f.close()

        all_account_f = open(all_account_fp, 'r')
        all_account = list(map(lambda x: x.strip().split('----'), all_account_f.readlines()))
        all_account_f.close()

        used_account_key = []
        for account in used_account:
            if len(account) == 4:
                used_account_key.append(account[-2])
            else:
                used_account_key.append(account[-1])

        # Keep only usable account.

        all_account = list(filter(lambda x: x[-1] not in used_account_key, all_account))
        temp_all_account = []
        for account in all_account:
            if len(account) == 4 and account[-2] not in used_account_key:
                temp_all_account.append(account)
            elif len(account) == 3 and account[-1] not in used_account_key:
                temp_all_account.append(account)
            else:
                raise Exception
        all_account = temp_all_account

        if limit_account_num > 0:
            all_account = all_account[:limit_account_num]

        self.used_account = used_account
        self.used_account_key = set(used_account_key)
        self.all_account = all_account

        self.using_account = []
        self.thread_to_account = {}
        logger.info('successfully build OpenAI_Account_Manager, now the number of available accounts is {}'.format(len(self.all_account)))

        self.next_account_lock = threading.Lock()
        self.empty_account_lock = threading.Lock()


    def get_next_account(self, thread_id, last_empty_account=None):
        with self.next_account_lock:
            available_num = self.check_available_account_num()
            if available_num == 0:
                logger.info('all accounts used, so..')
                logger.info('all accounts used, so..')
                logger.info('all accounts used, so..')
                logger.info('all accounts used, so..')
                logger.info('all accounts used, so..')
            else:
                logger.info('now available accounts : {}'.format(available_num))

            while True:
                result = self.all_account[self.now_account_idx]
                if result[-1] in self.used_account_key or result[-2] in self.used_account_key:
                    self.now_account_idx += 1
                    self.now_account_idx = self.now_account_idx % len(self.all_account)
                else:
                    break

            result = self.all_account[self.now_account_idx]
            self.now_account_idx += 1
            self.now_account_idx = self.now_account_idx % len(self.all_account)

            if last_empty_account != None:
                self.record_empty_account(last_empty_account)
                logger.info('Thread {} account: [{}, {}, {}] '
                            'runs out'.format(thread_id,
                                              self.used_account[-1][0],
                                              self.used_account[-1][1],
                                              self.used_account[-1][2]))
                logger.info('Thread {} use next account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            else:
                logger.info('Thread {} first account: [{}, {}, {}] '
                            .format(thread_id, result[0],
                                    result[1],
                                    result[2]))
            return result


    def record_empty_account(self, empty_account):
        with self.empty_account_lock:
            self.used_account.append(empty_account)
            if len(empty_account) == 4:
                self.used_account_key.add(empty_account[-2])
            else:
                self.used_account_key.add(empty_account[-1])
            with open(self.used_account_fp, 'a') as tmp_used_account_f:
                fcntl.fcntl(tmp_used_account_f.fileno(), fcntl.LOCK_EX)
                print('----'.join(self.used_account[-1]), file=tmp_used_account_f)


    def check_available_account_num(self):
        available_num = 0
        for account in self.all_account:
            if len(account) == 4 and account[-2] not in self.used_account_key:
                available_num += 1
            elif len(account) == 3 and account[-1] not in self.used_account_key:
                available_num += 1
            else:
                raise Exception
        return available_num


def get_account_manager(
    account_file: str, 
    used_file: str, 
    multi_thread: bool=False, 
    limit_account_num: int=-1
) -> Union[OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used, OpenAI_Account_Manager]:
    """Get an instance of managing openai accounts.
    Args
    ----
    account_file: str
        The file containing available username, password and key of OpenAI API account.
    used_file: str
        The file containing unavailable username, password and key of OpenAI API account.
    multi_thread: bool=False
        Whether to use multi-thread or not.
    limit_account_num: int=-1
        Number of available accounts.

    Returns
    -------
    result: Union[OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used, OpenAI_Account_Manager]
        An instance of class OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used or OpenAI_Account_Manager
    """
    if multi_thread:
        result = OpenAI_Account_Manager_MultiThread_One_Acount_Many_Used(account_file, used_file, limit_account_num=limit_account_num)
    else:
        result = OpenAI_Account_Manager(account_file, used_file)
    return result


class OpenAI_API_inp_Manager_MultiThread:
    def __init__(self, idx_x_list_to_decode, inference_hyper_parameter):

        self.idx_x_list_to_decode = idx_x_list_to_decode

        self.inp_lock = threading.Lock()
        self.progress_index = 0

        assert type(inference_hyper_parameter) == type([])
        assert type(inference_hyper_parameter[0]) == type({})

        if len(inference_hyper_parameter) == 1:
            inference_hyper_parameter = inference_hyper_parameter * len(self.idx_x_list_to_decode)

        assert len(self.idx_x_list_to_decode) == len(inference_hyper_parameter), \
            'idx_x_list_to_decode:{}, inference_hyper_parameter:{}' \
            .format(len(idx_x_list_to_decode), len(inference_hyper_parameter))

        self.inference_hyper_parameter = inference_hyper_parameter

        for i in range(len(inference_hyper_parameter)):
            assert 'max_tokens' in inference_hyper_parameter[i], "{} th inference_hyper_parameter has no max_length"


    def get_next_gpt_idx_inp(self):
        with self.inp_lock:
            if self.progress_index < len(self.idx_x_list_to_decode):
                tmp_inp = self.idx_x_list_to_decode[self.progress_index]
                tmp_hyper_parameter = self.inference_hyper_parameter[self.progress_index]
                self.progress_index += 1
                return {'inp': tmp_inp, 'hyper_parameter': tmp_hyper_parameter}
            else:
                return None


def openai_llm_generate_multi_thread(eval_data_openai_queries, llm, num_threads, use_tqdm,turbo_system_message=None):
    # hyper_parameter = None
    x_list_to_decode = list(map(lambda x:x['input'],eval_data_openai_queries))
    max_tokens = list(map(lambda x:x['max_tokens'],eval_data_openai_queries))
    idx_x_list_to_decode = list(enumerate(x_list_to_decode))
    # eval_data_openai_queries = list(enumerate(eval_data_openai_queries))
    hyper_parameter = list(map(lambda x:{'max_tokens':x},max_tokens))

    inp_manager = OpenAI_API_inp_Manager_MultiThread(idx_x_list_to_decode, hyper_parameter)
    thread_list = []
    account_manager = get_account_manager(1)
    if use_tqdm:
        pbar = tqdm.tqdm(total=len(idx_x_list_to_decode))
    else:
        pbar = None
    for i in range(num_threads):
        thread_list.append(MyThread(i, llm, account_manager, inp_manager, 1, pbar, turbo_system_message))

    for t in thread_list:
        t.start()
    for i, t in enumerate(thread_list):
        t.join()

    responses_with_idx = []

    for t in thread_list:
        responses_with_idx.extend(t.responses_with_idx)

    responses_with_idx.sort(key=lambda x: x[0])

    responses = list(map(lambda x: x[1], responses_with_idx))
    return responses
