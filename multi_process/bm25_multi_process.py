from typing import List, Dict
import torch.multiprocessing as mp
import queue
import math
import json
import time
import logging
from pyserini.search import LuceneSearcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BM25MultiProcess():
    def __init__(self, corpus_path: str=None, top_k: int=100):
        """
        Init class.
        """
        self.top_k = top_k
        self.corpus_path = corpus_path


    def start_multi_process_pool(self, process_num: int=8) -> Dict:
        """
        :param process_num: Number of process to use.
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        target_devices = ['cpu'] * process_num
        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for _ in target_devices:
            p = ctx.Process(target=BM25MultiProcess._multi_process_worker, args=(self, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}
    

    @staticmethod
    def _multi_process_worker(model, input_queue, results_queue) -> None:
        """
        Internal working process to retrieve documnents in multi-process setup.
        """
        searcher = LuceneSearcher(model.corpus_path)
        while True:
            try:
                id, queries = input_queue.get()
                docs_list = model.retrieve(queries, searcher)
                results_queue.put([id, docs_list])
            except queue.Empty:
                break


    def retrieve(self, queries: List[str], searcher: LuceneSearcher) -> List[List]:
        """
        Do retrieval using bm25.
        """
        docs_list = []
        for query in queries:
            start_time = time.time()
            try:
                hits = searcher.search(query, self.top_k)
            except Exception as e:
                if "maxClauseCount" in str(e):
                    query = " ".join(query.split())[:950]
                    hits = searcher.search(query, self.top_k)
                else:
                    raise e
                
            # For bm25 sphere
            docs = []
            for hit in hits:
                h = json.loads(str(hit.docid).strip())
                docs.append({
                    "title": h["title"],
                    "text": hit.raw,
                    "url": h["url"],
                    'score': hit.score,
                    'id':hit.docid
                })

            docs_list.append(docs)
            end_time = time.time()
            logger.warning(f"It took {end_time - start_time} seconds.")
        return docs_list


    def retrieve_multi_process(self, queries: List[str], pool: Dict[str, object], chunk_size: int=None) -> List[List]:
        """
        :param queries: List of queries
        :param pool: A pool of workers started with BM25MultiProcess.start_multi_process_pool
        :param chunk_size: Queries are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Retrieved documents.
        """
        if chunk_size is None:
            chunk_size = min(math.ceil(len(queries) / len(pool["processes"]) / 10), 5000)

        logger.info(f"Chunk queries into {math.ceil(len(queries) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for query in queries:
            chunk.append(query)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        docs_list = []
        for result in results_list:
            docs_list += result[1]
        return docs_list
    

    @staticmethod
    def stop_multi_process_pool(pool: Dict):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()
