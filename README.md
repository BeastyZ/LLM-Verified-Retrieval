# LLatrieval: LLM-Verified Retrieval for Verifiable Generation
This repository contains the code and data for paper [LLatrieval: LLM-Verified Retrieval for Verifiable Generation](https://arxiv.org/abs/2311.07838). This repository also includes code to reproduce the method we propose in our paper.

## :new:News
- **[2024/03/13]** Our submission to NAACL 2024, LLatrieval: LLM-Verified Retrieval for Verifiable Generation, has been accepted to the main conference.
- **[2023/11/14]** We have published the preprint version of the paper on [arXiv](https://arxiv.org/abs/2311.07838).
- **[2023/11/09]** We have released the code for reproducing our method.


## Quick Links
- [Requirements](#requirements)
- [Data](#data)
- [Code Structure](#code-structure)
- [Reproduce Our Method](#reproduce-our-method)
- [Citation](#citation)


## Requirements
1. We recommend that you use the python virtual environment and then install the dependencies.
    ```
    conda create -n lvr python=3.9.7
    ```
2. Next, activate the python virtual environment you just created.
    ```
    conda activate lvr
    ```
3. Finally, before running the code, make sure you have set up the environment and installed the required packages.
    ```
    pip install -r requirements.txt
    ```

## Data
We uploaded the data to [Hugging Face](https://huggingface.co/datasets/BeastyZ/LLM-Verified-Retrieval)🤗. 

**Start by installing 🤗 Datasets:**
```
pip install datasets
```

**Load a dataset**
```
from datasets import load_dataset

dataset = load_dataset("BeastyZ/LLM-Verified-Retrieval")
```
After downloading the data, you need to manually specify the data path. This may be a little difficult, so we recommend that you manually download the data from huaggingface to your current working directory. Perhaps you can refer to the following command
```bash
wget https://huggingface.co/datasets/BeastyZ/LLM-Verified-Retrieval/resolve/main/origin/asqa_eval_dpr_top100.json?download=true
```

**NOTE**

For the Sphere and Wikipedia snapshot corpora, please refer to [ALCE](https://github.com/princeton-nlp/ALCE) for more information.


## Code Structure
* `commands/`: folder that contains all shell files.
* `data/`: folder that contains all datasets.
* `llm_retrieval_prompt_drafts/`: folder that contains all prompt files.
* `llm_retrieval_related/`: folder that contains code for iteratively selecting supporting documents.
* `multi_process/`: folder that contains code for BM25 retrieval with multi-process support.
* `openai_account_files/`: folder that contains all OpenAI account files.
* `prompts/`: folder that contains all instruction and demonstration files.
* `eval.py`: eval file to evaluate generations.
* `Iterative_retrieval.py`: code for reproduce our method.
* `llm.py`: code for using LLM.
* `multi_thread_openai_api_call.py`: code for using gpt-3.5-turbo with multi-thread.
* `searcher.py`: code for retrieval using TfidfVectorizer.
* `run.py`: run file to generate citations.
* `utils.py`: file that contains auxiliary function.


## Reproduce Our Method
**NOTE:** There must be raw data and a corpus for retrieval before running the following commands. Once you have them, you also need to modify the parameters of the corresponding files in the `commands` directory.

For ASQA, use the following command
```bash
bash commands/asqa_iterative_retrieval.sh
```

For QAMPARI, use the following command
```bash
bash commands/qampari_iterative_retrieval.sh
```

For ELI5, use the following command
```bash
bash commands/eli5_iterative_retrieval.sh
```

The result will be saved in `iter_retrieval_50/`.


## Citation
```
@inproceedings{li-etal-2024-llatrieval,
    title = "{LL}atrieval: {LLM}-Verified Retrieval for Verifiable Generation",
    author = "Li, Xiaonan  and
      Zhu, Changtai  and
      Li, Linyang  and
      Yin, Zhangyue  and
      Sun, Tianxiang  and
      Qiu, Xipeng",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.305",
    pages = "5453--5471",
}
```
