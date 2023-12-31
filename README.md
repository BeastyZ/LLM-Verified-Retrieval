# LLatrieval: LLM-Verified Retrieval for Verifiable Generation
This repository contains the code and data for paper [LLatrieval: LLM-Verified Retrieval for Verifiable Generation](https://arxiv.org/abs/2311.07838). This repository also includes code to reproduce the method we propose in our paper.


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
@misc{li2023llatrieval,
      title={LLatrieval: LLM-Verified Retrieval for Verifiable Generation}, 
      author={Xiaonan Li and Changtai Zhu and Linyang Li and Zhangyue Yin and Tianxiang Sun and Xipeng Qiu},
      year={2023},
      eprint={2311.07838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
