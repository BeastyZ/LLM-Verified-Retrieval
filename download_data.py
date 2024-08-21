from datasets import load_dataset
import json

names = ['asqa_questions', 'qampari_questions', 'eli5_questions']
for name in names:
    ds = load_dataset("BeastyZ/Llatrieval", name)
    data = []
    for d in ds['train']:
        data.append(dict(d))
    if name == 'asqa_questions':
        save_path = './data/asqa_gtr_top100.json'
    elif name == 'qampari_questions':
        save_path = './data/qampari_gtr_top100.json'
    else:
        save_path = './data/eli5_bm25_top100.json'
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
