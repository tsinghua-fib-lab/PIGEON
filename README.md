# 🐦 PIGEON
PIGEON: An Open-Set Living Need Prediction System with Large Language Models

# 📰 News
- 2025.01 Our paper, "Open-Set Living Need Prediction with Large Language Models," has been accepted by ACL 2025 Findings!

# 🌍 Introduction
Living needs are the needs people generate in their daily lives for survival and well-being. On life service platforms like Meituan, user purchases are driven by living needs, making accurate living need predictions crucial for personalized service recommendations. Traditional approaches treat this prediction as a closed-set classification problem, severely limiting their ability to capture the diversity and complexity of living needs. 

In this work, we redefine living need prediction as an open-set classification problem and propose **PIGEON**, a novel system leveraging large language models (LLMs) for unrestricted need prediction. PIGEON first employs a behavior-aware record retriever to help LLMs understand user preferences, then incorporates Maslow’s hierarchy of needs to align predictions with human living needs. For evaluation and application, we design a recall module based on a fine-tuned text embedding model that links flexible need descriptions to appropriate life services.

![](./framework.png)

# ⌨️ Repo Structure
```
- PIGEON.py                     # Main entry point for prediction and evaluation.
- Encoder.py                    # GNN-based encoder for learning user and spatiotemporal context embeddings.
- fine_tuning.py                # Fine-tuning script for the text embedding model used in the recall module.
- Query.py                      # Script to generate refined need descriptions using LLMs.
- data_sample.csv               # A sample of the data used.
- README.md                     # This file.
```

# 💡 Running Experiments

## Installation
It is recommended to use Conda to manage the environment.
```bash
conda create -n pigeon python=3.10
conda activate pigeon
pip install -r requirements.txt
```

## API Key
Configure your OpenAI API key in `PIGEON.py` and `Query.py`:
```python
API_KEY = 'your-openai-api-key'
```

## Running
The project execution is divided into three main steps:

### 1. Train the GNN Encoder
This step trains the LightGCN model to generate embeddings for users and spatiotemporal contexts. These embeddings are crucial for retrieving relevant historical records.
```bash
python Encoder.py
```
This script will train the model and save the embeddings and mappings in the `./model_output/` directory.

### 2. Predict Living Needs with LLM
This step uses the trained GNN encoder to retrieve relevant records and then leverages an LLM to predict living needs in an open-set manner.
```bash
python PIGEON.py
```
The script will perform the following actions:
1.  Load the pre-trained embeddings.
2.  For each entry in the test set, retrieve relevant personal and similar users' historical records.
3.  Use a large language model (GPT-4o-mini by default) to predict the living need based on the retrieved records.
4.  Refine the prediction using Maslow's hierarchy of needs.
5.  Save the results to a CSV file (e.g., `llm_results_YYYYMMDD_HHMMSS.csv`).
6.  Use a fine-tuned sentence transformer model to recall relevant services based on the predicted need.
7.  Evaluate the recall performance using NDCG and Recall@k metrics.

### 3. Fine-tune the Recall Model
To adapt the recall model to flexible need descriptions, you first need to generate refined predictions.

**3.1. Generate Refined Predictions for Fine-tuning Data**
Assuming you have a file `llm_results_finetune.csv` with `order_intention` and `predicted_intention` columns, run `Query.py` to add a `refined_prediction` column.
```bash
python Query.py
```
This will generate `llm_results_finetune_refined.csv`.

**3.2. Fine-tune the Sentence Transformer Model**
This step fine-tunes a text embedding model (e.g., `BAAI/bge-base-zh-v1.5`) to better map the flexible living need descriptions to specific life services.
```bash
python fine_tuning.py
```
The script will:
1.  Load the refined prediction data.
2.  Construct triplet training examples (anchor, positive, negative).
3.  Fine-tune the sentence transformer model.
4.  Save the best model to the `output_model/best_model_triplet_loss_llm_refined` directory, which can then be used in `PIGEON.py` for evaluation.


# 🌟 Citation

If you find this work helpful, please cite our paper:

```latex
@article{lan2025open,
  title={Open-Set Living Need Prediction with Large Language Models},
  author={Lan, Xiaochong and Feng, Jie and Sun, Yizhou and Gao, Chen and Lei, Jiahuan and Shi, Xinlei and Luo, Hengliang and Li, Yong},
  journal={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```

# 📩 Contact

If you have any questions or want to use the code, feel free to contact:
Jie Feng (fengjie@tsinghua.edu.cn)
