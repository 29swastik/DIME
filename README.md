# DIME: Dimension Importance Estimation for Dense Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/dime_implementation.ipynb)

This repository contains a comprehensive implementation of **DIME (Dimension Importance Estimation)** methods for improving dense retrieval systems. The implementation includes three different approaches for determining dimension importance in query embeddings, each with both reranking and refetching capabilities.

## 📖 Medium Article

For a detailed explanation and analysis, read the accompanying Medium article: **[Your Article Title Here]** - *Add your Medium article link*

## 🌟 Overview

Dense retrieval systems often suffer from the "curse of dimensionality" where not all embedding dimensions contribute equally to relevance. DIME addresses this by identifying and zeroing out less important dimensions in query vectors, leading to improved retrieval performance.

### 🔬 Implemented Approaches

1. **Magnitude-based DIME** 🎯
   - Uses absolute values `|qi|` of query dimensions
   - Simplest and fastest approach
   - No external dependencies

2. **PRF-based DIME** 📊
   - Uses Pseudo-Relevance Feedback from initial retrieval
   - Computes centroids from top-k retrieved documents
   - Supports weighted and unweighted averaging

3. **LLM-based DIME** 🤖
   - Uses LLM-generated documents for importance estimation
   - Works without initial retrieval
   - Can be enhanced with actual LLM APIs

### 🔄 Operation Modes

Each approach supports two operation modes:
- **Rerank**: Re-score existing retrieval results with modified query vectors
- **Refetch**: Perform new retrieval with modified query vectors

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook dime_implementation.ipynb
```

Or use the example below:

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data and model
df = pd.read_csv('apparel_dataset.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
product_embeddings = model.encode(
    df['text'].tolist(), 
    normalize_embeddings=True
)

# Initialize DIME classes
from dime_implementation import MagnitudeBasedDIME, PRFBasedDIME, LLMBasedDIME

magnitude_dime = MagnitudeBasedDIME(model, product_embeddings, df)
prf_dime = PRFBasedDIME(model, product_embeddings, df)
llm_dime = LLMBasedDIME(model, product_embeddings, df)

# Test different approaches
query = "black dress shirt"

# Magnitude-based approach
mag_results = magnitude_dime.magnitude_rerank(query, zero_out_ratio=0.2)

# PRF-based approach
prf_results = prf_dime.prf_rerank(query, prf_k=5, zero_out_ratio=0.2)

# LLM-based approach
llm_results = llm_dime.llm_rerank(query, zero_out_ratio=0.2)
```


## 📊 Dataset

The included `apparel_dataset.csv` contains 2000 e-commerce apparel products with:
- **Product ID**: Unique identifier
- **Title**: Product name
- **Description**: Detailed product description  
- **Text**: Combined text for embedding (title + description + brand + category)

## 🔧 Configuration

Key parameters you can adjust:

- `zero_out_ratio`: Fraction of dimensions to zero out (0.0-0.5)
- `prf_k`: Number of documents for PRF centroid computation (3-20)
- `initial_top_k`: Initial retrieval size (100-1000)
- `final_top_k`: Final results to return (5-20)
- `weighted`: Use weighted vs. unweighted centroids
- `attention_type`: "linear" or "softmax" attention
- `temperature`: Softmax temperature (0.5-2.0)


## 🔍 Algorithm Details

### Magnitude-based DIME
1. Compute `importance = |query_embedding|`
2. Sort dimensions by importance
3. Zero out bottom `(1-α)` fraction
4. Use modified query for retrieval/reranking

### PRF-based DIME  
1. Initial retrieval with original query
2. Compute centroid from top-k results
3. Compute `importance = centroid ⊙ query_embedding`
4. Zero out least important dimensions
5. Use modified query for retrieval/reranking

### LLM-based DIME
1. Generate expanded document for query
2. Embed the LLM document  
3. Compute `importance = llm_embedding ⊙ query_embedding`
4. Zero out least important dimensions
5. Use modified query for retrieval/reranking


⭐ **Star this repository if you find it helpful!** ⭐
