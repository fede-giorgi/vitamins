# NEISS Vitamin & Supplement ED Visit Analysis

## Project Overview
Vitamins, herbs, and dietary supplements are widely marketed as harmless and possibly helpful. This project investigates the **National Electronic Injury Surveillance System (NEISS)** database over the past 21 years to determine national temporal trends in emergency department (ED) visits related to poisoning or adverse reactions associated with these substances (e.g., ashwagandha, turmeric, vitamin B12, melatonin). 

### The Data (NEISS)
The National Electronic Injury Surveillance System (NEISS) is a U.S. system managed by the Consumer Product Safety Commission (CPSC). It collects data on consumer product-related injuries treated in hospital emergency departments. 

**National Estimates & Statistical Weighting:**
By utilizing a probability sample of approximately 100 U.S. hospitals, the NEISS allows researchers to extrapolate and create highly accurate national estimates. Each case in the dataset is assigned a **statistical weight**; the sum of these weights represents the total estimated number of ED visits nationwide. This project leverages these weights to ensure all temporal and demographic trends reflect the entire U.S. population.

### Project Goals
* **Automated Classification:** Scalable processing of >90,000 medical narratives.
* **Epidemiological Modeling:** Identifying statistically significant shifts in exposure patterns.
* **Public Health Impact:** Distinguishing between safe vitamin use and high-risk supplement exposures (e.g., iron toxicity or non-vitamin botanicals).

---

## System Architecture

The repository utilizes a **Teacher-Student Machine Learning Pipeline** to achieve high-precision classification at scale, followed by a rigorous statistical evaluation.

```mermaid
graph LR
    %% Define Node Styles
    classDef block fill:#ffffff,stroke:#004479,stroke-width:2px,color:#000000,font-weight:bold;
    classDef model fill:#e6e8ea,stroke:#6e757c,stroke-width:2px,color:#000000,font-weight:bold;
    classDef agent fill:#fff3e6,stroke:#ffa500,stroke-width:2px,stroke-dasharray: 5 5,color:#000000,font-weight:bold;
    classDef user fill:#f4f5f6,stroke:#6e757c,stroke-width:1px;

    %% Nodes
    Data[(Raw NEISS Data<br>90k+ Narratives)]:::block
    Agent{Ollama Agent<br>Llama 3.1 Labeling}:::agent
    
    Preproc[Few-Shot Bootstrapping<br>& Hybrid Sampling]:::block
    
    subgraph Machine Learning Pipeline
        BERT[BERT-base-uncased]:::model
        LoRA[LoRA Fine-Tuning]:::model
        Inf[Mass Batch Inference]:::model
    end
    
    Stats[OLS Multivariate<br>Regression]:::block
    Viz[National Trend<br>Visualizations]:::block

    %% Data Flow
    Data ==> Preproc
    Preproc ==> Agent
    Agent -.->|Ground Truth Generation| BERT
    BERT ==> LoRA
    LoRA ==> Inf
    Inf ==> Stats
    Stats ==> Viz
```

---

## Repository Architecture & Workflow

### 1. Zero-Shot / Few-Shot Data Labeling (The Teacher)
* **Script:** `scripts/1_prepare_data.py`
* **Model:** `Llama 3.1:8b` (via Ollama)
* **Function:** Instead of manual labeling, we use a local LLM as a "Silver Standard" labeler. Through strict Chain-of-Thought (CoT) prompting, the model evaluates a subset of narratives to identify true supplement exposures.
* **Hybrid Sampling:** To handle extreme class imbalance (0.3% prevalence), we implement **oversampling** of positive cases and **undersampling** of hard negatives (e.g., iron, bleach, pharmaceuticals) to provide a balanced training signal for the next stage.

### 2. Fine-Tuning & Mass Inference (The Student)
* **Script:** `scripts/2_train_bert.py`
* **Model:** `bert-base-uncased` + LoRA
* **Function:** A BERT sequence classifier is fine-tuned on the LLM-generated dataset using **Low-Rank Adaptation (LoRA)**. 
* **Scalability:** Once the student model (BERT) reaches convergence, it performs high-speed inference across the entire 21-year database. This approach achieves a **200x speedup** compared to processing the whole dataset via LLM while maintaining deep semantic understanding.

### 3. Statistical & Temporal Analysis
* **Notebook:** `temporal_analysis.ipynb`
* **Modeling:** Ordinary Least Squares (OLS) Linear Regression with **Heteroscedasticity and Autocorrelation Consistent (HAC)** standard errors.
* **Interaction Terms:** We model the interaction between `Year` and `Category` to detect if specific demographics (e.g., pediatric age groups) are diverging from the national baseline.
* **Seasonality:** Analysis of normalized monthly shares to identify cyclic patterns in vitamin-related hospitalizations.

---

## Setup & Usage

1. **Prerequisites:**
   * Install [Ollama](https://ollama.com/) and pull the required model: `ollama pull llama3.1`.
   * Install Python dependencies: `pip install -r requirements.txt`.

2. **Data Preparation:**
   ```bash
   python scripts/1_prepare_data.py --samples 2000
   ```

3. **Model Training:**
   ```bash
   python scripts/2_train_bert.py
   ```

4. **Analysis:**
   Open `temporal_analysis.ipynb` to generate final epidemiological plots and regression summaries.