# NEISS Vitamin & Supplement ED Visit Analysis

## Project Overview
Vitamins, herbs, and dietary supplements are widely marketed as harmless and possibly helpful. This project investigates the National Electronic Injury Surveillance System (NEISS) database over the past 21 years to determine national temporal trends in emergency department (ED) visits related to poisoning or adverse reactions associated with these substances (e.g., ashwagandha, turmeric, vitamin B12, melatonin). 

### The Data (NEISS)
The National Electronic Injury Surveillance System (NEISS) is a U.S. system managed by the Consumer Product Safety Commission (CPSC). It collects data on consumer product-related injuries treated in hospital emergency departments. By utilizing a probability sample of U.S. hospitals, the NEISS allows researchers to extrapolate and create highly accurate national estimates to inform public safety efforts and develop safety standards.

### Project Goals & Responsibilities
The core responsibilities of this research involve extensive data cleansing, natural language classification of medical narratives, big data system analysis, sample-based extrapolation, and advanced data visualization. The ultimate measure of success is leveraging robust data analysis to accurately represent national trends in ED presentations. If the findings are sufficiently robust, this analysis will be submitted as a medical epidemiology paper.

---

## Repository Architecture & Workflow

Given the massive volume of the 21-year dataset and the complexity of medical narratives, this repository utilizes a modern, two-step Machine Learning pipeline to classify the data, followed by rigorous statistical analysis.

### 1. Zero-Shot / Few-Shot Data Labeling (LLaMA)
* **Script:** `scripts/1_prepare_data.py`
* **Function:** Instead of manually labeling thousands of medical narratives, we use a local Large Language Model (`Llama 3.1:8b` via Ollama). Utilizing a strict Chain-of-Thought (CoT) prompt and JSON structured output, the model evaluates a sample of narratives to identify true supplement exposures while rigorously filtering out exceptions (e.g., iron toxicity, pharmaceuticals, household chemicals). This creates a high-quality "Ground Truth" training dataset.

### 2. Fine-Tuning & Mass Inference (BERT + LoRA)
* **Script:** `scripts/2_train_bert.py`
* **Function:** A `bert-base-uncased` sequence classifier is fine-tuned on the LLaMA-generated dataset using Low-Rank Adaptation (LoRA). Once trained, the BERT model performs high-speed, batched inference across the entire remaining NEISS database (tens of thousands of records). This approach perfectly balances the deep reasoning capabilities of LLMs with the raw speed and efficiency of BERT.

### 3. Statistical & Temporal Analysis
* **Notebook:** `temporal_analysis.ipynb`
* **Function:** The final classified dataset is ingested for epidemiological evaluation. We utilize Ordinary Least Squares (OLS) Multivariate Linear Regression models, complete with interaction terms, to analyze absolute yearly trends, normalized seasonality, and stratified demographic shifts (Age, Sex, Race, Patient Disposition). The outputs include automated national extrapolations, interaction $p$-values, and production-ready data visualizations for the academic paper.