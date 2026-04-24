# NEISS Vitamin ED Visit Analysis

## Project Overview
Vitamin and dietary supplement exposures are a common reason for toxicology-related emergency department (ED) encounters, particularly among children with unintentional ingestions. While the **National Electronic Injury Surveillance System (NEISS)** enables the assessment of long-term trends, accurate identification of relevant cases can be limited by variable coding and the need to interpret short, free-text case narratives.

This project investigates 20 years of NEISS data (2004–2023) to estimate the annual incidence of ED-treated vitamin exposures. By applying a Large Language Model (LLM) to process the entire dataset of over 91,000 case narratives, this study demonstrates a scalable methodology to improve case ascertainment in large surveillance databases where comprehensive manual review is impossible.

### The Data & National Estimates
The NEISS is a national ED surveillance database managed by the Consumer Product Safety Commission (CPSC). By utilizing a probability sample of approximately 100 U.S. hospitals, the NEISS allows researchers to extrapolate and create highly accurate national estimates. Each case in the dataset is assigned a **statistical weight**; the sum of these weights represents the total estimated number of ED visits nationwide. This project leverages these weights to ensure all temporal trends reflect the entire U.S. population.



## Methodology

### 1. Hybrid Cloud-Edge LLM Classification
Because full manual review is impractical at the NEISS scale, we applied a dual-LLM pipeline to review and classify the coder-written case narratives:
* **Primary Classifier:** `Gemini 2.5 Flash Lite` processed the vast majority of the dataset via API for high throughput.
* **Local Fallback:** `Gemma 4: 9B` (via Ollama) served as a secure, local fallback to process any narratives blocked by commercial API safety filters (e.g., narratives containing graphic descriptions of unrelated injuries).

The LLM prompt was developed using an **iterative design process**: the database was sampled in batches of 200, 500, and 1,500 cases, which were analyzed by the LLM and manually reviewed by a medical toxicologist. The final classification logic was instructed to **include strictly harmless vitamins** and explicitly **exclude non-vitamin dietary supplements, iron-containing products, medications, and household chemicals**.

### 2. Temporal & Statistical Analysis
Temporal trends across the 2004–2023 period were assessed using **Ordinary Least Squares (OLS) linear regression** models. The analysis also incorporates multivariate interaction terms to detect if specific demographics (e.g., pediatric age groups vs. adults) or clinical severities (Admitted vs. Treated/Released) diverged from the national baseline over the 20-year study period.



## Repository Structure

The repository follows a standard Data Science layout to separate raw data, reusable logic, execution scripts, and final analyses.

```text
vitamins/
├── data/
│   ├── raw/                 # The original NEISS Excel dataset
│   ├── processed/           # The final LLM-classified dataset (91k rows)
│   └── archive/             # Iterative design samples (200, 500, 1500 rows)
├── src/                     # Reusable Python modules
│   ├── __init__.py
│   ├── load_data.py         # Data cleaning and preprocessing functions
│   └── classification.py    # LLM business logic (Prompts, JSON parsers, Regex rules)
├── scripts/                 # Entry points for execution
│   └── run_classification.py# Main script to trigger the hybrid LLM pipeline
├── notebooks/               # Final statistical analyses and visualizations
│   └── temporal_analysis.ipynb
├── README.md
├── requirements.txt
└── .env.example
```

---

## Setup & Execution

### Prerequisites
1. **API Keys:** Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```
2. **Ollama (Required for Fallback):** Install [Ollama](https://ollama.com/) locally to run the unfiltered fallback model. Open your terminal and run:
   ```bash
   ollama run gemma4:e4b
   ```
3. **Python Environment:** Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
1. **Classify the Data:**
   Run the main classification script. This will iterate through the raw NEISS data, apply the Gemini/Gemma pipeline, and save the output to `data/processed/`.
   ```bash
   python scripts/run_classification.py
   ```
2. **Analyze the Trends:**
   Open `notebooks/temporal_analysis.ipynb` via Jupyter or VS Code to run the OLS regressions, view the interaction terms, and generate the final epidemiological plots.