# dose-whisperer

Research prototype that flags prescription dosage errors and patient-condition drug risks. It combines openFDA label evidence, deterministic dose checks, and a LoRA/QLoRA-fine-tuned small LLM, with a Streamlit app. Built on the MIMIC-III Demo. **Not for clinical use.**

> **Disclaimer:** This is a research and educational prototype. Its outputs are experimental safety flags, not medical advice, diagnoses, or a substitute for clinical judgment. Do not use it to make decisions about real patient care.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Data Sources](#data-sources)
5. [Repository Structure](#repository-structure)
6. [Setup](#setup)
7. [Usage](#usage)
8. [Models and Fine-Tuning](#models-and-fine-tuning)
9. [Evaluation](#evaluation)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Data Licensing and Attribution](#data-licensing-and-attribution)
13. [License](#license)

---

## Overview

Dose Whisperer checks a single prescription against drug-label evidence and the patient's conditions, and reports three things:

- **Dosage status**: is the dose, unit, or frequency consistent with the retrieved label information?
- **Condition risk**: does the drug carry a warning or contraindication relevant to the patient's conditions?
- **Combined risk**: an elevated flag when both issues are present together.

Every result comes with a plain-English explanation and the supporting openFDA label evidence.

## Features

- Interactive Streamlit app: enter or edit a prescription (drug, dose, unit, frequency) and patient conditions, then re-run the analysis.
- Deterministic dose checker for dose, unit, frequency, and daily-dose calculations.
- Condition-risk checker that matches patient conditions to retrieved warnings and contraindications.
- Retrieval of dosage, warnings, contraindications, and precautions from the openFDA Drug Label API (no vector database).
- Synthetic prescription-error generator built from real MIMIC-III Demo prescriptions.
- Comparison of lightweight LLMs, plus LoRA and QLoRA fine-tuning of the selected model.
- Ablation study, classification metrics, BLEU/ROUGE explanation scoring, and error analysis.
- Automatically generated evaluation report.

## How It Works

```
Prescription + patient conditions
            |
            v
   openFDA label retrieval  --->  stored evidence
            |
            +--> Deterministic checks (dose, unit, frequency, daily dose, label range)
            |
            +--> LLM (grounded in retrieved text only)
                   - interprets warnings
                   - matches conditions to warnings
                   - writes a short explanation
            |
            v
 Dosage status | Condition risk | Combined risk | Explanation | Evidence
```

**Design rule:** numbers come from code, not from the model. The LLM is not allowed to invent dosage ranges or contraindications; it only interprets the evidence retrieved from openFDA.

The app output clearly separates three kinds of information:

1. **Retrieved evidence** (openFDA label text)
2. **Deterministic calculations** (Python logic)
3. **LLM-generated explanation**

Safety flags use the statuses `SAFE`, `WARNING`, and `HIGH`.

## Data Sources

| Source | Used for | Access |
|---|---|---|
| MIMIC-III Clinical Database Demo (100 patients) | Real prescription examples | Open access on PhysioNet |
| openFDA Drug Label API | Dosage, warnings, contraindications, precautions | Public API, free API key recommended |
| Curated drug and condition list (10-15 drugs) | Scope of the prototype | Included in this repo |

MIMIC prescriptions are **not** treated as clinically safe ground truth. They are only a realistic starting point for generating labeled errors.

### Synthetic error types

Errors are generated from real prescriptions. Each example keeps the original prescription, the modified prescription, the error type, and the expected result.

- **Dose scaling:** x2, x5, x10, /2, /10
- **Unit changes:** mg / mcg / g
- **Frequency changes:** once daily, twice daily, etc.
- **Other simple transcription-style errors**

The train/validation/test split is done so that an original prescription and its perturbed versions never land in different splits (no leakage).

## Repository Structure


```
dose-whisperer/
├── notebooks/        # Pipeline notebooks (data, errors, retrieval, checks, training, evaluation)
├── app/              # Streamlit application (app.py)
├── data/             # Not committed; see Setup for how to download
├── reports/          # Generated outputs (report, metrics, charts)
├── configs/          # Selected model configuration
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Pipeline stages, roughly in order:

1. MIMIC-III Demo prescription processing
2. Synthetic error generation
3. openFDA retrieval and evidence storage
4. Dose checker and condition-risk checker
5. RAG pipeline (retrieved evidence passed to the model)
6. Model comparison
7. LoRA / QLoRA fine-tuning
8. Evaluation, ablation, and error analysis
9. Automated report generation
10. Streamlit app

## Setup

**Requirements:** Python 3.10+, a GPU for fine-tuning (a free Colab or Kaggle GPU is enough for a ~3B model with QLoRA), and internet access for openFDA.

1. Clone the repository and create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Download the MIMIC-III Demo from PhysioNet and place the CSV files in `data/raw/`:
   https://physionet.org/content/mimiciii-demo/
4. (Recommended) Register a free openFDA API key at https://open.fda.gov/apis/authentication/ and set it as an environment variable named `OPENFDA_API_KEY`. Never commit the key.

The raw data, cached openFDA responses, and model checkpoints are intentionally git-ignored.

## Usage

**Run the notebooks** in pipeline order (see Repository Structure), or open them in Google Colab for GPU access.

**Run the app:**

```
streamlit run app/app.py
```

In the app you can enter a drug, dose, unit, frequency, and patient conditions. The analysis runs immediately, and you can edit any field and run it again.

## Models and Fine-Tuning

Candidate lightweight models:

- Qwen2.5-3B-Instruct
- Phi-3-mini
- Llama-3.2-3B

The final model is chosen on overall results (precision, recall, F1, BLEU, ROUGE-L, JSON validity, latency, memory), not on a single metric. Three variants are compared: **base**, **LoRA**, and **QLoRA**.

Training settings recorded for each run:

| Setting | Value |
|---|---|
| Model | TBD |
| Learning rate | TBD |
| Epochs | TBD |
| Batch size | TBD |
| LoRA rank | TBD |
| Quantization | TBD |
| Training time | TBD |
| GPU / VRAM | TBD |

Model weights and adapters are not stored in this repo.

## Evaluation

**Classification metrics:** accuracy, precision, recall, F1, sensitivity, specificity, false-positive rate, false-negative rate, with confusion matrices. Reported separately for dose errors, unit errors, frequency errors, condition risks, and combined risks.

**Explanation quality:** BLEU, ROUGE-1, ROUGE-2, ROUGE-L, and optionally METEOR and BERTScore. Because text-overlap scores do not prove correctness, outputs are also checked for JSON validity, missing fields, unsupported claims, hallucinated medical information, and evidence grounding.

**Ablation study** (same test set for every system):

1. Rules only
2. Base LLM
3. LLM + RAG
4. Fine-tuned LLM + RAG
5. Fine-tuned LLM + RAG + deterministic dose checking

**Error analysis** covers false positives and negatives, dose-parsing errors, unit and frequency errors, condition-matching errors, retrieval failures, LLM reasoning failures, and hallucinations.

## Results

Results will be added here after evaluation. The generated report (`evaluation_report.md`), `results.json`, `metrics.csv`, confusion matrices, and the model comparison chart are saved in `reports/`.

| System | Precision | Recall | F1 |
|---|---|---|---|
| Rules only | TBD | TBD | TBD |
| Base LLM | TBD | TBD | TBD |
| LLM + RAG | TBD | TBD | TBD |
| Fine-tuned LLM + RAG | TBD | TBD | TBD |
| Fine-tuned LLM + RAG + dose checker | TBD | TBD | TBD |

## Limitations

- Research prototype only; not validated for clinical use.
- Covers a small curated list of drugs and conditions.
- Based on 100 MIMIC-III Demo patients, so the sample is small and from a single hospital's ICU.
- Errors are synthetic and may not reflect real-world mistakes.
- openFDA label text varies in structure, so retrieval and dose parsing can fail for some drugs.
- Does not consider drug-drug interactions, allergies, renal/hepatic function, weight-based or age-specific dosing beyond what is explicitly handled.
- Small LLMs can misread warnings or produce unsupported statements, which is why numeric checks are deterministic.

## Data Licensing and Attribution

- **MIMIC-III Clinical Database Demo** is distributed on PhysioNet under the Open Data Commons Open Database License v1.0. Please cite the dataset and the original MIMIC-III paper (Johnson et al.). Raw data is not included in this repository.
- **openFDA** data is provided by the U.S. FDA. See https://open.fda.gov/terms/.
- This project does not use the credentialed full MIMIC-III database. If you adapt it to do so, you must not redistribute that data.

## License

Released under the MIT License. See [LICENSE](LICENSE).
