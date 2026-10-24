# Dose Whisperer

**A lightweight, evidence-grounded clinical-safety assistant for prescription dosage and condition-risk checking.**

> **Research prototype. Not a medical device.** Outputs are prototype safety flags, not diagnoses or treatment advice, and must not be used for prescribing, dispensing or any patient-care decision. See [Limitations and Ethics](#10-limitations-and-ethics).

**Team 07**: Erick Xavier (KH.EN.P2MCA25126), Sathyaprakash P (KH.EN.P2MCA25155), Noyal John (KH.EN.P2MCA25143), Sourabh S (KH.EN.P2MCA25157), Sabhyam Mishra (KH.EN.P2MCA25151), Sreelakshmi V (KH.EN.P2MCA25158)

**Trained adapter (Hugging Face):** [100rabhs/dose-whisperer-phi3-qlora](https://huggingface.co/100rabhs/dose-whisperer-phi3-qlora)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Research Questions](#2-research-questions)
3. [System Overview](#3-system-overview)
4. [Data](#4-data)
5. [Methods](#5-methods)
6. [Results](#6-results)
7. [Error Analysis](#7-error-analysis)
8. [Repository Structure](#8-repository-structure)
9. [Reproducing the Work](#9-reproducing-the-work)
10. [Limitations and Ethics](#10-limitations-and-ethics)
11. [Future Work](#11-future-work)
12. [Licenses and Acknowledgements](#12-licenses-and-acknowledgements)
13. [Citation](#13-citation)

---

## 1. Abstract

Dose Whisperer reviews a single prescription and flags two kinds of problems: (i) **dosage errors** (incorrect dose, unit or frequency) and (ii) **condition-related drug risks** (a label warning or contraindication that matches a patient condition). When both occur, a higher **combined risk** flag is raised. Every flag is accompanied by a plain-English explanation and the supporting FDA drug-label evidence.

The design separates concerns deliberately: **numbers are checked by deterministic Python code, label facts come from retrieved openFDA text, and a small language model only interprets warnings, matches conditions and writes the explanation.** The model is never allowed to invent dosage ranges or contraindications.

We evaluate on synthetic prescription errors injected into real MIMIC-III Demo prescriptions, using a patient-level train/validation/test split. On a held-out test set of 385 examples, QLoRA fine-tuning with retrieval raises F1 from 0.372 (Phi-3-mini base + RAG) to 0.992, against 0.608 for a rules-only baseline. These scores come from rule-generated errors on a small cohort and should be read cautiously (see Section 10).

## 2. Research Questions

1. Can real prescription records, perturbed with synthetic errors, measure error detection fairly?
2. Does retrieval of label text (RAG) improve a small language model compared with using the model alone?
3. Does LoRA or QLoRA fine-tuning improve a 3B-class model beyond prompting?
4. Does adding deterministic Python dose checking improve reliability over the LLM alone?

## 3. System Overview

Five layers: data, retrieval, deterministic checking, model reasoning, presentation.

```
MIMIC-III Demo ──► Processing & cleaning ──► Synthetic error generator ─┐
                                                                         ▼
openFDA label API ──► Retrieval + local cache ──► Evidence store ───────┤
                                                                         ▼
                              ┌──────────────────────┐   ┌──────────────────────────┐
                              │ Deterministic checker │   │ LLM (RAG prompt)         │
                              │ dose / unit / freq    │   │ warning interpretation,  │
                              │ daily dose vs label   │   │ condition matching,      │
                              └──────────┬───────────┘   │ explanation (JSON)       │
                                         │               └────────────┬─────────────┘
                                         └────────────┬───────────────┘
                                                      ▼
                                              Risk fusion (rules)
                                                      ▼
                                   Streamlit app / evaluation / report
```

The same pipeline serves bulk evaluation, ablation and the live app, so what is evaluated is what the user sees.

**Design principles**

- Numbers are checked by code; language is handled by the model.
- Evidence first: every claim traces to a retrieved label passage or a displayed calculation.
- Original MIMIC prescriptions are **not** treated as clinically safe ground truth.
- No vector database; section-based retrieval is sufficient and easier to audit for a small drug list.
- The UI distinguishes three output types: *retrieved evidence*, *deterministic calculation*, *LLM-generated explanation*.

## 4. Data

### 4.1 MIMIC-III Clinical Database Demo

The openly available 100-patient subset of MIMIC-III. The `PRESCRIPTIONS` table (orders, not necessarily administrations) is the main source; `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES` and `ADMISSIONS` supply conditions and stay linkage.

Processing results (Notebook 2):

| Step | Result |
|---|---|
| Adult filter (age at admission) | 128 adult admissions kept, 1 pediatric excluded; 9 admissions with date-shifted age 300 flagged, not dropped |
| Prescriptions | 10,398 → 10,256 adult rows (93 patients, 121 admissions) |
| Dose parsing | 9,686 single values, 569 ranges, 1 unparsed |
| Canonical table | 10,256 rows × 22 columns |

MIMIC-III has no frequency field, so frequency is assigned from per-drug rules (a default rule is used for 3 of the 8 drugs).

### 4.2 openFDA Drug Label API

Structured Product Labeling retrieved by generic name: `dosage_and_administration`, `contraindications`, `warnings` / `warnings_and_cautions`, `precautions`, `boxed_warning`, and the `openfda` block. Responses are cached locally with timestamps for repeatability.

Notebook 1 results: 12 cached label files covering 506 label documents; 34 dose-limit entries extracted with label quotes.

### 4.3 Curated drugs and conditions

The planned 10–15 drugs were narrowed to **8 drug/route combinations** that have both a verified label dose limit and label evidence, so that no dose range comes from model memory. Conditions: MIMIC ICD-9 diagnoses mapped to 14 conditions; each original prescription also receives 2 synthetic contexts (1 relevant, 1 irrelevant condition) so condition-risk cases are represented.

### 4.4 Synthetic error generation

Ten error types applied to each original prescription (plus the unmodified original):

| Family | Types |
|---|---|
| Dose scaling | ×2, ×5, ×10, ÷2, ÷10 |
| Unit change | mg / mcg / g swap |
| Frequency change | per-drug frequency rules |
| Missing unit | unit removed |
| Transcription-style | digit transposition, dropped digit |

Expected labels are **computed by deterministic rules against the label range**, not assumed. A ×2 change is not automatically an error, and an original is not automatically safe.

**Selection funnel:** 10,256 prescriptions → 9,468 single numeric doses → 1,102 for drugs with verified evidence → 527 after de-duplication → **2,635 labelled examples** (4 modified versions + 1 original each).

**Split (70/15/15 by patient), with asserts that no patient, admission or original prescription appears in more than one split:**

| Split | Examples | Originals | Patients |
|---|---|---|---|
| Train | 1,870 | 374 | 52 |
| Validation | 380 | 76 | 16 |
| Test | 385 | 77 | 18 |
| Total | 2,635 | 527 | 86 |

Combined-risk labels: 886 HIGH, 1,749 SAFE, **0 WARNING** (the combined WARNING class is empty).

## 5. Methods

### 5.1 Retrieval

Normalize drug name → query openFDA by generic name → select the best human prescription label with a non-empty dosage section → extract sections → split into tagged passages → cache. In the RAG prompt each drug contributes label snippets of 350 characters (up to 8), with sections capped at 1,800 characters. Dose limits are extracted by pattern rules and **manually verified**; if extraction fails the system reports a retrieval failure instead of guessing.

### 5.2 Deterministic safety checker

Parse and validate → normalize units to the label unit → compute daily dose → compare single dose, daily dose and frequency with retrieved label values → assign SAFE / WARNING / HIGH with a machine-readable reason code (e.g. `DAILY_DOSE_EXCEEDS_MAX`, `UNIT_MISMATCH`, `FREQUENCY_ABOVE_LABELED`, `RANGE_NOT_FOUND`). Margins are configuration values tuned on the validation set only.

*Worked example:* "2 mcg" of IV magnesium sulfate converts to 2e-06 g, far below the label range of 0.003–40 g, and is flagged as a unit error.

### 5.3 Condition-risk checking and risk fusion

Two layers: a curated drug-to-condition map with synonyms produces candidate warnings; the LLM then reads candidate passages and patient conditions and answers in a fixed JSON schema (matched condition, passage, severity, short reason). Condition risk is HIGH on a contraindication or boxed-warning match, WARNING on a warning/precaution match, SAFE otherwise.

Combined risk is a **deterministic fusion table**, not LLM output, with an **escalate-only merge** (dosage status and condition risk can raise but never lower the combined flag), selected on validation and reported on test. A separate combined flag is set whenever both a dosage issue and a condition issue exist.

### 5.4 Hallucination guardrails

The prompt contains only retrieved text and deterministic results; the model must answer "no evidence found" when nothing applies; output is schema-validated JSON; dosage numbers not matching computed values or retrieved text are flagged as unsupported; invalid JSON is retried once, then falls back to a rule-based message.

### 5.5 Models and fine-tuning

| Model | Size | Variants tested |
|---|---|---|
| Phi-3-mini-4k-instruct | ~3.8B | base, base + RAG, base bf16 + RAG (reference), LoRA, QLoRA |
| MedGemma-4B-it | ~4B | base, base + RAG, base bf16 + RAG (reference), LoRA, QLoRA |

The original plan listed Qwen2.5-3B-Instruct, Phi-3-mini and Llama-3.2-3B; the executed experiments compared Phi-3-mini and MedGemma-4B-it plus a rules-only baseline, all on the same test set (n = 385).

- **LoRA:** frozen 16-bit base, low-rank adapters on attention (and optionally MLP) layers.
- **QLoRA:** 4-bit NF4 base with the same style of adapters, lowering GPU memory.
- Training teaches the model to emit the fixed JSON output given evidence, deterministic results and conditions. Early stopping and checkpoint selection use the validation set only.

**Training configuration** (copy from `results.json` / `training_info.json` before release):

| Item | Value |
|---|---|
| Base model | google/medgemma-4b-it && microsoft/Phi-3-mini-4k-instruct|
| Learning rate | 1e-4 (cosine schedule, 5% warmup) |
| Epochs | 2 (the code uses 1 in quick mode) |
| Batch size (grad. accumulation) | 8 per device × 2 accumulation steps = 16 effective (A100 profile) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Quantization | 4-bit NF4 (QLoRA) |
| Training time |  |
| GPU / peak VRAM | |
| Seed | 42 |

Measured QLoRA inference memory per example: 3.38 GB (Phi-3-mini), 3.75 GB (MedGemma-4B).

## 6. Results

All results are on the held-out test set (**n = 385, 18 patients**). Latency and memory are per single example.

### 6.1 Ablation (F1)

| System | Phi-3-mini | MedGemma-4B |
|---|---|---|
| S1 Rules only | 0.608 | 0.608 |
| S2 Base LLM | 0.338 | 0.312 |
| S3 LLM + RAG | 0.372 | 0.290 |
| S4 QLoRA + RAG | 0.992 | 0.996 |
| S5 QLoRA + RAG + deterministic dose check | 0.986 | 0.989 |

Findings:

- Rules only (F1 0.608) outperforms both untuned LLMs, with or without retrieval.
- RAG alone does not fix accuracy but removes most ungrounded output.
- Fine-tuning gives the large gain, to about 0.99.
- Adding the deterministic dose check slightly lowered F1: precision fell from 0.996 to 0.983 (Phi-3-mini) and from 0.996 to 0.982 (MedGemma-4B), while recall stayed about the same (0.988 → 0.989 and 0.996 → 0.997).

### 6.2 Model comparison (selected rows)

| Model | Variant | Prec. | Recall | F1 | BLEU | ROUGE-L | JSON | Lat. (s) | Mem (GB) |
|---|---|---|---|---|---|---|---|---|---|
| None | Rules only | 0.584 | 0.748 | 0.608 | n/a | n/a | 1.000 | 0.03 | 0.00 |
| Phi-3-mini | Base (no retrieval) | 0.276 | 0.544 | 0.338 | 0.007 | 0.159 | 0.977 | 5.26 | 2.70 |
| Phi-3-mini | Base + RAG | 0.277 | 0.737 | 0.372 | 0.017 | 0.177 | 0.964 | 8.04 | 3.14 |
| Phi-3-mini | **QLoRA + RAG** | 0.996 | 0.988 | 0.992 | 0.959 | 0.974 | 1.000 | 6.78 | 3.38 |
| Phi-3-mini | QLoRA + RAG + dose check | 0.983 | 0.989 | 0.986 | 0.959 | 0.974 | 1.000 | 6.78 | 3.38 |
| MedGemma-4B | Base (no retrieval) | 0.336 | 0.329 | 0.312 | 0.016 | 0.175 | 1.000 | 6.71 | 3.43 |
| MedGemma-4B | Base + RAG | 0.320 | 0.402 | 0.290 | 0.033 | 0.187 | 0.992 | 7.65 | 3.57 |
| MedGemma-4B | **QLoRA + RAG** | 0.996 | 0.996 | 0.996 | 0.968 | 0.981 | 1.000 | 10.60 | 3.75 |
| MedGemma-4B | QLoRA + RAG + dose check | 0.982 | 0.997 | 0.989 | 0.968 | 0.981 | 1.000 | 10.60 | 3.75 |

Phi-3-mini LoRA + RAG reached precision 0.988, recall 0.977, F1 0.983 and JSON validity 1.000, so QLoRA (F1 0.992) was slightly better than LoRA without an accuracy cost. The full table (including bf16 reference rows) is in `metrics.csv`.

### 6.3 Model selection

Phi-3-mini and MedGemma-4B are near-tied on accuracy (F1 0.992 vs 0.996; 0.986 vs 0.989 with the dose check), a ~0.004 gap on 385 examples. Phi-3-mini is about 36% faster (6.78 s vs 10.60 s) and uses less memory (3.38 GB vs 3.75 GB); hallucination rates are similar (0.117 vs 0.125). The final model is chosen on overall results rather than a single metric, following the validation-based configuration saved in the model config file.

**Final model:** `<confirm from saved model config>`

### 6.4 Explanation quality

- Text similarity for QLoRA + RAG: BLEU 0.959 / ROUGE-L 0.974 (Phi-3-mini); BLEU 0.968 / ROUGE-L 0.981 (MedGemma-4B). Base models score BLEU 0.01–0.04. These measure wording similarity, not factual correctness.
- JSON validity rises from 0.964 (Phi-3-mini base + RAG) to 1.000 after fine-tuning.
- Hallucination rate without retrieval: 0.691 (Phi-3-mini), 0.912 (MedGemma-4B). With RAG: 0.108 and 0.013. After QLoRA: 0.117 and 0.125, mostly invented ranges.
- Only 0.426 of explanations cite evidence.
- METEOR depended on NLTK data that failed to download; confirm before reporting.

## 7. Error Analysis

- **Hallucinations:** 45 of 385 Phi-3-mini QLoRA + RAG explanations were flagged, all as invented ranges and none as invented contraindications. By error type: unit_change 15, dose_div10 10, dose_x5 6, dose_x2 5, dose_x10 3, missing_unit 3, frequency_change 2, none 1. Many flags are unit conversions of the label range (e.g. 40 g shown as 40,000 mg), so **0.117 is likely an upper bound** owing to checker noise.
- **False positives:** Base + RAG over-flags (precision 0.277). The deterministic dose check also cost some precision (0.996 → 0.983).
- **Condition matching:** loose matches observed, such as heart failure linked to a heart-rate or heart-block contraindication, and pregnancy linked to a toxemia warning.
- **Unit errors:** clearest correct catch is the IV magnesium sulfate "2 mcg" case; unit changes are also the largest source of hallucination flags.
- **Grounding:** only 0.426 of explanations cite evidence.

The full categorized analysis (false positives/negatives, parsing, unit, frequency, condition matching, retrieval, reasoning, hallucination) with representative examples is in `evaluation_report.md`.

## 8. Repository Structure

> Adjust to match the final repository layout.

```
.
├── app/                    # Streamlit application
├── notebooks/
│   ├── 1_openfda_evidence.ipynb       # label cache, dose_table.json
│   ├── 2_mimic_processing.ipynb       # canonical_medications.csv
│   ├── 2b_errors_labels_splits.ipynb  # train/val/test.csv, evidence.csv, split_report.json
│   └── 3_models_rag_evaluation.ipynb  # results.json, metrics.csv, charts, report, model config
├── outputs/
│   ├── evaluation_report.md
│   ├── results.json
│   ├── metrics.csv
│   ├── confusion matrices, model comparison chart, error analysis
│   └── model config (selected model)
├── adapters/               # or hosted on Hugging Face (see below)
├── presentation/
└── README.md
```

Each notebook writes files that the next one reads, so every stage can be re-run independently.

**Adapter weights** are hosted on Hugging Face Hub: [100rabhs/dose-whisperer-phi3-qlora](https://huggingface.co/100rabhs/dose-whisperer-phi3-qlora). Only adapter files are distributed (`adapter_model.safetensors`, `adapter_config.json`, `training_info.json`); the base model is not redistributed. MedGemma adapter: `<add link if uploaded>`.

## 9. Reproducing the Work

1. **Environment.** Python environment with the libraries used in the notebooks (transformers, peft, bitsandbytes, datasets, streamlit, scikit-learn, evaluation metrics packages). A CUDA GPU is needed for fine-tuning; QLoRA keeps memory use to a single consumer or free-tier GPU. `<add requirements.txt>`
2. **Data access.** MIMIC-III Demo is openly available from PhysioNet; comply with its terms of use. **Do not commit raw MIMIC tables or the generated train/val/test files.**
3. **Run the notebooks in order:** 1 → 2 → 2b → 3. Use a free openFDA API key to avoid rate limits; responses are cached after first retrieval.
4. **Base models.** Phi-3-mini-4k-instruct and MedGemma-4B-it from Hugging Face. MedGemma is a gated model: accept Google's terms on its model page before downloading.
5. **Load the adapter.** Load the base model in 4-bit (NF4), then attach the adapter from the Hugging Face repo above using PEFT.
6. **Launch the app.** Start the Streamlit app from `app/` (`<add entry-point filename>`). Enter drug, dose, unit, frequency and conditions; edit and re-run as needed.
7. **Regenerate the report.** The report script rebuilds `evaluation_report.md`, `results.json`, `metrics.csv`, confusion matrices and the comparison chart from the saved predictions.

Splits are saved with a fixed random seed; validation is used for model, prompt and threshold selection and the test set is used once for final reporting.

### Application behaviour

Users enter prescription fields and patient conditions; analysis runs immediately and all inputs remain editable. Panels show: Prescription, Dosage Check (status, calculated daily dose, retrieved range, reason), Condition Check (condition, status, relevant label text, evidence), Combined Risk, and Explanation. Each panel is tagged as *Retrieved evidence*, *Deterministic calculation* or *LLM-generated explanation*. Status is shown in words (SAFE / WARNING / HIGH), not by color alone. Unknown drugs, unrecognized units, missing labels and API failures show clear messages and fall back to cached data; the app never fills in missing numbers itself.

## 10. Limitations and Ethics

**Intended use:** research and education only. Not for real prescribing or dispensing, patient care, or replacing pharmacist and physician judgment.

| Limitation | Mitigation / next step |
|---|---|
| F1 above 0.98 on rule-generated errors may reflect learned generator patterns, not real-world skill | Add real, clinician-reviewed error cases and external test sets |
| Only 8 drug/route combinations and 18 test patients; 3 of 8 drugs use a default frequency rule | Verify limits and frequency rules for more drugs; repeat with several seeds |
| Labels are rule-generated; combined WARNING class is empty (886 HIGH, 1,749 SAFE) | Add graded cases so WARNING is represented |
| Hallucination about 0.12 after fine-tuning; checker noise | Normalize g/mg/mcg before comparison; manually audit a sample |
| Only 0.426 of explanations cite evidence | Require evidence citation in output; score only cases with a condition risk |
| Loose condition matching | Add a condition-warning ontology and stricter match threshold |
| METEOR relied on NLTK data that failed to download | Confirm METEOR ran correctly before reporting |
| No drug-drug interaction checking, no renal/hepatic adjustment | Out of scope for this prototype |
| No clinical validation | Pharmacist review, prospective evaluation, regulatory pathway |

MIMIC prescriptions are order records, not verified ground truth. MIMIC data are de-identified and used under their terms of use. Outputs are prototype safety flags, not diagnoses.

## 11. Future Work

Real, clinician-reviewed error sets; more drugs with verified label limits and frequency rules; graded combined-risk labels; a condition-warning ontology; explanation citation requirements; drug-drug interactions; multi-seed runs and bootstrap confidence intervals; and pharmacist review before any clinical consideration.

## 12. Licenses and Acknowledgements

- **Code:** 
- **Phi-3-mini-4k-instruct:** MIT license (Microsoft).
- **MedGemma-4B-it:** released under Google's health-model terms; review them before redistributing derived weights. Gated access.
- **Adapters:** derived from the base models above; consult each base model's license. The adapter repository is currently marked MIT on Hugging Face; confirm this is compatible with the base model's terms.
- **Data:** MIMIC-III Clinical Database Demo (PhysioNet) under its data use terms; drug labeling from the U.S. FDA via openFDA.

## 13. Citation

```
@misc{dosewhisperer2026,
  title  = {Dose Whisperer: A Lightweight Clinical-Safety Assistant for Prescription Dosage and Condition-Risk Checking},
  author = {Xavier, Erick and P, Sathyaprakash and John, Noyal and S, Sourabh and Mishra, Sabhyam and V, Sreelakshmi},
  year   = {2026},
  note   = {Team 07 research prototype. Adapter: https://huggingface.co/100rabhs/dose-whisperer-phi3-qlora}
}
```
*Dose Whisperer is a research prototype, not a medical device or a substitute for clinical judgment.*
