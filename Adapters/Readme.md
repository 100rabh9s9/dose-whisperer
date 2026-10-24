# Adapters in Dose Whisperer

> Research prototype. Outputs are prototype safety flags, not medical advice or diagnoses.

## What are adapters?

A large language model has billions of weights. Retraining all of them needs a lot of GPU memory and produces a huge file for every experiment.

An **adapter** is a small set of extra weights (a few to a few hundred MB) that is trained on top of a frozen base model. The base model is not changed. This technique is called **LoRA** (Low-Rank Adaptation): instead of updating the original weight matrices, it learns two small low-rank matrices per layer, and their product is added to the layer's output.

**QLoRA** is the same idea, but the frozen base model is loaded in 4-bit precision (NF4), which cuts memory use a lot. Dose Whisperer uses QLoRA.

Think of the base model as a general-purpose textbook and the adapter as a thin set of sticky notes that teaches it one specific job.

## How are they used?

1. The base model is loaded in 4-bit.
2. The adapter weights are attached on top of it.
3. The combined model receives the prompt (prescription, patient conditions and retrieved openFDA label snippets) and returns a structured JSON answer with statuses, matched evidence and a short explanation.

Without the adapter, you still have the base model, which you can use as a comparison. The adapter can be swapped, replaced or removed without touching the base model.

## Why are they used in this project?

- **Limited hardware.** QLoRA fine-tuning of a 3-4B model fits on a single GPU. Full fine-tuning would not.
- **Small, shareable files.** The repository can share the adapter without redistributing the base model.
- **Reliable output format.** The adapter teaches the model to return valid JSON with the required fields and to quote the retrieved evidence instead of making up information.
- **Fair comparison.** The same base model is evaluated with and without the adapter (Base vs QLoRA), so any improvement can be credited to fine-tuning.
- **Safety by design.** The adapter does not decide dose ranges. Numerical checks (dose, unit, frequency, daily dose) come from deterministic Python code and the openFDA label evidence. The adapter is used for interpreting warnings, matching patient conditions and writing the explanation.

## What the adapter was trained on

- Training data: the train split only (1,870 examples), built from MIMIC-III Demo prescriptions with synthetic errors.
- Validation: used for early stopping.
- The test split was held out and used only for the final evaluation.
- Training settings: see the table in the main README.

## How to access it

**Files.** An adapter folder usually contains:
- `adapter_config.json` (settings and the base model name)
- `adapter_model.safetensors` (the trained weights)
- `training_info.json` (hyperparameters, training time, VRAM)

**Where to get it.**
- Selected adapter: `[Hugging Face link or release link]`
- Local path after training: `dose_whisperer_nb3_outputs_full/adapters/<model-name>/`

**Requirements.**
1. Access to the base model: `[selected base model name]`. Some base models (for example MedGemma and Llama) are gated, so accept the licence on Hugging Face and create an access token.
2. Set the token as the environment variable `HF_TOKEN`. Never put a token in code or commit it.
3. A GPU with enough memory for a 4-bit model (roughly 6 GB or more; confirm for your chosen model).
4. Install the packages listed in `requirements.txt`.

**Running it.**
- Point `selected_model_config.json` to the base model and adapter path.
- Start the Streamlit app. In the sidebar, switch on the fine-tuned model.
- If the model cannot load (no GPU, no access, invalid output), the app falls back to the rules-only mode and says so.

## Limitations

- Trained on a small dataset: 527 original prescriptions from 86 patients, 8 drug and route combinations.
- Condition contexts are mostly synthetic.
- Only QLoRA was trained; a plain LoRA comparison is planned.
- Check the licence of the base model before reusing or redistributing the adapter.

## Disclaimer

For research and education only. Not for clinical use. Always follow professional clinical judgment.
