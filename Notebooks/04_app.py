"""Dose Whisperer - Streamlit prototype.  Run:  streamlit run app.py
RESEARCH PROTOTYPE: outputs are safety flags, not diagnoses or medical advice."""
import json, os, re, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
import streamlit as st
import dw_core as C

OUT_DIR = os.getenv("DW_OUT_DIR", "dose_whisperer_nb3_outputs_full")
EVID_CSV = os.getenv("DW_EVIDENCE", "/home/user9/SOURABH_S/data/splits/evidence.csv")
# Merge policy chosen on validation (must match CHOSEN_* in the gap-closing notebook cell 5):
MERGE_THR, MERGE_FIELDS = "HIGH", ("dose_status", "unit_status", "frequency_status")
DISCLAIMER = ("**Research prototype.** Outputs are prototype safety flags (SAFE / WARNING / HIGH), not medical diagnoses, "
              "and no substitute for clinical judgment.")
COLORS = {"SAFE": "#1b7f3b", "WARNING": "#b36b00", "HIGH": "#b3261e"}
st.set_page_config(page_title="Dose Whisperer", page_icon="💊", layout="wide")


# ------------------------------------------------------------------ helpers
def badge(status):
    s = status or "SAFE"
    return f"<span style='background:{COLORS.get(s, '#555')};color:#fff;padding:2px 10px;border-radius:12px;font-weight:600'>{s}</span>"

def tag(kind):
    return {"ev": "🔎 *Retrieved evidence (openFDA label text)*",
            "py": "🧮 *Deterministic calculation (Python)*",
            "llm": "🤖 *LLM-generated (fine-tuned model)*"}[kind]

@st.cache_data(show_spinner=False)
def load_config(out_dir):
    p = Path(out_dir) / "selected_model_config.json"
    return json.loads(p.read_text()) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_local_evidence(path):
    """Curated drug evidence produced in the earlier notebooks (same text the model was trained on)."""
    if not Path(path).exists():
        return {}
    df, out = pd.read_csv(path).fillna(""), {}
    for r in df.to_dict("records"):
        d = str(r.get("drug", "")).strip()
        if d:
            e = out.setdefault(d, dict(dosage=[], warnings=[], contra=[]))
            e["dosage"].append(str(r.get("dosage", ""))); e["warnings"].append(str(r.get("warnings", "")))
            e["contra"].append(str(r.get("contraindications", "")))
    return {d: {k: "\n".join(x for x in v if x.strip()) for k, v in e.items()} for d, e in out.items()}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_openfda(drug):
    """Live openFDA label lookup -> (evidence dict | None, error text)."""
    name, err = re.sub(r"\(.*?\)", "", drug).strip(), ""
    for field in ("openfda.generic_name", "openfda.brand_name", "openfda.substance_name"):
        q = urllib.parse.urlencode({"search": f'{field}:"{name}"', "limit": 3})
        try:
            with urllib.request.urlopen(f"https://api.fda.gov/drug/label.json?{q}", timeout=15) as r:
                data = json.load(r)
        except Exception as e:
            err = str(e); continue
        for lab in data.get("results", []):
            if lab.get("dosage_and_administration"):
                j = lambda *ks: "\n".join(x for k in ks for x in lab.get(k, []))[:30000]
                return dict(dosage=j("dosage_and_administration"),
                            warnings=j("boxed_warning", "warnings", "warnings_and_cautions", "precautions"),
                            contra=j("contraindications")), ""
    return None, err or "no label with dosing text found"

@st.cache_resource(show_spinner="Loading fine-tuned model (first run only)...")
def load_llm(hf_id, adapter_path, family):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU available")
    tok = AutoTokenizer.from_pretrained(hf_id, token=os.getenv("HF_TOKEN"))
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    classes = [AutoModelForCausalLM]
    if family == "gemma3":
        from transformers import AutoModelForImageTextToText
        classes.insert(0, AutoModelForImageTextToText)
    last = None
    for cls in classes:
        for attn in ("sdpa", "eager"):
            try:
                m = cls.from_pretrained(hf_id, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16,
                                        attn_implementation=attn, token=os.getenv("HF_TOKEN"))
                m = PeftModel.from_pretrained(m, adapter_path, is_trainable=False).eval()
                return tok, m
            except Exception as e:
                last = e
    raise RuntimeError(f"could not load model: {last!r}")

def llm_predict(tok, model, rec):
    import torch
    bp = C.build_prompt(rec, True, tok)
    enc = tok(bp["text"], return_tensors="pt", add_special_tokens=False)
    enc.pop("token_type_ids", None)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    stop = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for n in ("<end_of_turn>", "<|end|>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        t = tok.convert_tokens_to_ids(n)
        if isinstance(t, int) and t >= 0 and t != tok.unk_token_id:
            stop.add(t)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=C.MAX_NEW_TOKENS, do_sample=False, num_beams=1, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id, eos_token_id=sorted(stop))
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return dict(raw=raw, val=C.validate_prediction(C.extract_json(raw)), snippets=bp["snippets"])

def merge_escalate(llm_norm, det):
    """LLM keeps its status; Python may only RAISE dose/unit/frequency when its own verdict is >= MERGE_THR."""
    out, changed = dict(llm_norm), []
    for f in MERGE_FIELDS:
        rs, ls = det[f], out.get(f) or "SAFE"
        out[f] = ls
        if C.LEVELS[rs] >= C.LEVELS[MERGE_THR] and C.LEVELS[ls] < C.LEVELS[rs]:
            changed.append((f, ls, rs)); out[f] = rs
    if changed and C.LEVELS.get(out.get("condition_status"), 0) > 0 and C.LEVELS.get(out.get("combined_status"), 0) == 0:
        out["combined_status"] = "WARNING"
    return out, changed

def calc_summary(rec, det):
    dpar, unit, fp, ref = C.parse_dose(rec["dose"]), C.norm_unit(rec["unit"]), C.parse_freq_per_day(rec["freq"]), det["ref"]
    lines = [f"Parsed dose: **{dpar[1]:g} {unit or '(no unit)'}**" if dpar else f"Dose '{rec['dose']}' could not be parsed"]
    if fp:
        lines.append(f"Doses per day: **{fp[1]:g}**" + (" (PRN)" if fp[2] else ""))
        if dpar:
            daily = dpar[1] * fp[1]
            txt = f"Calculated daily dose: **{daily:g} {unit}**"
            if unit in C.UNIT_MG and ref.get("unit") in C.UNIT_MG and unit != ref["unit"]:
                txt += f" (= {daily * C.UNIT_MG[unit] / C.UNIT_MG[ref['unit']]:g} {ref['unit']})"
            lines.append(txt)
    else:
        lines.append(f"Frequency '{rec['freq']}' could not be interpreted")
    if ref.get("lo") is not None:
        lines.append(f"Label dose range: **{ref['lo']:g}-{ref['hi']:g} {ref['unit']}** per dose"
                     + (f" | label max/day: **{ref['max_daily']:g} {ref['unit']}**" if ref.get("max_daily") else "")
                     + (f" | label frequency: {ref['freq'][0]:g}-{ref['freq'][1]:g}/day" if ref.get("freq") else ""))
    else:
        lines.append("No numeric dose range could be extracted from the label text (checker abstains on dose/unit).")
    return lines


# ------------------------------------------------------------------ sidebar / config
cfg = load_config(OUT_DIR)
local_ev = load_local_evidence(EVID_CSV)
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use fine-tuned LLM", value=True, help="Off = deterministic rules only (no GPU needed).")
    if cfg:
        sm = cfg["selected_model"]
        st.caption(f"Selected model: **{sm['hf_id']}** ({cfg['loading']['quantization'] if isinstance(cfg['loading']['quantization'], str) else '4-bit NF4'} + LoRA adapter)")
        st.caption(f"Test macro-F1 {cfg['key_metrics']['macro_f1']:.3f} | JSON validity {cfg['key_metrics']['json_validity']:.3f}")
    else:
        st.warning(f"selected_model_config.json not found in '{OUT_DIR}'. Running rules-only.")
        use_llm = False
    st.caption(f"Curated drugs loaded: {len(local_ev)}")
    st.markdown("---"); st.markdown(DISCLAIMER)

# ------------------------------------------------------------------ input form
st.title("💊 Dose Whisperer")
st.warning(DISCLAIMER)
drugs = sorted(local_ev) + ["Other (live openFDA lookup)"]
find = lambda n: next((d for d in sorted(local_ev) if d.lower() == n.lower()), None)
PRESETS = {"Example: IV metoprolol 5 mg + heart failure": ("metoprolol (INTRAVENOUS)", "5", "mg", "Q6H", "heart failure"),
           "Example: same drug, dose x10": ("metoprolol (INTRAVENOUS)", "50", "mg", "Q6H", "heart failure"),
           "Example: unit error (mcg)": ("magnesium sulfate (INTRAVENOUS)", "2", "mcg", "BID", "pregnancy")}
PRESETS = {k: v for k, v in PRESETS.items() if find(v[0])}

def load_preset():
    p = PRESETS.get(st.session_state.get("preset"))
    if p:
        st.session_state.update(drug_sel=find(p[0]), dose_in=p[1], unit_in=p[2], freq_in=p[3], cond_in=p[4])

st.session_state.setdefault("drug_sel", drugs[0]); st.session_state.setdefault("dose_in", "100")
st.session_state.setdefault("unit_in", "mg"); st.session_state.setdefault("freq_in", "BID")
st.session_state.setdefault("cond_in", "")
if PRESETS:
    st.selectbox("Load an example (then edit freely)", [""] + list(PRESETS), key="preset", on_change=load_preset)

with st.form("rx"):
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.4])
    drug_sel = c1.selectbox("Drug", drugs, key="drug_sel")
    custom = c1.text_input("Drug name for live openFDA lookup", "") if drug_sel.startswith("Other") else ""
    dose = c2.text_input("Dose", key="dose_in")
    unit = c3.selectbox("Unit", ["mg", "mcg", "g", "mL", "units", "(none)"], key="unit_in")
    freq = c4.text_input("Frequency", key="freq_in", help="e.g. BID, TID, Q6H, once daily, twice daily, 3 times/day")
    conds_txt = st.text_area("Patient conditions (comma or semicolon separated)", key="cond_in", height=70,
                             placeholder="e.g. heart failure, chronic kidney disease")
    submitted = st.form_submit_button("Analyze", type="primary")

if not (submitted or st.session_state.get("ran")):
    st.session_state["ran"] = True            # analyse the defaults immediately on first load
drug = (custom or "").strip() if drug_sel.startswith("Other") else drug_sel
if not drug:
    st.info("Enter a drug name for the live openFDA lookup."); st.stop()
conds = [c.strip() for c in re.split(r"[;,\n|]", conds_txt) if c.strip()]

# ------------------------------------------------------------------ evidence retrieval
if drug in local_ev:
    ev, ev_src = local_ev[drug], "Curated openFDA label evidence (same text the model was fine-tuned on)"
else:
    ev, err = fetch_openfda(drug)
    ev_src = "Live openFDA drug-label API"
    if ev is None:
        st.error(f"No label evidence could be retrieved for '{drug}' ({err}). Dose/unit checks and warnings cannot be grounded.")
        st.stop()
    st.info("This drug is outside the curated list the model was fine-tuned on; treat the LLM output with extra caution.")

rec = dict(id="app", drug=drug, dose=dose.strip(), unit="" if unit == "(none)" else unit, freq=freq.strip(),
           conditions=conds, ev=ev, rng=None)

# ------------------------------------------------------------------ analysis
det = C.check_prescription(rec)                       # deterministic dose / unit / frequency
rules_pred, rule_snips = C.rules_predict(rec)         # keyword condition matcher + combined rule
llm, llm_err, changed, snips = None, None, [], rule_snips
if use_llm:
    try:
        tok, model = load_llm(cfg["selected_model"]["hf_id"], cfg["loading"]["adapter_path"], cfg["selected_model"]["family"])
        with st.spinner("Running the fine-tuned model..."):
            llm = llm_predict(tok, model, rec)
        snips = llm["snippets"]
    except Exception as e:
        llm_err = str(e)
if llm_err:
    st.warning(f"LLM unavailable ({llm_err}). Showing the deterministic rules-only analysis instead.")
if llm and llm["val"]["norm"] is not None and llm["val"]["schema_valid"]:
    final, changed = merge_escalate(llm["val"]["norm"], det)
    mode = "Fine-tuned LLM + RAG + deterministic escalation"
else:
    if llm:
        st.warning("The model returned invalid JSON; falling back to the deterministic rules-only analysis.")
    final, mode, llm = dict(rules_pred), "Rules only (no LLM)", None

# ------------------------------------------------------------------ output
st.caption(f"Analysis mode: **{mode}**")
st.subheader("1. Prescription")
st.markdown(f"**{drug}** — {dose} {unit if unit != '(none)' else '(no unit)'}, {freq or '(no frequency)'} | "
            f"Conditions: {', '.join(conds) if conds else 'none entered'}")

st.subheader("2. Dosage check")
cols = st.columns(3)
for col, (lab, f, note) in zip(cols, [("Dose", "dose_status", det["dose_note"]), ("Unit", "unit_status", det["unit_note"]),
                                      ("Frequency / daily dose", "frequency_status", det["freq_note"])]):
    col.markdown(f"**{lab}**<br>{badge(final.get(f))}", unsafe_allow_html=True)
    col.caption(note or ("checker found no issue" if not det["abstain_dose" if f != "frequency_status" else "abstain_freq"]
                         else "checker abstained (no usable label range)"))
st.markdown(tag("py")); [st.markdown("- " + l) for l in calc_summary(rec, det)]
if changed:
    st.info("Python raised: " + "; ".join(f"{f.replace('_status', '')} {a} → {b}" for f, a, b in changed)
            + ". The LLM explanation below may not mention this; the calculation above is authoritative for numbers.")
st.markdown(tag("ev"))
with st.expander("Retrieved dosage evidence"):
    for s in [s for s in snips if s["section"] == "D"] or [dict(id="-", text="(none retrieved)")]:
        st.markdown(f"**[{s['id']}]** {s['text']}")

st.subheader("3. Condition check")
cond_status, mw, eid = final.get("condition_status"), final.get("matched_warning"), final.get("evidence_quote_or_id")
st.markdown(f"Patient condition(s): **{', '.join(conds) or 'none'}** &nbsp; Status: {badge(cond_status)}", unsafe_allow_html=True)
st.markdown(tag("ev"))
if mw:
    st.markdown(f"> {mw}")
    ids = {s["id"] for s in snips}
    grounded = (eid in ids) or any(mw[:60].lower() in s["text"].lower() for s in snips)
    st.caption(f"Evidence reference: {eid or '—'} " + ("✅ found in retrieved label text" if grounded else "⚠️ not found in retrieved evidence — treat with caution"))
else:
    st.caption("No label warning/contraindication was matched to the patient's conditions.")
m = C.match_condition_rules(rec, snips)
st.caption("Keyword-matcher cross-check: " + (f"{m['level']} (snippet {m['id']}, condition '{m['condition']}')" if m else "no match")
           + (" — agrees ✅" if (m["level"] if m else "SAFE") == cond_status else " — differs ⚠️"))
with st.expander("Retrieved warnings / contraindications"):
    for s in [s for s in snips if s["section"] in "WC"] or [dict(id="-", text="(none retrieved)")]:
        st.markdown(f"**[{s['id']}]** {s['text']}")

st.subheader("4. Combined risk")
st.markdown(f"Overall prototype safety flag: {badge(final.get('combined_status'))}", unsafe_allow_html=True)
st.caption("🧮 Rule: raised one level when a prescription problem AND a condition risk are both present.")

st.subheader("5. Explanation")

if llm:
    st.markdown(tag("llm")); st.write(final.get("explanation") or "(none)")
else:
    st.markdown(tag("py")); st.write(final.get("explanation"))
st.caption("Prototype safety flag only — not a diagnosis or medical advice.")

result = dict(prescription=dict(drug=drug, dose=dose, unit=unit, frequency=freq, conditions=conds), mode=mode,
              dosage_status=dict((k.replace("_status", ""), final.get(k)) for k in MERGE_FIELDS),
              condition_risk=cond_status, combined_risk=final.get("combined_status"), explanation=final.get("explanation"),
              supporting_evidence=dict(source=ev_src, matched_warning=mw, evidence_ref=eid))
with st.expander("Raw result / diagnostics"):
    st.json(result)
    if llm:
        st.text_area("Raw model output", llm["raw"], height=160)
    st.caption(f"Evidence source: {ev_src}")
    st.download_button("Download result JSON", json.dumps(result, indent=1), "dose_whisperer_result.json")