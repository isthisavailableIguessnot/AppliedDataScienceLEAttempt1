"""LeafNPK - rice leaf nitrogen / phosphorus / potassium deficiency checker.

Applied Data Science project, College of Information and Computing,
University of Southeastern Philippines (USeP).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import npk_core as core

st.set_page_config(
    page_title="LeafNPK - Rice NPK Deficiency Checker",
    page_icon="🌾",
    layout="wide",
)

BLUE = "#2a78d6"     # highlight (one series)
GRAY = "#b9b8b2"     # everything else
SAMPLES_DIR = core.APP_DIR / "samples"

# Results from the ML comparative study (mean of 3 seeds; CPU = 2-core Xeon, Colab).
MODEL_RESULTS = pd.DataFrame(
    [
        ("ResNet50",          0.964, 95.0, 356),
        ("EfficientNet-B0",   0.952, 17.1, 342),
        ("VGG16",             0.941, 59.0, 319),
        ("MobileNetV2",       0.938,  9.6, 191),
        ("MobileNetV3-Small", 0.935,  4.3, 165),
        ("Baseline CNN",      0.925,  1.6,  61),
    ],
    columns=["Model", "Macro F1", "Size (MB)", "CPU ms / image"],
)
TEAM = "Raul II A. Patenio and Samuel C. Usares"   # edit before deploying
SIZE_TARGET_MB = 25
LATENCY_TARGET_MS = 1000


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def get_predictor():
    _, backbone, head = core.load_model()
    return core.build_predictor(backbone, head)


@st.cache_data
def get_card():
    return core.load_json(core.CARD_PATH)


@st.cache_data
def get_fert():
    return core.load_json(core.FERT_PATH)


card = get_card()
fert = get_fert()
CLASSES = card["class_names"]
THRESHOLD = float(card["confidence_threshold"])
SHORT = {"Nitrogen(N)": "N", "Phosphorus(P)": "P", "Potassium(K)": "K"}

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 LeafNPK")
    st.caption("Rice leaf N, P, K deficiency checker")

    locations = list(fert["prices"].keys())
    location = st.selectbox(
        "Fertilizer prices for",
        locations,
        index=locations.index(fert["meta"]["default_location"]),
        help="FPA average retail prices per 50 kg bag.",
        key="location",
    )
    st.caption(
        f"Prices: FPA {fert['meta']['price_report_week']}, {fert['meta']['price_period']}."
    )

    st.markdown("### 📷 Taking a good photo")
    st.markdown(
        "- One leaf, filling most of the frame\n"
        "- Shade or even daylight, **not** direct sun or flash\n"
        "- Normal colour (no filters, not black-and-white)\n"
        "- In focus; plain background if possible"
    )

    st.markdown("### 🔒 Privacy")
    st.caption(
        "Photos are processed in memory to make the prediction and are not saved or "
        "shared. The results log below lives only in your browser session."
    )

    st.markdown("### Model")
    st.caption(
        f"{card['model_name']} · {core.MODEL_PATH.stat().st_size / 1e6:.1f} MB · "
        f"test accuracy {card['test_accuracy']:.1%} · macro F1 {card['test_macro_f1']:.3f}"
    )


# ---------------------------------------------------------------------------
# Helpers for the Diagnose tab
# ---------------------------------------------------------------------------
def probability_chart(probs) -> alt.Chart:
    top = int(probs.argmax())
    df = pd.DataFrame(
        {
            "Class": [fert["classes"][c]["label"] for c in CLASSES],
            "Probability": probs,
            "Top": [i == top for i in range(len(CLASSES))],
        }
    )
    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, size=20)
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%", grid=False, title="Probability")),
            y=alt.Y("Class:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.condition("datum.Top", alt.value(BLUE), alt.value(GRAY)),
            tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".1%")],
        )
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("Probability:Q", format=".0%"), color=alt.value("#52514e")
    )
    rule = (
        alt.Chart(pd.DataFrame({"Probability": [THRESHOLD]}))
        .mark_rule(strokeDash=[4, 3], color="#52514e")
        .encode(x="Probability:Q",
                tooltip=[alt.Tooltip("Probability:Q", title="Confidence threshold", format=".0%")])
    )
    return (bars + labels + rule).properties(height=alt.Step(40))


def reliability_text(cls: str) -> str:
    r = card["reliability"][cls]
    n = r["test_predictions"]
    correct = round(r["precision"] * n)
    txt = (
        f"On the held-out test set, the model said **{fert['classes'][cls]['label'].lower()}** "
        f"{n} times and was right **{correct} of {n}** ({r['precision']:.0%})."
    )
    if r["most_confused_with"]:
        other = fert["classes"][r["most_confused_with"]]["label"].lower()
        txt += f" The rest were actually **{other}**."
    return txt


def fertilizer_table(cls: str) -> pd.DataFrame:
    info = fert["classes"][cls]
    key, label = info["nutrient_key"], info["nutrient_label"]
    prices = fert["prices"][location]
    rows = []
    for pk in info["fertilizers"]:
        p = fert["products"][pk]
        price = prices.get(pk)
        rows.append(
            {
                "Fertilizer": p["name"],
                "Grade": p["grade"],
                "₱ / 50 kg bag": price,
                f"₱ / kg {label}": core.cost_per_kg_nutrient(price, p[key]),
            }
        )
    return pd.DataFrame(rows)


def run_diagnosis(source, name: str):
    try:
        img, x = core.prepare_image(source, card["img_size"])
    except Exception:
        st.error("Could not read that file as an image. Please upload a JPG or PNG photo.")
        return

    quality = core.check_quality(img)
    probs, cam, ms = core.predict_with_gradcam(get_predictor(), x)
    top = int(probs.argmax())
    cls = CLASSES[top]
    conf = float(probs[top])
    info = fert["classes"][cls]
    certain = conf >= THRESHOLD

    # ---- Photo checks ------------------------------------------------------
    for w in quality.warnings:
        st.warning("📷 " + w)

    # ---- Result -------------------------------------------------------------
    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        t1, t2 = st.tabs(["Heatmap (Grad-CAM)", "Original photo"])
        with t1:
            st.image(core.overlay_heatmap(img, cam), width="stretch")
            st.caption(
                "Red and yellow areas influenced the prediction most. If the hot spots are on "
                "the background instead of the leaf, do not trust the result; retake the photo."
            )
        with t2:
            st.image(img, width="stretch")

    with col_res:
        if certain:
            st.success(f"### {info['label']}\nConfidence **{conf:.0%}**")
        else:
            order = probs.argsort()[::-1]
            second = fert["classes"][CLASSES[order[1]]]["label"].lower()
            st.warning(
                f"### Uncertain result\n"
                f"Best guess: **{info['label'].lower()}** at {conf:.0%}, "
                f"next: {second} at {probs[order[1]]:.0%}. This is below the "
                f"{THRESHOLD:.0%} confidence threshold, so **do not buy fertilizer based on this "
                f"result**. Retake the photo (see tips in the sidebar) or confirm with a soil test "
                f"or your Municipal Agriculture Office."
            )
        st.altair_chart(probability_chart(probs), width="stretch")
        st.caption(f"Dashed line = {THRESHOLD:.0%} confidence threshold · analysed in {ms:.0f} ms")
        st.info("📊 **How reliable is this answer?** " + reliability_text(cls))

    st.session_state.history.append(
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Photo": name,
            "Prediction": info["label"],
            "Confidence": round(conf, 3),
            "Status": "Confident" if certain else "Uncertain",
            "Photo warnings": len(quality.warnings),
        }
    )

    # ---- Guidance -----------------------------------------------------------
    st.divider()
    heading = info["label"] if certain else f"If it is {info['label'].lower()}"
    st.subheader(f"What to do next · {heading}")

    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown("**What it usually looks like**")
        st.markdown(info["symptoms"])
        st.markdown("**Next steps**")
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(info["next_steps"], 1)))
    with g2:
        st.markdown("**⚠️ If this result is wrong**")
        st.markdown(info["wrong_prediction"])

    st.markdown(f"**Fertilizer options and prices · {location}**")
    df = fertilizer_table(cls)
    cost_col = df.columns[-1]
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Grade": st.column_config.TextColumn(help="N - P₂O₅ - K₂O, % by weight"),
            "₱ / 50 kg bag": st.column_config.NumberColumn(format="%.2f"),
            cost_col: st.column_config.NumberColumn(
                format="%.2f",
                help="Bag price ÷ kg of this nutrient in the bag. Lower = cheaper source of "
                "this nutrient. Multi-nutrient products also supply other nutrients.",
            ),
        },
    )
    cheapest = df.dropna(subset=[cost_col]).sort_values(cost_col).iloc[0]
    st.caption(
        f"Cheapest source of {info['nutrient_label']} here: **{cheapest['Fertilizer']}** "
        f"(₱{cheapest[cost_col]:,.2f} per kg). {info['fertilizer_notes']}"
    )
    st.caption(
        f"Source: {fert['meta']['price_source']}, {fert['meta']['price_report_week']} "
        f"({fert['meta']['price_period']}). [Report]({fert['meta']['price_url']}) · "
        f"[All weekly prices]({fert['meta']['price_index_page']}). Prices change weekly; "
        f"empty = no FPA data for this location."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🌾 LeafNPK")
st.markdown(
    "Upload or take a photo of a rice leaf to check for **nitrogen (N)**, **phosphorus (P)** "
    "or **potassium (K)** deficiency, see where the model looked, and get next steps with "
    "current FPA fertilizer prices."
)

tab_diag, tab_model, tab_about = st.tabs(["🔍 Diagnose", "📊 Model & data", "ℹ️ About"])

with tab_diag:
    sample_files = sorted(SAMPLES_DIR.glob("*.[jp][pn]g")) if SAMPLES_DIR.exists() else []
    modes = ["Upload a photo", "Take a photo"] + (["Try a sample leaf"] if sample_files else [])
    mode = st.radio("Input", modes, horizontal=True, label_visibility="collapsed", key="mode")

    source, name = None, None
    if mode == "Upload a photo":
        up = st.file_uploader("Rice leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if up is not None:
            source, name = io.BytesIO(up.getvalue()), up.name
    elif mode == "Take a photo":
        cam_img = st.camera_input("Point the camera at one rice leaf")
        if cam_img is not None:
            source, name = io.BytesIO(cam_img.getvalue()), "camera photo"
    else:
        pick = st.selectbox("Sample leaf", sample_files, format_func=lambda p: p.stem, key="sample")
        source, name = pick, pick.name

    if source is not None:
        run_diagnosis(source, name)
    else:
        st.info("Waiting for a leaf photo. Tips for a good photo are in the sidebar.")

    st.divider()
    st.caption(
        "**Disclaimer.** LeafNPK is a student research prototype and a screening aid only. "
        "It recognises only N, P and K deficiency: it cannot tell a healthy leaf, a disease "
        "(such as blast or bacterial leaf blight), pest damage, or other nutrient problems, "
        "and will still pick one of the three. Results can be wrong. Confirm with a soil test, "
        "the PhilRice MOET kit or Leaf Color Chart, or your Municipal Agriculture Office before "
        "buying or applying fertilizer. Prices are FPA averages and vary by store."
    )

    if st.session_state.history:
        with st.expander(f"Results this session ({len(st.session_state.history)})"):
            hist = pd.DataFrame(st.session_state.history)
            st.dataframe(hist, hide_index=True, width="stretch")
            st.download_button(
                "Download as CSV",
                hist.to_csv(index=False).encode(),
                file_name="leafnpk_results.csv",
                mime="text/csv",
            )

with tab_model:
    st.subheader("Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leaf images", "1,156")
    c2.metric("Classes", "3 (N, P, K)")
    c3.metric("Train / val / test", "809 / 173 / 174")
    c4.metric("Test accuracy (deployed model)", f"{card['test_accuracy']:.1%}")
    counts = pd.DataFrame({"Class": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"], "Images": [440, 333, 383]})
    st.altair_chart(
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4, size=20, color=BLUE)
        .encode(
            x=alt.X("Images:Q", axis=alt.Axis(grid=False)),
            y=alt.Y("Class:N", title=None, sort=None),
            tooltip=["Class", "Images"],
        )
        .properties(height=alt.Step(40), title="Images per class"),
        width="stretch",
    )
    st.caption(
        "Source: Kaggle 'Nutrient Deficiency Symptoms in Rice' (guy007), public RGB leaf images. "
        "Audited for corrupt files and duplicates; near-duplicate pairs kept in the same split; "
        "group-aware stratified 70/15/15 split (seed 42)."
    )

    st.subheader("Which model to deploy?")
    st.markdown(
        f"Six CNNs were trained and compared (3 seeds each). Deployment targets: model file "
        f"**under {SIZE_TARGET_MB} MB** and **under {LATENCY_TARGET_MS / 1000:.0f} second per image** on CPU."
    )
    mr = MODEL_RESULTS.copy()
    mr["Meets targets"] = (mr["Size (MB)"] < SIZE_TARGET_MB) & (mr["CPU ms / image"] < LATENCY_TARGET_MS)
    mr["Deployed"] = mr["Model"] == "EfficientNet-B0"
    xs = alt.Scale(type="log", domain=[1, 150], clamp=True)
    ys = alt.Scale(domain=[0.915, 0.97], zero=False)
    base = alt.Chart(mr).encode(
        x=alt.X("Size (MB):Q", scale=xs, title="Model file size (MB, log scale)", axis=alt.Axis(grid=False)),
        y=alt.Y("Macro F1:Q", scale=ys, title="Macro F1 (mean of 3 seeds)", axis=alt.Axis(format=".2f")),
        tooltip=["Model", alt.Tooltip("Macro F1:Q", format=".3f"), "Size (MB)", "CPU ms / image", "Meets targets"],
    )
    pts = base.mark_point(filled=True, size=140, stroke="white", strokeWidth=2, opacity=1).encode(
        color=alt.condition("datum.Deployed", alt.value(BLUE), alt.value(GRAY))
    )
    txt = base.mark_text(align="center", dy=-18, color="#52514e").encode(text="Model")
    target = (
        alt.Chart(pd.DataFrame({"x": [SIZE_TARGET_MB]}))
        .mark_rule(strokeDash=[4, 3], color="#52514e")
        .encode(x=alt.X("x:Q", scale=xs))
    )
    target_lbl = (
        alt.Chart(pd.DataFrame({"x": [SIZE_TARGET_MB], "y": [0.918], "t": ["25 MB size target →  too big"]}))
        .mark_text(align="left", dx=6, color="#52514e")
        .encode(x=alt.X("x:Q", scale=xs), y=alt.Y("y:Q", scale=ys), text="t:N")
    )
    st.altair_chart((target + target_lbl + pts + txt).properties(height=340), width="stretch")
    st.dataframe(
        mr.drop(columns=["Deployed"]),
        hide_index=True,
        width="stretch",
        column_config={"Macro F1": st.column_config.NumberColumn(format="%.3f")},
    )
    st.markdown(
        "**Choice: EfficientNet-B0.** It is the most accurate model that meets both targets "
        "(ResNet50 scored slightly higher but is 95 MB). No pairwise difference between models "
        "was statistically significant, so deployment cost decided between them. MobileNetV3-Small "
        "(4.3 MB) is kept for a future offline phone version."
    )

    st.subheader("How reliable is each answer?")
    rel = pd.DataFrame(
        [
            {
                "When the app says": fert["classes"][c]["label"],
                "Right (test set)": f"{round(r['precision'] * r['test_predictions'])} of {r['test_predictions']}",
                "Precision": f"{r['precision']:.1%}",
                "Otherwise it was": fert["classes"][r["most_confused_with"]]["label"] if r["most_confused_with"] else "—",
            }
            for c, r in card["reliability"].items()
        ]
    )
    st.dataframe(rel, hide_index=True, width="stretch")
    st.markdown(
        "- **76% of all errors** across the six models were phosphorus/potassium mix-ups, mostly P "
        "predicted as K. The app therefore tells users to confirm K results with a soil test.\n"
        "- **Robustness:** pretrained models handled moderate lighting changes but failed on "
        "grayscale photos and strong overexposure (brightness ×1.30). The app checks every photo "
        "for these conditions and warns the user.\n"
        f"- Below **{THRESHOLD:.0%}** confidence the app reports *uncertain* instead of a diagnosis."
    )

with tab_about:
    st.subheader("The problem")
    st.markdown(
        "Nitrogen, phosphorus and potassium deficiencies reduce rice yield, but they are hard to "
        "tell apart by eye, and lab soil tests take time and money. Farmers often guess, which "
        "wastes fertilizer and money and can leave the real deficiency untreated."
    )
    st.subheader("Who it is for")
    st.markdown(
        "Rice farmers and agricultural extension workers (for example Municipal Agriculture Office "
        "technicians) in Davao Region who need a quick first check from a phone photo, before "
        "confirming with a soil test or field kit."
    )
    st.subheader("How it works")
    st.markdown(
        "1. The photo is checked for brightness, overexposure, colour and size.\n"
        "2. An EfficientNet-B0 model (transfer learning, trained on 809 leaf images) predicts N, P or K.\n"
        f"3. Below {THRESHOLD:.0%} confidence the result is reported as uncertain.\n"
        "4. Grad-CAM shows which parts of the leaf drove the prediction.\n"
        "5. The app shows symptoms, next steps, FPA fertilizer prices and cost per kg of nutrient, "
        "and what happens if the prediction is wrong."
    )
    st.subheader("Limitations")
    st.markdown(
        "- Only three classes: no 'healthy' or 'disease' class, so any leaf gets labelled N, P or K.\n"
        "- Trained on one public dataset; accuracy on local field photos from Davao is not yet measured.\n"
        "- Photo-based screening cannot replace soil testing for fertilizer rates.\n"
        "- Prices are weekly FPA averages and must be updated in `fertilizer_info.json`."
    )
    st.subheader("Team")
    st.markdown(
        f"{TEAM} · College of Information and Computing, "
        "University of Southeastern Philippines (USeP)"
    )
    st.subheader("References")
    st.markdown("\n".join(f"- [{r['title']}]({r['url']})" for r in fert["references"]))
