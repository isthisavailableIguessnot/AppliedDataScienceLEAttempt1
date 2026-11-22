# 🌾 LeafNPK — Rice Leaf N, P, K Deficiency Checker

A Streamlit web app that classifies a rice leaf photo as **nitrogen (N)**, **phosphorus (P)** or
**potassium (K)** deficient, shows a Grad-CAM heatmap of where the model looked, and gives next
steps with current FPA fertilizer prices for Davao Region.

Applied Data Science project · College of Information and Computing, University of Southeastern
Philippines (USeP).

## What the app does

| Feature | Details |
|---|---|
| Input | Upload a JPG/PNG or take a photo with the phone/laptop camera |
| Photo checks | Warns about overexposure, very dark photos, grayscale/filtered photos and tiny images (the model's known weak spots from the robustness tests) |
| Prediction | EfficientNet-B0 (17 MB), probabilities for N / P / K |
| Uncertainty | Below 0.60 confidence the result is shown as **Uncertain** with advice not to buy fertilizer |
| Reliability note | Per-class precision on the held-out test set and which class it is most often confused with |
| Explainability | Grad-CAM heatmap over the photo |
| Guidance | Symptoms, next steps, fertilizer options, FPA prices, ₱ per kg of nutrient, effects of a wrong prediction |
| Dashboard | "Model & data" tab: dataset, six-model comparison against deployment targets, reliability, robustness |
| Privacy | Photos are processed in memory and never stored; the results log stays in the browser session |

Measured on a 2-core CPU: **~30–60 ms per image** for prediction + Grad-CAM (warm), model file
**17.0 MB** — both well inside the targets (< 1 s, < 25 MB).

## Files

```
app.py                  Streamlit UI
npk_core.py             model loading, prediction + Grad-CAM, photo checks
fertilizer_info.json    symptoms, next steps, fertilizers, FPA prices, references
model/efficientnetb0.keras        trained model (representative seed 3)
model/efficientnetb0_card.json    class names, threshold, test metrics, per-class reliability
requirements.txt
.streamlit/config.toml
samples/ (optional)     add a few test-split leaf photos to enable "Try a sample leaf"
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Create a **public** GitHub repository (e.g. `leafnpk`) and upload everything in this folder,
   keeping the folder structure. The model is 17 MB, under GitHub's 25 MB browser-upload limit.
   Make sure `.streamlit/config.toml` is included (it is a hidden folder).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app → Deploy a public
   app from GitHub**.
3. Pick the repo, branch `main`, main file `app.py`. Under **Advanced settings** choose
   **Python 3.12**. Optionally set a custom subdomain such as `leafnpk-usep`.
4. Click **Deploy**. The first build installs TensorFlow and takes a few minutes.
5. Open the app, upload a leaf photo, and check the "analysed in … ms" line.

**Keep it awake for the evaluation.** Community Cloud puts apps to sleep after 12 hours without
traffic; anyone can wake it with one click, but the restart takes a minute or two. Open the app
yourself shortly before the panel (and the morning of the evaluation period) so it is warm.

**Memory.** TensorFlow alone uses ~800 MB; the app runs at ~1.0–1.4 GB. Community Cloud allows up
to ~2.7 GB, but if you ever see "This app has gone over its resource limits", reboot it from the
app menu (Manage app → Reboot).

## Updating fertilizer prices

FPA posts weekly prices at <https://fpa.da.gov.ph/weekly-prices/>. Open the latest
"Fertilizer" report, copy the **REGION XI**, **Davao City** and **National Weighted Average** rows
into `fertilizer_info.json → prices`, and update `meta.price_report_week`, `meta.price_period`,
`meta.price_url` and `meta.retrieved`. Column order in the FPA table: Urea (prilled), Urea
(granular), Ammosul, Complete, Ammophos, Muriate of Potash, DAP. Use `null` for a dash.

## Technical notes

- **Preprocessing** is inside the model: the app feeds raw 0–255 RGB resized to 224×224 (same as
  training), after fixing phone EXIF rotation.
- **Grad-CAM** uses the output of the `backbone` layer (7×7×1280). Prediction and heatmap come from
  one compiled forward/backward pass (`tf.function`), traced once at start-up.
- **Streamlit + Keras:** Streamlit calls `keras.backend.clear_session()` after every script run,
  which wipes TensorFlow's kernel cache and made each new prediction take ~0.7 s. Because this app
  loads one model once and never builds new ones, `npk_core.load_model()` turns that call into a
  no-op, bringing predictions back to ~30–60 ms.
- **Photo-check thresholds** are constants at the top of `npk_core.py`.

## Limitations

- Three classes only: a healthy or diseased leaf is still labelled N, P or K.
- Trained and tested on one public Kaggle dataset; accuracy on local Davao field photos is unknown.
- A screening aid, not a replacement for soil testing (DA Regional Soils Laboratory, PhilRice MOET
  kit, Leaf Color Chart) or advice from the Municipal Agriculture Office.

## Data and sources

- Dataset: Kaggle "Nutrient Deficiency Symptoms in Rice" (guy007), 1,156 RGB leaf images.
- Prices: Fertilizer and Pesticide Authority (FPA), weekly fertilizer prices.
- Agronomy: ATI (DA), IRRI Rice Knowledge Bank, PhilRice — full list in `fertilizer_info.json`.
