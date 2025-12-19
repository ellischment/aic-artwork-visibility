# Artwork Visibility Prediction  
**Art Institute of Chicago · ML Final Project**

## Overview

This project builds a machine learning system to **estimate the visibility potential of artworks in a museum collection** using structured metadata from the Art Institute of Chicago (AIC).

The goal is **not to replace curators**, but to provide a **data-driven decision support tool** that helps:
- identify potentially underexposed artworks,
- understand drivers of visibility,
- optimize exhibition, publication and digital infrastructure strategies.

This project models a real museum business process:

**Visibility → Exhibition → Publication → Commercial monetization**

---

## Problem formulation

We solve a **binary classification task**:

> **Predict whether an artwork is “visible”**, based only on information that would be available *before* future exhibitions, publications or commercial actions.

### Why “visibility” is a meaningful target
Visibility aggregates multiple public-facing signals:
- exhibition history,
- presence in publications,
- presence in museum products,
- digital accessibility (zoom, multimedia, educational content).

This reflects **actual audience exposure**, not internal storage or curatorial intent.

---

## Dataset

- **Source**: Art Institute of Chicago Open Data
- **Size**: ~134,000 artworks, 57 raw columns
- **Granularity**: artwork-level
- **Images**: referenced via IIIF (`image_id`), not stored locally

Dataset is downloaded automatically via `gdown`.

> The project intentionally uses **one large notebook** to simplify review and defense.  
> Later stages may already appear in the file — please focus on the relevant checkpoint when grading.

---

## Feature engineering

### Artistic block
We constructed a compact, interpretable representation of artworks:

- **Categorical**
  - `classification_title`
  - `department_title` → KFold Target Encoding
  - `style_title` (missing values treated as meaningful state)

- **Size & image proxies**
  - `side_normalised`
  - `thumbnail_area`

- **Medium**
  - 40 binary `md_*` features (top materials)
  - `medium_word_count`
  - `TruncatedSVD` → `md_svd_*`
  - `md_cluster` (KMeans)

- **Text (lightweight, non-overfitted)**
  - `title_word_count`
  - `title_topic_*` (TF-IDF + SVD, 20 components)
  - `credit_line_word_count`

Raw text fields are **not** passed to the model.

---

### Anomalies & structure
- IQR-based outlier flags for sizes and text lengths
- Isolation Forest anomaly score
- Helps separate “typical archive objects” from exceptional works

---

## Leakage control (critical design choice)

All features that **directly or indirectly encode future visibility** were removed:
- future exhibition signals,
- future publications,
- `last_visible_year` and similar look-ahead features.

This ensures the model predicts **future visibility**, not reconstructs the label.

---

## Models

### Model A — Production model
- **CatBoostClassifier**
- Clean feature set
- 5-fold cross-validation
- Metrics:
  - **ROC-AUC ≈ 0.90**
  - **F1 ≈ 0.51**

This is the **realistic performance** expected in deployment.
Lower F1 reflects class imbalance and conservative precision/recall trade-off.

---

### Model B — SHAP-only (analytical, not deployable)
- Trained only on SHAP-derived embeddings
- Achieves near-perfect metrics due to information leakage
- **Used only for analysis and interpretation**

---

### Model C — Hybrid (features + SHAP structure)
- Similar raw quality to Model A
- **Much more stable across epochs, object types and departments**
- Errors become structured and explainable

> For museums, **stability and interpretability are more important than marginal AUC gains**.

---

## Interpretation & analysis

### SHAP
- Global feature importance
- Local explanations for individual artworks
- Clear separation of:
  - author/context block,
  - artistic properties,
  - digital infrastructure

### SHAP embeddings & clustering
- Reveal **visibility regimes**
- Allow reasoning like:
  > “This artwork is invisible not because of quality, but because its digital layer is weak.”

---

### Shapley Flow (network view)
- Correlation graph of SHAP contributions
- Shows **how features act together**, not independently
- Key insight:
  - Digital infrastructure acts as a **catalyst**, not a replacement for artistic quality.

---

## Practical conclusions for museums

- Digital accessibility (zoom, multimedia, metadata completeness) can **move artworks between visibility regimes**
- For archival works, digital actions alone are insufficient — curatorial context matters
- For contemporary works, weak digital packaging can nullify strong artistic value

The model provides **actionable levers**, not just predictions.

!!! Do not use “Run all” blindly — some stages are computationally heavy. !!!

Disclaimer

Uses AIC Open Data

No copyrighted images stored

Model is analytical, not a curatorial authority
