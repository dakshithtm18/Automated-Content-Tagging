import os
import json
from datetime import datetime, timezone

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "model_cache"))

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Automated Content Tagging",
    page_icon="🏷️",
    layout="wide"
)

MODEL_DIR = "saved_model"
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pth")
TAXONOMY_PATH = "dataset/final_tag_taxonomy_phase4_complete.csv"
MIN_WORDS = 20

torch.set_num_threads(4)


def is_approved(value):
    """Accept only explicit positive approval values; PENDING is never approved."""
    return str(value).strip().lower() in {"true", "1", "yes", "approved"}


@st.cache_data
def load_approved_taxonomy(path):
    if not os.path.exists(path):
        fallback = "dataset/final_tag_taxonomy.csv"
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                "Approved taxonomy not found. Expected "
                "dataset/final_tag_taxonomy_phase4_complete.csv."
            )
        path = fallback

    taxonomy = pd.read_csv(path, dtype="string").fillna("")
    required = {"tag", "approved"}
    if not required.issubset(taxonomy.columns):
        raise ValueError("Taxonomy must contain 'tag' and 'approved' columns.")

    taxonomy["tag"] = taxonomy["tag"].astype(str).str.strip()
    approved = sorted({
        row["tag"] for _, row in taxonomy.iterrows()
        if row["tag"] and is_approved(row["approved"])
    })
    return approved, taxonomy


@st.cache_resource
def load_model():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            "saved_model/classifier.pth is missing. Restore the trained "
            "checkpoint before generating predictions."
        )

    encoder = SentenceTransformer(config["model_name"], device="cpu")
    classifier = nn.Linear(
        config["embedding_dimension"],
        len(config["eligible_labels"])
    )
    state_dict = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=True
    )
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return encoder, classifier, config


def predict_tags(title, article_text, encoder, classifier, config, approved_tags):
    """Rank only labels that are explicitly approved in the current taxonomy."""
    model_labels = config["eligible_labels"]
    approved_set = set(approved_tags)
    approved_indices = [
        index for index, label in enumerate(model_labels)
        if label in approved_set
    ]

    if not approved_indices:
        raise ValueError(
            "The trained model has no labels matching the approved taxonomy. "
            "Retrain the model using the updated approved-only training script."
        )

    content = (title.strip() + " " + article_text.strip()).strip()
    embeddings = encoder.encode(
        [content],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    input_tensor = torch.tensor(embeddings, dtype=torch.float32)

    with torch.no_grad():
        logits = classifier(input_tensor)
        probabilities = torch.sigmoid(logits)[0]

    approved_scores = probabilities[approved_indices]
    top_k = min(int(config.get("top_k", 5)), len(approved_indices))
    scores, local_indices = torch.topk(approved_scores, top_k)

    return [
        {
            "tag": model_labels[approved_indices[index]],
            "confidence": round(float(score) * 100, 2)
        }
        for score, index in zip(scores.tolist(), local_indices.tolist())
    ]


def clear_inputs():
    st.session_state["title_input"] = ""
    st.session_state["article_input"] = ""
    st.session_state["post_id_input"] = ""
    st.session_state["current_result"] = None
    st.session_state["editor_tags"] = []


def ensure_state():
    st.session_state.setdefault("current_result", None)
    st.session_state.setdefault("saved_records", [])
    st.session_state.setdefault("editor_tags", [])


ensure_state()

# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------
st.title("🏷️ Automated Content Tagging")
st.write(
    "Generate blog tags with a trained Sentence Transformer model, "
    "then review and approve the final tags as an editor."
)
st.info(
    "Only tags explicitly marked approved in the project taxonomy can be "
    "suggested or selected as final editor-approved tags."
)

st.divider()
st.subheader("1. Enter Blog Details")

post_id = st.text_input(
    "Post ID",
    placeholder="Enter an existing post ID, or leave blank to generate one",
    key="post_id_input"
)
title = st.text_input(
    "Blog Title",
    placeholder="Enter your blog title...",
    key="title_input"
)
article_text = st.text_area(
    "Blog Content",
    placeholder="Paste your blog article content here...",
    height=250,
    key="article_input"
)

word_count = len(article_text.split())
st.caption(f"Word count: {word_count} (minimum for automatic tagging: {MIN_WORDS})")

col_generate, col_clear = st.columns(2)
with col_generate:
    generate = st.button("Generate / Review Tags", type="primary", use_container_width=True)
with col_clear:
    st.button("Clear", use_container_width=True, on_click=clear_inputs)

if generate:
    effective_post_id = post_id.strip() or (
        "POST-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    )
    try:
        approved_tags, _ = load_approved_taxonomy(TAXONOMY_PATH)
        if not title.strip() and not article_text.strip():
            st.session_state["current_result"] = {
                "post_id": effective_post_id,
                "title": title.strip(),
                "article_text": article_text.strip(),
                "word_count": word_count,
                "suggestions": [],
                "review_status": "NEEDS_REVIEW",
                "review_reason": "Empty title and article content",
            }
            st.session_state["editor_tags"] = []
        elif word_count < MIN_WORDS:
            st.session_state["current_result"] = {
                "post_id": effective_post_id,
                "title": title.strip(),
                "article_text": article_text.strip(),
                "word_count": word_count,
                "suggestions": [],
                "review_status": "NEEDS_REVIEW",
                "review_reason": f"Article contains fewer than {MIN_WORDS} words",
            }
            st.session_state["editor_tags"] = []
        else:
            with st.spinner("Loading model and generating approved-taxonomy tags..."):
                encoder, classifier, config = load_model()
                suggestions = predict_tags(
                    title, article_text, encoder, classifier, config, approved_tags
                )
            st.session_state["current_result"] = {
                "post_id": effective_post_id,
                "title": title.strip(),
                "article_text": article_text.strip(),
                "word_count": word_count,
                "suggestions": suggestions,
                "review_status": "PENDING_EDITOR_REVIEW",
                "review_reason": "",
            }
            st.session_state["editor_tags"] = [
                item["tag"] for item in suggestions
            ]
    except Exception as error:
        st.error(f"Unable to process this article: {error}")

# --------------------------------------------------
# EDITOR REVIEW AND APPROVAL
# --------------------------------------------------
current = st.session_state.get("current_result")
if current:
    st.divider()
    st.subheader("2. Editor Review")

    st.write(f"**Post ID:** `{current['post_id']}`")
    st.write(f"**Review status:** `{current['review_status']}`")

    if current["review_status"] == "NEEDS_REVIEW":
        st.warning(
            "This article has been routed to manual review. "
            + current["review_reason"]
            + ". No automatic tags have been approved."
        )
    else:
        st.write("**Suggested tags and model scores**")
        for item in current["suggestions"]:
            st.write(f"- **{item['tag']}** — {item['confidence']:.2f}% model score")

    try:
        approved_tags, _ = load_approved_taxonomy(TAXONOMY_PATH)
        if current["review_status"] != "NEEDS_REVIEW":
            st.caption(
                "Keep selected suggestions to accept them; remove suggestions to reject them. "
                "You may also add other approved taxonomy tags."
            )
        final_tags = st.multiselect(
            "Final editor-approved tags",
            options=approved_tags,
            default=[
                tag for tag in st.session_state.get("editor_tags", [])
                if tag in set(approved_tags)
            ],
            key="editor_tags"
        )
    except Exception as error:
        st.error(f"Cannot load approved tags for editor review: {error}")
        final_tags = []

    if st.button("Save Editor Decision", type="primary"):
        record = {
            "post_id": current["post_id"],
            "title": current["title"],
            "word_count": current["word_count"],
            "suggested_tags": [x["tag"] for x in current["suggestions"]],
            "suggestion_scores": {
                x["tag"]: x["confidence"] for x in current["suggestions"]
            },
            "review_status": (
                "EDITOR_APPROVED" if final_tags else
                ("NEEDS_REVIEW" if current["review_status"] == "NEEDS_REVIEW"
                 else "EDITOR_REJECTED_ALL")
            ),
            "review_reason": current["review_reason"],
            "final_editor_approved_tags": list(final_tags),
            "decision_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        records = st.session_state["saved_records"]
        existing = next(
            (i for i, old in enumerate(records) if old["post_id"] == record["post_id"]),
            None
        )
        if existing is None:
            records.append(record)
        else:
            records[existing] = record
        st.session_state["saved_records"] = records
        st.success("Editor decision saved in this session. Download the export below.")

# --------------------------------------------------
# CSV / JSON EXPORT
# --------------------------------------------------
st.divider()
st.subheader("3. Export Review Results")

records = st.session_state["saved_records"]
if records:
    export_rows = []
    for record in records:
        export_rows.append({
            "post_id": record["post_id"],
            "title": record["title"],
            "word_count": record["word_count"],
            "suggested_tags": "|".join(record["suggested_tags"]),
            "suggestion_scores": json.dumps(record["suggestion_scores"], ensure_ascii=False),
            "review_status": record["review_status"],
            "review_reason": record["review_reason"],
            "final_editor_approved_tags": "|".join(
                record["final_editor_approved_tags"]
            ),
            "decision_timestamp_utc": record["decision_timestamp_utc"],
        })

    csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig")
    json_bytes = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="editor_tagging_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with download_col2:
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="editor_tagging_results.json",
            mime="application/json",
            use_container_width=True
        )
    st.dataframe(pd.DataFrame(export_rows), use_container_width=True)
else:
    st.caption("No editor decisions saved yet. Generate or review an article and save a decision.")

st.divider()
st.caption(
    "Powered by Sentence Transformers and PyTorch. Model scores are ranking scores, "
    "not calibrated probabilities. Editor approval is required before final tags are exported."
)
