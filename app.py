
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoModel, PreTrainedTokenizerFast
from safetensors.torch import load_file

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TICKETS_FILE = os.path.join(BASE_DIR, "smartcampus_tickets.csv")
MODEL_NAME = "distilbert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKET_COLUMNS = [
    "Timestamp",
    "Complaint",
    "Ticket Type",
    "Department",
    "Priority",
    "Type Confidence",
    "Department Confidence",
    "Priority Confidence",
]

# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartCampus-AI",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------
# Label mappings
# ---------------------------------------------------------

@st.cache_data
def load_mappings():
    type_mapping = pd.read_csv(
        os.path.join(BASE_DIR, "smartcampus_type_mapping.csv")
    )
    department_mapping = pd.read_csv(
        os.path.join(BASE_DIR, "smartcampus_department_mapping.csv")
    )
    priority_mapping = pd.read_csv(
        os.path.join(BASE_DIR, "smartcampus_priority_mapping.csv")
    )

    return (
        type_mapping["label_name"].tolist(),
        department_mapping["label_name"].tolist(),
        priority_mapping["label_name"].tolist(),
    )


type_labels, department_labels, priority_labels = load_mappings()

# ---------------------------------------------------------
# Multitask model
# ---------------------------------------------------------

class CampusMultiTaskDistilBERT(nn.Module):
    def __init__(self, model_name, num_type, num_queue, num_priority):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(
            self.encoder.config,
            "seq_classif_dropout",
            0.1,
        )

        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, num_type)
        self.queue_head = nn.Linear(hidden, num_queue)
        self.priority_head = nn.Linear(hidden, num_priority)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        return (
            self.type_head(pooled),
            self.queue_head(pooled),
            self.priority_head(pooled),
        )


@st.cache_resource
def load_model():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.json"),
        pad_token="[PAD]",
    )

    # Explicitly register the existing DistilBERT padding token. This is
    # important because tokenizer.json alone may omit special-token metadata.
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # The pretrained DistilBERT vocabulary uses [PAD] at ID 0. Do not add
    # a new embedding or resize the trained model's embedding matrix.
    if tokenizer.pad_token_id != 0:
        raise ValueError(
            f"Expected DistilBERT [PAD] token ID 0, got {tokenizer.pad_token_id}. "
            "Check the saved tokenizer.json before running SHAP."
        )

    model = CampusMultiTaskDistilBERT(
        MODEL_NAME,
        len(type_labels),
        len(department_labels),
        len(priority_labels),
    )

    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    state_dict = load_file(weights_path)

    # Fail clearly if the saved weights do not match the architecture.
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as exc:
    st.error(f"Could not load the AI model: {exc}")
    st.stop()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_complaint(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        type_logits, department_logits, priority_logits = model(**inputs)

    type_probs = torch.softmax(type_logits, dim=-1)[0]
    department_probs = torch.softmax(department_logits, dim=-1)[0]
    priority_probs = torch.softmax(priority_logits, dim=-1)[0]

    type_id = int(torch.argmax(type_probs))
    department_id = int(torch.argmax(department_probs))
    priority_id = int(torch.argmax(priority_probs))

    return {
        "Ticket Type": type_labels[type_id],
        "Type Confidence": float(type_probs[type_id]),
        "Department": department_labels[department_id],
        "Department Confidence": float(department_probs[department_id]),
        "Priority": priority_labels[priority_id],
        "Priority Confidence": float(priority_probs[priority_id]),
    }


# ---------------------------------------------------------
# SHAP explainability for the three DistilBERT output heads
# ---------------------------------------------------------

def predict_probabilities_for_shap(texts):
    """
    SHAP calls this function with batches of masked complaint texts.
    Return all three heads' probabilities in one matrix so the encoder
    is evaluated once per SHAP batch.
    """
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(text) for text in texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.inference_mode():
        type_logits, department_logits, priority_logits = model(**inputs)

        type_probs = torch.softmax(type_logits, dim=-1)
        department_probs = torch.softmax(department_logits, dim=-1)
        priority_probs = torch.softmax(priority_logits, dim=-1)

    return np.concatenate(
        [
            type_probs.cpu().numpy(),
            department_probs.cpu().numpy(),
            priority_probs.cpu().numpy(),
        ],
        axis=1,
    )


@st.cache_resource
def load_shap_explainer():
    # DistilBERT uses [MASK] as its token replacement baseline.
    # Pass the tokenizer with an explicit, registered pad token to SHAP.
    masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")

    output_names = (
        [f"Ticket Type: {label}" for label in type_labels]
        + [f"Department: {label}" for label in department_labels]
        + [f"Priority: {label}" for label in priority_labels]
    )

    return shap.Explainer(
        predict_probabilities_for_shap,
        masker,
        algorithm="partition",
        output_names=output_names,
    )


# ---------------------------------------------------------
# Ticket storage
# ---------------------------------------------------------

def read_tickets():
    if not os.path.exists(TICKETS_FILE):
        return pd.DataFrame(columns=TICKET_COLUMNS)

    try:
        tickets = pd.read_csv(TICKETS_FILE)
        for column in TICKET_COLUMNS:
            if column not in tickets.columns:
                tickets[column] = None
        return tickets[TICKET_COLUMNS]
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read the ticket file: {exc}") from exc


def save_ticket(complaint, prediction):
    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Complaint": complaint,
        "Ticket Type": prediction["Ticket Type"],
        "Department": prediction["Department"],
        "Priority": prediction["Priority"],
        "Type Confidence": round(prediction["Type Confidence"], 4),
        "Department Confidence": round(prediction["Department Confidence"], 4),
        "Priority Confidence": round(prediction["Priority Confidence"], 4),
    }

    try:
        tickets = read_tickets()
        tickets = pd.concat(
            [tickets, pd.DataFrame([row])],
            ignore_index=True,
        )

        # Write to a temporary file first to reduce the risk of a partial CSV.
        temporary_file = TICKETS_FILE + ".tmp"
        tickets.to_csv(temporary_file, index=False)
        os.replace(temporary_file, TICKETS_FILE)

    except Exception as exc:
        raise RuntimeError(f"Could not save the ticket: {exc}") from exc


# ---------------------------------------------------------
# Application interface
# ---------------------------------------------------------

st.title("🎓 SmartCampus-AI")
st.subheader("Intelligent Student Support Ticket Routing & Priority Prediction")
st.write(
    "Submit a student complaint. The multitask DistilBERT model predicts "
    "the ticket type, responsible department, and priority."
)

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "analyzed_complaint" not in st.session_state:
    st.session_state.analyzed_complaint = ""

if "ticket_created_for" not in st.session_state:
    st.session_state.ticket_created_for = None

if "shap_values" not in st.session_state:
    st.session_state.shap_values = None

if "shap_complaint" not in st.session_state:
    st.session_state.shap_complaint = None

tab1, tab2 = st.tabs(["Student Complaint Portal", "Admin Dashboard"])

# ---------------------------------------------------------
# Student portal
# ---------------------------------------------------------

with tab1:
    st.header("Submit a Complaint")

    complaint = st.text_area(
        "Describe your complaint",
        height=160,
        placeholder=(
            "Example: The computer lab Wi-Fi is not working "
            "and students cannot access the internet."
        ),
        key="complaint_input",
    )

    if st.button("Analyze Complaint", type="primary"):
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            try:
                with st.spinner("Analyzing complaint..."):
                    st.session_state.prediction = predict_complaint(complaint.strip())
                    st.session_state.analyzed_complaint = complaint.strip()
                    st.session_state.ticket_created_for = None
                    st.session_state.shap_values = None
                    st.session_state.shap_complaint = None
            except Exception as exc:
                st.error(f"Complaint analysis failed: {exc}")

    prediction = st.session_state.prediction
    analyzed_complaint = st.session_state.analyzed_complaint

    if prediction and analyzed_complaint:
        st.success("Complaint analyzed successfully.")
        st.caption(f"Complaint: {analyzed_complaint}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Ticket Type", prediction["Ticket Type"])
            st.caption(f"Confidence: {prediction['Type Confidence'] * 100:.2f}%")

        with col2:
            st.metric("Department", prediction["Department"])
            st.caption(
                f"Confidence: {prediction['Department Confidence'] * 100:.2f}%"
            )

        with col3:
            st.metric("Priority", prediction["Priority"])
            st.caption(
                f"Confidence: {prediction['Priority Confidence'] * 100:.2f}%"
            )

        st.caption(
            "Confidence is the model's estimated probability for its selected "
            "label; it is not a guarantee of correctness."
        )

        # -------------------------------------------------
        # SHAP explanation panel
        # -------------------------------------------------
        st.divider()
        st.subheader("Explainable AI — Why did the model make this prediction?")
        st.write(
            "Generate word-level SHAP explanations for the selected "
            "ticket type, department, and priority predictions."
        )

        if st.button("Generate SHAP Explanations", key="generate_shap"):
            try:
                with st.spinner(
                    "Calculating SHAP values for the three model outputs. "
                    "This may take a little while..."
                ):
                    explainer = load_shap_explainer()

                    # The explanation uses the same 256-token input limit as prediction.
                    token_count = len(
                        tokenizer(
                            analyzed_complaint,
                            truncation=True,
                            max_length=256,
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                    max_evals = max(2 * token_count + 1, 3)

                    st.session_state.shap_values = explainer(
                        [analyzed_complaint],
                        max_evals=max_evals,
                    )
                    st.session_state.shap_complaint = analyzed_complaint

            except Exception as exc:
                st.session_state.shap_values = None
                st.session_state.shap_complaint = None
                st.error(f"SHAP explanation could not be generated: {exc}")

        shap_values = st.session_state.shap_values
        if shap_values is not None and st.session_state.shap_complaint == analyzed_complaint:
            st.caption(
                "Positive bars support the selected class; negative bars "
                "push the model away from that class. SHAP describes model "
                "behaviour and does not establish causation."
            )

            tasks = [
                (
                    "Ticket Type",
                    type_labels,
                    "Type Confidence",
                    0,
                ),
                (
                    "Department",
                    department_labels,
                    "Department Confidence",
                    len(type_labels),
                ),
                (
                    "Priority",
                    priority_labels,
                    "Priority Confidence",
                    len(type_labels) + len(department_labels),
                ),
            ]

            for task_name, labels, confidence_key, output_offset in tasks:
                predicted_label = prediction[task_name]
                predicted_id = labels.index(predicted_label)
                output_index = output_offset + predicted_id

                st.markdown(f"**{task_name}: {predicted_label}**")
                selected_explanation = shap_values[0, :, output_index]

                fig, ax = plt.subplots(figsize=(10, 4))
                plt.sca(ax)
                shap.plots.bar(
                    selected_explanation,
                    max_display=12,
                    show=False,
                )
                ax.set_title(f"Word contributions — {task_name}")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

        if st.session_state.ticket_created_for == analyzed_complaint:
            st.success("This complaint has already been saved as a support ticket.")
        elif st.button("Create Support Ticket", type="primary"):
            try:
                save_ticket(analyzed_complaint, prediction)
                st.session_state.ticket_created_for = analyzed_complaint
                st.success("Support ticket created successfully.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

# ---------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------

with tab2:
    st.header("Admin Dashboard")

    if st.button("Refresh Dashboard"):
        st.rerun()

    try:
        tickets = read_tickets()

        if tickets.empty:
            st.info("No tickets have been created yet.")
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Total Tickets", len(tickets))

            with c2:
                high_count = int(
                    tickets["Priority"].astype(str).str.strip().str.lower().eq("high").sum()
                )
                st.metric("High Priority", high_count)

            with c3:
                st.metric("Departments", tickets["Department"].nunique())

            st.subheader("Ticket Distribution")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Priority")
                st.bar_chart(tickets["Priority"].value_counts())

            with col2:
                st.write("Department")
                st.bar_chart(tickets["Department"].value_counts())

            st.subheader("Recent Tickets")
            st.dataframe(
                tickets.tail(20).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Ticket Data",
                data=tickets.to_csv(index=False).encode("utf-8"),
                file_name="smartcampus_tickets.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Dashboard could not load tickets: {exc}")
