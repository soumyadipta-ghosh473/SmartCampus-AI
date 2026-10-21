
import os
from datetime import datetime

import pandas as pd
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
        tokenizer_file=os.path.join(MODEL_DIR, "tokenizer.json")
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
