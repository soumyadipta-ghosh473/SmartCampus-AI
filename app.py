import os
import glob
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

# ---------------------------------------------------------
# Label mappings
# ---------------------------------------------------------

type_mapping = pd.read_csv(
    os.path.join(BASE_DIR, "smartcampus_type_mapping.csv")
)
department_mapping = pd.read_csv(
    os.path.join(BASE_DIR, "smartcampus_department_mapping.csv")
)
priority_mapping = pd.read_csv(
    os.path.join(BASE_DIR, "smartcampus_priority_mapping.csv")
)

type_labels = type_mapping["label_name"].tolist()
department_labels = department_mapping["label_name"].tolist()
priority_labels = priority_mapping["label_name"].tolist()

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class CampusMultiTaskDistilBERT(nn.Module):

    def __init__(self, model_name, num_type, num_queue, num_priority):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size
        dropout = getattr(
            self.encoder.config,
            "seq_classif_dropout",
            0.1
        )

        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, num_type)
        self.queue_head = nn.Linear(hidden, num_queue)
        self.priority_head = nn.Linear(hidden, num_priority)

    def forward(self, input_ids=None, attention_mask=None):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.dropout(
            outputs.last_hidden_state[:, 0]
        )

        return (
            self.type_head(pooled),
            self.queue_head(pooled),
            self.priority_head(pooled)
        )

@st.cache_resource
def load_model():

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(
            MODEL_DIR,
            "tokenizer.json"
        )
    )

    model = CampusMultiTaskDistilBERT(
        MODEL_NAME,
        len(type_labels),
        len(department_labels),
        len(priority_labels)
    )

    state_dict = load_file(
        os.path.join(MODEL_DIR, "model.safetensors")
    )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_complaint(text):

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        type_logits, department_logits, priority_logits = model(
            **inputs
        )

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
        "Department Confidence": float(
            department_probs[department_id]
        ),
        "Priority": priority_labels[priority_id],
        "Priority Confidence": float(
            priority_probs[priority_id]
        )
    }

# ---------------------------------------------------------
# Ticket storage
# ---------------------------------------------------------

def save_ticket(complaint, prediction):

    from datetime import datetime

    row = {
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Complaint": complaint,
        "Ticket Type": prediction["Ticket Type"],
        "Department": prediction["Department"],
        "Priority": prediction["Priority"],
        "Type Confidence": round(
            prediction["Type Confidence"], 4
        ),
        "Department Confidence": round(
            prediction["Department Confidence"], 4
        ),
        "Priority Confidence": round(
            prediction["Priority Confidence"], 4
        )
    }

    if os.path.exists(TICKETS_FILE):
        df = pd.read_csv(TICKETS_FILE)
        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )
    else:
        df = pd.DataFrame([row])

    df.to_csv(TICKETS_FILE, index=False)

# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartCampus-AI",
    page_icon="ðŸŽ“",
    layout="wide"
)

st.title("ðŸŽ“ SmartCampus-AI")
st.subheader(
    "Intelligent Student Support Ticket Routing & Priority Prediction"
)

st.write(
    "Submit a student complaint and the multi-task DistilBERT "
    "model will predict the ticket type, responsible department "
    "and priority."
)

tab1, tab2 = st.tabs(
    ["Student Complaint Portal", "Admin Dashboard"]
)

with tab1:

    st.header("Submit a Complaint")

    complaint = st.text_area(
        "Describe your complaint",
        height=160,
        placeholder=(
            "Example: The computer lab Wi-Fi is not working "
            "and students cannot access the internet."
        )
    )

    if st.button("Analyze Complaint", type="primary"):

        if not complaint.strip():

            st.warning("Please enter a complaint.")

        else:

            prediction = predict_complaint(complaint)

            st.success("Complaint analyzed successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Ticket Type",
                    prediction["Ticket Type"]
                )
                st.caption(
                    f"Confidence: "
                    f"{prediction['Type Confidence']*100:.2f}%"
                )

            with col2:
                st.metric(
                    "Department",
                    prediction["Department"]
                )
                st.caption(
                    f"Confidence: "
                    f"{prediction['Department Confidence']*100:.2f}%"
                )

            with col3:
                st.metric(
                    "Priority",
                    prediction["Priority"]
                )
                st.caption(
                    f"Confidence: "
                    f"{prediction['Priority Confidence']*100:.2f}%"
                )

            if st.button("Create Support Ticket"):

                save_ticket(
                    complaint,
                    prediction
                )

                st.success(
                    "Support ticket created successfully."
                )

with tab2:

    st.header("Admin Dashboard")

    if os.path.exists(TICKETS_FILE):

        tickets = pd.read_csv(TICKETS_FILE)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total Tickets",
                len(tickets)
            )

        with c2:
            high_count = int(
                (tickets["Priority"].astype(str).str.lower() == "high").sum()
            )
            st.metric(
                "High Priority",
                high_count
            )

        with c3:
            dept_count = tickets["Department"].nunique()
            st.metric(
                "Departments",
                dept_count
            )

        st.subheader("Ticket Distribution")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Priority")
            st.bar_chart(
                tickets["Priority"].value_counts()
            )

        with col2:
            st.write("Department")
            st.bar_chart(
                tickets["Department"].value_counts()
            )

        st.subheader("Recent Tickets")

        st.dataframe(
            tickets.tail(20),
            use_container_width=True
        )

        csv_data = tickets.to_csv(index=False)

        st.download_button(
            "Download Ticket Data",
            csv_data,
            file_name="smartcampus_tickets.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No tickets have been created yet."
        )

