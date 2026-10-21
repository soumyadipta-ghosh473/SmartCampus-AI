# SmartCampus-AI

AI-based Smart Campus complaint classification system.

## Model

The system uses a multitask DistilBERT model trained to predict:

1. Ticket Type
2. Responsible Department
3. Priority

## Model Files

The trained model and tokenizer are located inside:

model/

## Application

The Streamlit application is:

app.py

## Run

Install dependencies:

pip install -r requirements.txt

Then run:

streamlit run app.py

## Included Data

- smartcampus_synthetic_dataset.csv
- smartcampus_final_metrics.csv
- smartcampus_type_mapping.csv
- smartcampus_department_mapping.csv
- smartcampus_priority_mapping.csv

## Experimental AI/ML Component

The project uses a trained multitask DistilBERT neural network rather than an external prediction API.
