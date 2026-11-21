# Automated Content Tagging for Blogs

An AI-powered blog content tagging application developed using Python, Streamlit, and Sentence Transformers.

## Features
- Automatic tag suggestions for blog articles
- Trained Sentence Transformer classification model
- Approved taxonomy-based tag selection
- Manual editor review and approval
- CSV and JSON export of tagging results

## Technologies Used
- Python
- Streamlit
- PyTorch
- Sentence Transformers
- Pandas
- Scikit-learn

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Automated_Content_Tagging/
├── app.py
├── requirements.txt
├── README.md
├── dataset/
│   └── final_tag_taxonomy_phase4_complete.csv
└── saved_model/
    ├── classifier.pth
    └── model_config.json
```

Developed as an Artificial Intelligence minor project.