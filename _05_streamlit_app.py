import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from bertopic import BERTopic
import base64
import io
import re
import spacy
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)

# Set up Streamlit app
st.set_page_config(
    page_title="SciReX: AI-Powered Research Paper Reviewer",
    page_icon="📚",
    layout="wide"
)

# Load NLP model for NER
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load the summarization model
@st.cache_resource
def load_summarization_model():
    model_path = "models/fine_tuned_bart"
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_summarization_model()

# Load the topic model
@st.cache_resource
def load_topic_model():
    return BERTopic.load("models/bertopic_model")

topic_model = load_topic_model()

# Load research gaps data
@st.cache_data
def load_research_gaps():
    return pd.read_csv('data/results/research_gaps.csv')

research_gaps = load_research_gaps()

# Function to clean text
def clean_text(text):
    """Clean text by removing special characters, extra spaces, etc."""
    # Convert to lowercase
    text = text.lower()

    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Function to extract named entities
def extract_entities(text):
    """Extract named entities using spaCy"""
    doc = nlp(text)
    entities = {}

    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)

    return entities

# Function to extract methods mentioned in text
def extract_methods(text):
    """Extract potential AI methods and techniques from text"""
    method_patterns = [
        r'\b(?:Conv(?:olutional)?|Recurrent|Transformer|LSTM|GRU|MLP)\s+Neural\s+Networks?\b',
        r'\bGAN(?:s)?\b',
        r'\bVAE(?:s)?\b',
        r'\bBERT(?:-[A-Za-z]+)?\b',
        r'\bGPT(?:-[0-9]+)?\b',
        r'\bT5(?:-[A-Za-z0-9]+)?\b',
        r'\bResNet(?:-[0-9]+)?\b',
        r'\bVGG(?:-[0-9]+)?\b',
        r'\bEfficientNet(?:-[A-Za-z0-9]+)?\b',
        r'\bU-Net\b',
        r'\bYOLO(?:v[0-9]+)?\b',
        r'\bDiffusion\s+Models?\b',
        r'\bNeural\s+Radiance\s+Fields?\b',
        r'\bNeRF\b',
        r'\bContrastive\s+Learning\b',
        r'\bSelf-(?:Supervised|Attention)\b',
        r'\bReinforcement\s+Learning\b',
        r'\bTransfer\s+Learning\b',
        r'\bMulti(?:-|\s+)Modal(?:ity)?\b'
    ]

    methods = []
    for pattern in method_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        methods.extend([m.strip() for m in matches if m.strip()])

    return list(set(methods))

# Function to extract metrics
def extract_metrics(text):
    """Extract evaluation metrics mentioned in text"""
    metric_patterns = [
        r'\bBLEU\b',
        r'\bROUGE(?:-[LN1-9])?\b',
        r'\bMETEOR\b',
        r'\bPPL\b',
        r'\bPerplexity\b',
        r'\bAccuracy\b',
        r'\bPrecision\b',
        r'\bRecall\b',
        r'\bF1(?:-Score)?\b',
        r'\bMAE\b',
        r'\bMSE\b',
        r'\bRMSE\b',
        r'\bAUC(?:-ROC)?\b',
        r'\bFID\b',
        r'\bInception\s+Score\b',
    ]

    metrics = []
    for pattern in metric_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        metrics.extend([m.strip() for m in matches if m.strip()])

    return list(set(metrics))

# Generate summary function
def generate_summary(abstract):
    """Generate a summary using the fine-tuned BART model"""
    inputs = tokenizer(abstract, max_length=512, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=10,
        max_length=64,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to predict topics
def predict_topic(text):
    """Predict topics using the BERTopic model"""
    topics, probs = topic_model.transform([text])
    return topics[0], topic_model.get_topic(topics[0])

# Function to analyze research gaps
def analyze_research_gaps(text):
    """Analyze research gaps based on the text"""
    # Extract methods and techniques
    methods = extract_methods(text)

    # Check which research gaps are being addressed
    addressed_gaps = []
    unaddressed_gaps = []

    # Simplified approach: check if any keywords from the gap are in the methods
    for _, row in research_gaps.iterrows():
        gap_topic = row['topic']
        gap_keywords = gap_topic.replace('_', ' ').split()

        if any(keyword.lower() in text.lower() for keyword in gap_keywords):
            addressed_gaps.append({
                'topic': gap_topic,
                'priority': row['priority'],
                'impact_score': row['impact_score']
            })
        else:
            unaddressed_gaps.append({
                'topic': gap_topic,
                'priority': row['priority'],
                'impact_score': row['impact_score']
            })

    # Sort by impact score
    addressed_gaps = sorted(addressed_gaps, key=lambda x: x['impact_score'], reverse=True)
    unaddressed_gaps = sorted(unaddressed_gaps, key=lambda x: x['impact_score'], reverse=True)

    return addressed_gaps, unaddressed_gaps

# Function to download results as CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.title("🧠 SciReX: AI-Powered Research Paper Reviewer")
st.markdown("""
This tool helps researchers analyze academic papers by:
- Summarizing the paper content
- Extracting key information (methods, datasets, metrics)
- Analyzing research gaps
- Suggesting potential research directions
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Paper Analysis", "Research Gaps", "About"])

with tab1:
    st.header("Paper Analysis")

    # Input text area for paper abstract
    paper_abstract = st.text_area("Paste paper abstract or full text here:", height=250)

    if st.button("Analyze Paper"):
        if paper_abstract:
            with st.spinner("Analyzing paper content..."):
                # Clean text
                cleaned_text = clean_text(paper_abstract)

                # Create columns for results
                col1, col2 = st.columns(2)

                with col1:
                    # Generate summary
                    st.subheader("Summary")
                    summary = generate_summary(paper_abstract)
                    st.write(summary)

                    # Extract entities
                    st.subheader("Named Entities")
                    entities = extract_entities(paper_abstract)

                    if 'ORG' in entities:
                        st.write("**Organizations:**")
                        st.write(", ".join(list(set(entities['ORG']))[:10]))

                    if 'PERSON' in entities:
                        st.write("**People:**")
                        st.write(", ".join(list(set(entities['PERSON']))[:10]))

                with col2:
                    # Extract methods
                    st.subheader("AI Methods & Techniques")
                    methods = extract_methods(paper_abstract)
                    if methods:
                        for method in methods:
                            st.write(f"- {method}")
                    else:
                        st.write("No specific AI methods detected.")

                    # Extract metrics
                    st.subheader("Evaluation Metrics")
                    metrics = extract_metrics(paper_abstract)
                    if metrics:
                        for metric in metrics:
                            st.write(f"- {metric}")
                    else:
                        st.write("No evaluation metrics detected.")

                # Topic analysis
                st.subheader("Topic Analysis")
                topic_id, topic_words = predict_topic(paper_abstract)

                if topic_id != -1:  # Not an outlier
                    st.write(f"**Main topic:** Topic {topic_id}")
                    st.write("**Topic words:**")
                    topic_df = pd.DataFrame(topic_words, columns=["Word", "Weight"])
                    st.dataframe(topic_df.head(10))
                else:
                    st.write("The paper doesn't fit well into any of the discovered topics (outlier).")

                # Research gap analysis
                st.subheader("Research Gap Analysis")
                addressed_gaps, unaddressed_gaps = analyze_research_gaps(paper_abstract)

                if addressed_gaps:
                    st.write("**Research gaps addressed by this paper:**")
                    for gap in addressed_gaps:
                        st.write(f"- {gap['topic'].replace('_', ' ').title()} (Priority: {gap['priority']})")

                if unaddressed_gaps:
                    st.write("**Research gaps NOT addressed in this paper:**")
                    for i, gap in enumerate(unaddressed_gaps[:3]):  # Show top 3 unaddressed gaps
                        st.write(f"- {gap['topic'].replace('_', ' ').title()} (Priority: {gap['priority']})")

                # Tokenize for word count
                tokens = word_tokenize(paper_abstract)

                # Create analysis results
                analysis_results = {
                    "Summary": summary,
                    "Word Count": len(tokens),
                    "AI Methods": ", ".join(methods),
                    "Evaluation Metrics": ", ".join(metrics),
                    "Main Topic": f"Topic {topic_id}" if topic_id != -1 else "Outlier",
                    "Addressed Gaps": ", ".join([g['topic'].replace('_', ' ').title() for g in addressed_gaps]),
                    "Unaddressed Gaps": ", ".join([g['topic'].replace('_', ' ').title() for g in unaddressed_gaps[:3]])
                }

                # Create download link
                results_df = pd.DataFrame([analysis_results])
                st.markdown(get_csv_download_link(results_df, "paper_analysis.csv", "📥 Download Analysis Results"), unsafe_allow_html=True)
        else:
            st.error("Please enter paper content before analyzing.")

with tab2:
    st.header("Research Gap Analysis")

    # Display research gap visualization
    st.subheader("Current Research Gaps in Generative AI")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    research_gaps_viz = research_gaps.copy()
    research_gaps_viz['topic'] = research_gaps_viz['topic'].str.replace('_', ' ').str.title()
    sns.barplot(x='impact_score', y='topic', hue='priority', data=research_gaps_viz, palette='viridis', ax=ax)
    ax.set_xlabel('Impact Score')
    ax.set_ylabel('Research Topic')
    ax.set_title('Research Gaps by Priority')
    st.pyplot(fig)

    # Display table of gaps
    st.subheader("Prioritized Research Gaps")

    # Format table
    display_gaps = research_gaps.copy()
    display_gaps['topic'] = display_gaps['topic'].str.replace('_', ' ').str.title()
    display_gaps['coverage'] = display_gaps['coverage'].round(2)
    display_gaps['impact_score'] = display_gaps['impact_score'].round(3)
    display_gaps.columns = ['Topic', 'Coverage (%)', 'Impact Score', 'Priority']

    st.dataframe(display_gaps)

    # Recommendations
    st.subheader("Research Recommendations")
    st.write("""
    Based on our analysis, the following research directions could have significant impact:

    1. **Ethics & Alignment**: More work needed on ensuring generative AI models align with human values and intent
    2. **Interpretability & Explainability**: Developing methods to understand and explain generative model decisions
    3. **Robustness & Security**: Building generative models resistant to adversarial attacks and misuse
    4. **Efficiency & Resource Usage**: Improving generative model performance with reduced computational requirements
    5. **Uncertainty Quantification**: Methods for reliable confidence estimation in generative outputs
    """)

with tab3:
    st.header("About SciReX")
    st.write("""
    **SciReX** is an AI-powered research paper analysis tool developed to help researchers and academics with literature review and gap analysis in the field of Generative AI.

    ### Key Features:
    - **Paper Summarization**: Using fine-tuned BART model to create concise summaries
    - **Information Extraction**: Identifying methods, metrics, datasets and entities
    - **Topic Analysis**: Using BERTopic to identify research themes
    - **Gap Analysis**: Identifying under-explored research areas

    ### Technologies Used:
    - **PyTorch & Hugging Face** for model fine-tuning and inference
    - **BERTopic** for topic modeling
    - **spaCy** for named entity recognition
    - **Streamlit** for the user interface

    ### Data Sources:
    The system was trained on papers from arXiv, primarily in the field of Generative AI and deep learning.
    """)

    st.write("---")
    st.write("© 2025 SciReX - AI-Powered Research Paper Reviewer")

# Footer
st.sidebar.header("SciReX Options")
st.sidebar.write("Upload a research paper abstract or full text to analyze its content, identify methods, and detect research gaps.")

# Sample abstracts
st.sidebar.subheader("Try Sample Abstracts")
sample_abstracts = {
    "Diffusion Models": """Diffusion models are a class of generative models that have recently shown impressive results in image synthesis tasks. These models work by gradually adding Gaussian noise to training data and then learning to reverse this process to generate new data samples. In this paper, we present a novel approach to improving diffusion models through adaptive noise scheduling and guided generation. Our approach achieves state-of-the-art results on several image generation benchmarks, including CIFAR-10 and ImageNet. We evaluate our method using FID scores and human evaluation studies. Additionally, we explore applications in text-to-image generation and image editing tasks.""",

    "Multimodal Learning": """Multimodal learning has become increasingly important as AI systems need to process and understand information from different modalities such as text, images, and audio. In this work, we introduce a new transformer-based architecture for multimodal representation learning that effectively aligns features across modalities. Our model employs a novel cross-attention mechanism and contrastive learning objective. We evaluate our approach on vision-language tasks including image-text retrieval and visual question answering, demonstrating significant improvements over existing methods. Our experiments show that our model achieves better generalization to unseen data and exhibits stronger zero-shot capabilities."""
}

selected_sample = st.sidebar.selectbox("Select a sample abstract", list(sample_abstracts.keys()))
if st.sidebar.button("Load Sample"):
    st.session_state['abstract_input'] = sample_abstracts[selected_sample]
    st.rerun()

# Info about the project
st.sidebar.markdown("---")
st.sidebar.info("""
    **SciReX** helps researchers stay informed about the latest developments and research gaps in their field.

    This project is part of a larger initiative to accelerate scientific discovery through AI assistance.
    """)
st.sidebar.markdown("---")