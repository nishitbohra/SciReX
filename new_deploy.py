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
import matplotlib
from io import BytesIO
import uuid
import time
from datetime import datetime
import os
import pymupdf
import spacy
import fitz
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import PyPDF2
import io
import tempfile
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Add this function to the beginning of your code to track analyzed methods
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'analyzed_methods' not in st.session_state:
        st.session_state['analyzed_methods'] = {}
    if 'historical_data' not in st.session_state:
        st.session_state['historical_data'] = None

# Call this at the beginning of your app
initialize_session_state()

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

# Function to build historical data based on analyzed methods
def build_historical_data_from_analyses():
    """Build historical paper data based on analyzed methods"""
    if not st.session_state['analyzed_methods']:
        return None
    
    data = []
    all_methods = set()
    years_range = set()
    
    # Collect all unique methods and years
    for paper_title, methods_info in st.session_state['analyzed_methods'].items():
        for item in methods_info:
            all_methods.add(item['method'])
            years_range.add(item['year'])
    
    # Ensure we have a reasonable range of years
    current_year = datetime.now().year
    years_range = sorted(list(years_range))
    if len(years_range) < 2:
        # If we only have one year or none, create a 5-year range ending at current year
        years_range = list(range(current_year - 4, current_year + 1))
    
    # Count method occurrences by year
    method_year_counts = {}
    for paper_title, methods_info in st.session_state['analyzed_methods'].items():
        for item in methods_info:
            method = item['method']
            year = item['year']
            key = (method, year)
            
            if key not in method_year_counts:
                method_year_counts[key] = 0
            method_year_counts[key] += 1
    
    # Build data structure for visualization
    for method in all_methods:
        for year in range(min(years_range), max(years_range) + 1):
            count = method_year_counts.get((method, year), 0)
            data.append({
                "year": year,
                "topic": method,
                "paper_count": count
            })
    
    # Return as DataFrame
    return pd.DataFrame(data)

# Load historical papers data (new for trend analysis)
@st.cache_data
def load_historical_papers():
    """Load historical papers data from session state or create synthetic data if none"""
    # First check if we have real data from analyzed papers
    dynamic_data = build_historical_data_from_analyses()
    if dynamic_data is not None and not dynamic_data.empty:
        return dynamic_data
    
    # If no real data, use synthetic data as before
    years = list(range(2015, 2026))
    topics = [
        "Diffusion Models", "Transformers", "GANs", "Reinforcement Learning", 
        "Multimodal Learning", "Ethics & Alignment", "Efficiency"
    ]
    
    data = []
    for year in years:
        for topic in topics:
            # Create simulated publication count with some trends
            if topic == "Transformers":
                # Growing trend
                count = int(10 + (year - 2015) * 15 + np.random.normal(0, 5))
            elif topic == "GANs":
                # Peaked and declining
                count = int(5 + (year - 2015) * 10 - (max(0, year - 2020) * 8) + np.random.normal(0, 3))
            elif topic == "Diffusion Models":
                # Recent explosion
                count = int(max(0, (year - 2020) * 25) + np.random.normal(0, 4))
            else:
                # Steady growth
                count = int(5 + (year - 2015) * 3 + np.random.normal(0, 2))
            
            count = max(0, count)  # Ensure non-negative
            data.append({"year": year, "topic": topic, "paper_count": count})
    
    return pd.DataFrame(data)

historical_papers = load_historical_papers()
# Create necessary directories
if not os.path.exists('static'):
    os.makedirs('static')
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

# Function to save matplotlib figures and provide download links
def get_img_download_link(fig, filename, text):
    """Generate a download link for a matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

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

# Function to perform fact checking on claims
def fact_check_claims(text):
    """
    Extract potential claims from text and check their factual accuracy
    Returns a list of claims with their verification status
    """
    # Define patterns for identifying claims
    claim_patterns = [
        r'(?:We|Our|The|This)(?:\s+(?:study|paper|research|work|approach|method))?\s+(?:show|demonstrate|prove|establish|confirm|validate|verify|find|report)\s+that\s+([^.;!?]+)',
        r'(?:Results|Experiments|Analysis|Data)(?:\s+(?:show|demonstrate|prove|establish|confirm|validate|verify|indicate|suggest))\s+that\s+([^.;!?]+)',
        r'(?:It\s+is|We\s+found)\s+that\s+([^.;!?]+)',
        r'([^.;!?]+)\s+(?:outperforms|exceeds|surpasses|achieves|reaches|attains)\s+(?:the|all|previous|existing|current|other)\s+(?:state-of-the-art|SOTA|methods|approaches|models|techniques|systems|frameworks)\s+([^.;!?]+)'
    ]
    
    # Extract potential claims
    potential_claims = []
    for pattern in claim_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):  # Some patterns might return tuple groups
                for m in match:
                    if len(m.split()) > 3:  # Only consider claims with enough substance
                        potential_claims.append(m.strip())
            elif len(match.split()) > 3:  # Only consider claims with enough substance
                potential_claims.append(match.strip())
    
    # Filter duplicates and near-duplicates
    filtered_claims = []
    for claim in potential_claims:
        is_duplicate = False
        for existing in filtered_claims:
            # Calculate similarity with existing claims to avoid duplicates
            similarity = cosine_similarity(
                vectorizer.fit_transform([claim, existing])
            )[0][1]
            if similarity > 0.7:  # High similarity threshold
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_claims.append(claim)
    
    # Categorize and verify claims
    verified_claims = []
    
    # For each claim, perform verification
    for claim in filtered_claims[:10]:  # Limit to 10 claims for performance
        # Extract entities and keywords from the claim
        claim_doc = nlp(claim)
        entities = [ent.text for ent in claim_doc.ents]
        keywords = [token.lemma_ for token in claim_doc if token.is_alpha and not token.is_stop]
        
        # Check if claim contains specific metrics or values
        has_metrics = any(re.search(r'\b\d+(?:\.\d+)?%?\b', claim) for metric in metrics_list)
        has_comparison = any(word in claim.lower() for word in ['better', 'worse', 'higher', 'lower', 'more', 'less', 'increase', 'decrease'])
        
        # Determine verification approach based on claim type
        if has_metrics or has_comparison:
            # Claims with specific metrics need empirical verification
            verification = {
                'status': 'Needs Empirical Verification',
                'confidence': 'N/A',
                'explanation': 'This claim contains specific metrics or comparisons that require experimental validation.'
            }
        elif any(entity in claim for entity in ["PERSON", "ORG"]):
            # Claims about specific people/organizations need source checking
            verification = {
                'status': 'Needs Source Check',
                'confidence': 'N/A',
                'explanation': 'This claim references specific entities that should be verified against primary sources.'
            }
        else:
            # For general claims, provide a plausibility check
            # In a real system, this would connect to a knowledge base or fact checking API
            implausible_phrases = ['impossible', 'magical', 'solves all problems', 'works perfectly', 'always correct']
            
            if any(phrase in claim.lower() for phrase in implausible_phrases):
                status = 'Likely Overstated'
                confidence = 'Medium'
                explanation = 'Claim contains language suggesting overstatement or exaggeration.'
            else:
                status = 'Plausible'
                confidence = 'Low'
                explanation = 'General claim without specific red flags, but requires domain expertise to fully verify.'
            
            verification = {
                'status': status,
                'confidence': confidence,
                'explanation': explanation
            }
        
        verified_claims.append({
            'claim': claim,
            'verification': verification
        })
    
    return verified_claims

# List of common metrics in ML/AI research for fact checking
metrics_list = [
    'accuracy', 'precision', 'recall', 'f1', 'auc', 'mae', 'mse', 'rmse', 
    'bleu', 'rouge', 'meteor', 'perplexity', 'fid', 'inception score'
]

# Vectorizer for similarity calculation in fact checking
vectorizer = TfidfVectorizer()

# Function to extract methods mentioned in text
def extract_methods(text, paper_title=None, store=False):
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

    # Remove duplicates and standardize method names
    methods = list(set(methods))
    
    # If store flag is True and we have a paper title, store in session state
    if store and paper_title and methods:
        # Extract publication year if it's in the text (for historical tracking)
        years = re.findall(r'(?:19|20)\d{2}', text)
        # Filter to reasonable years and get most recent
        valid_years = [int(y) for y in years if 1990 <= int(y) <= datetime.now().year]
        pub_year = max(valid_years) if valid_years else datetime.now().year
        
        # Store in session state with timestamp
        timestamp = time.time()
        if paper_title not in st.session_state['analyzed_methods']:
            st.session_state['analyzed_methods'][paper_title] = []
        
        for method in methods:
            st.session_state['analyzed_methods'][paper_title].append({
                'method': method,
                'year': pub_year,
                'timestamp': timestamp
            })

    return methods

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

# NEW: Function to extract references
def extract_references(text):
    """Extract references from text using regex patterns"""
    # Pattern for typical citation formats - fixed patterns
    citation_patterns = [
        r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+et\s+al\.\s*[,\s]\s*(?:19|20)\d{2}\b',  # Author et al., YEAR
        r'\b(?:[A-Z][a-zA-Z\-]+)(?:\s+and\s+[A-Z][a-zA-Z\-]+)+\s*[,\s]\s*(?:19|20)\d{2}\b',  # Author and Author, YEAR
        r'\[[\d,-\s]+\]',  # [1] or [1,2,3] or [1-3]
        r'\(\s*(?:[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.)?|[A-Z][a-zA-Z\-]+(?:\s+and\s+[A-Z][a-zA-Z\-]+)+)?\s*,\s*(?:19|20)\d{2}\s*\)'  # (Author, YEAR) or (Author et al., YEAR)
    ]
    
    references = []
    for pattern in citation_patterns:
        try:
            matches = re.findall(pattern, text)
            references.extend([m.strip() for m in matches if m.strip()])
        except re.error:
            # Skip any pattern that causes regex errors
            continue
    
    # Filter out duplicates and sort
    references = list(set(references))
    
    # Extract years for temporal analysis
    years = re.findall(r'(?:19|20)\d{2}', text)
    years = [int(y) for y in years if 1900 <= int(y) <= 2025]
    
    return references, years

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
    # Define gap categories and related keywords
    gap_categories = {
        "Ethics & Alignment": ["ethics", "alignment", "bias", "fairness", "responsible", "transparency", "explainable"],
        "Efficiency & Resources": ["efficiency", "resource", "computation", "parameter", "optimization", "inference time", "latency"],
        "Robustness & Security": ["robustness", "security", "adversarial", "attack", "defense", "privacy", "safety"],
        "Interpretability": ["interpretability", "explainability", "understand", "reasoning", "explain", "interpret"],
        "Multimodal Integration": ["multimodal", "cross-modal", "vision-language", "audio-visual", "text-image"],
        "Evaluation Methods": ["evaluation", "metric", "benchmark", "baseline", "human evaluation", "perceptual"],
        "Domain Adaptation": ["domain adaptation", "transfer learning", "few-shot", "zero-shot", "generalization"],
        "Uncertainty Quantification": ["uncertainty", "confidence", "calibration", "probabilistic", "bayesian"],
        "Real-world Applications": ["application", "deployment", "production", "practical", "industry", "clinical", "medical"]
    }
    
    # Calculate gap scores based on keyword presence and context
    gap_scores = {}
    gap_context = {}
    
    lowercase_text = text.lower()
    
    for gap, keywords in gap_categories.items():
        # Count keyword occurrences
        count = 0
        context_sentences = []
        
        for keyword in keywords:
            count += lowercase_text.count(keyword)
            
            # Find sentences mentioning research gaps
            gap_phrases = ["need for", "future work", "future research", "gap", "limitation", 
                          "challenge", "unexplored", "open question", "open problem"]
                          
            for phrase in gap_phrases:
                # Look for sentences containing both the keyword and gap phrase
                pattern = r'([^.!?]*(?:' + keyword + r')[^.!?]*(?:' + phrase + r')[^.!?]*[.!?])'
                matches = re.findall(pattern, lowercase_text, re.IGNORECASE)
                context_sentences.extend(matches)
                
                # Also check reverse order
                pattern = r'([^.!?]*(?:' + phrase + r')[^.!?]*(?:' + keyword + r')[^.!?]*[.!?])'
                matches = re.findall(pattern, lowercase_text, re.IGNORECASE)
                context_sentences.extend(matches)
        
        # Calculate gap score (normalized by number of keywords)
        gap_score = count / len(keywords) if keywords else 0
        
        # Add context factor for explicit mentions of gaps
        if context_sentences:
            gap_score += len(context_sentences) * 0.5
        
        gap_scores[gap] = gap_score
        gap_context[gap] = list(set(context_sentences))  # Remove duplicates
    
    # Normalize scores
    max_score = max(gap_scores.values()) if gap_scores else 1
    for gap in gap_scores:
        gap_scores[gap] = (gap_scores[gap] / max_score) * 10  # Scale to 0-10
    
    # Determine if gaps are addressed or not (threshold-based)
    addressed_gaps = []
    unaddressed_gaps = []
    
    for gap, score in gap_scores.items():
        gap_info = {
            'topic': gap,
            'score': score,
            'priority': 'High' if score < 3 else ('Medium' if score < 6 else 'Low'),
            'impact_score': 10 - score,  # Inverse of score (lower presence = higher impact opportunity)
            'context': gap_context[gap][:3]  # Include up to 3 context sentences
        }
        
        if score < 5:  # Threshold for considering a gap unaddressed
            unaddressed_gaps.append(gap_info)
        else:
            addressed_gaps.append(gap_info)
    
    # Sort by impact score
    addressed_gaps = sorted(addressed_gaps, key=lambda x: x['impact_score'], reverse=True)
    unaddressed_gaps = sorted(unaddressed_gaps, key=lambda x: x['impact_score'], reverse=True)
    
    return addressed_gaps, unaddressed_gaps

# Function to upload Pdf's
def extract_text_from_pdf(uploaded_file):
    """Extract text content from uploaded PDF file"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the uploaded file data to the temporary file
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        # Extract text from the PDF
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
    finally:
        # Clean up the temporary file
        import os
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# NEW: Function to compare papers
def compare_papers(abstracts):
    """Compare multiple paper abstracts and identify similarities and differences"""
    if len(abstracts) < 2:
        return None, None, None
    
    # Clean abstracts
    cleaned_abstracts = [clean_text(abstract) for abstract in abstracts]
    
    # Extract methods from each paper
    methods_list = [extract_methods(abstract) for abstract in abstracts]
    
    # Extract metrics from each paper
    metrics_list = [extract_metrics(abstract) for abstract in abstracts]
    
    # Extract references from each paper
    references_list, years_list = zip(*[extract_references(abstract) for abstract in abstracts])
    
    # Calculate similarity using TF-IDF and cosine similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_abstracts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find common and unique methods/metrics/references
    all_methods = set()
    for methods in methods_list:
        all_methods.update(methods)
        
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics)
        
    all_references = set()
    for refs in references_list:
        all_references.update(refs)
    
    # Find common elements
    common_methods = set.intersection(*[set(methods) for methods in methods_list]) if methods_list else set()
    common_metrics = set.intersection(*[set(metrics) for metrics in metrics_list]) if metrics_list else set()
    common_references = set.intersection(*[set(refs) for refs in references_list]) if references_list else set()
    
    # Create comparison data
    comparison_data = {
        "similarity_matrix": similarity_matrix,
        "methods": {
            "common": list(common_methods),
            "all": list(all_methods),
            "by_paper": methods_list
        },
        "metrics": {
            "common": list(common_metrics),
            "all": list(all_metrics),
            "by_paper": metrics_list
        },
        "references": {
            "common": list(common_references),
            "all": list(all_references),
            "by_paper": references_list
        },
        "citation_years": years_list
    }
    
    return comparison_data

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
- Comparing multiple papers side by side
- Analyzing citation networks and references
- Tracking research trends over time
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Paper Analysis", "Research Gaps", "Compare Papers", "Historical Trends", "About"])

with tab1:
    st.header("Paper Analysis")

    # Create tabs for input methods
    input_tab1, input_tab2 = st.tabs(["Text Input", "PDF Upload"])
    
    with input_tab1:
        # Text input method (existing code)
        paper_abstract = st.text_area("Paste paper abstract or full text here:", height=250)
    
    with input_tab2:
        # PDF upload method (new code)
        uploaded_file = st.file_uploader("Upload a research paper PDF", type="pdf")
        if uploaded_file is not None:
            # Display a preview of the PDF (first page)
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Create a preview image of the first page
                import fitz  # PyMuPDF
                doc = fitz.open(temp_path)
                first_page = doc.load_page(0)
                pix = first_page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                
                # Convert to PIL Image and display
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Display preview and extract button
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(img, caption="PDF Preview (First Page)", width=300)
                
                with col2:
                    st.info(f"File: {uploaded_file.name}\nPages: {len(doc)}")
                    if st.button("Extract Text from PDF"):
                        with st.spinner("Extracting text from PDF..."):
                            extracted_text = extract_text_from_pdf(uploaded_file)
                            st.session_state['paper_text'] = extracted_text
                            st.success(f"Successfully extracted {len(extracted_text)} characters")
                            
                            # Show sample of extracted text
                            st.subheader("Sample of extracted text:")
                            st.text_area("Preview", value=extracted_text[:500] + "...", height=150, disabled=True)
                
                # Clean up
                doc.close()
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.info("Please try again with a different PDF file.")

    # Get paper text either from text input or from PDF extraction
    paper_text = paper_abstract
    if 'paper_text' in st.session_state and not paper_abstract:
        paper_text = st.session_state['paper_text']
        st.info("Using text extracted from PDF. You can edit it in the Text Input tab if needed.")

    if st.button("Analyze Paper"):
        if paper_text:
            with st.spinner("Analyzing paper content..."):
                # Clean text
                cleaned_text = clean_text(paper_text)

                # Create columns for results
                col1, col2 = st.columns(2)

                with col1:
                    # Generate summary
                    st.subheader("Summary")
                    summary = generate_summary(paper_text)
                    st.write(summary)

                    # Extract entities
                    st.subheader("Named Entities")
                    entities = extract_entities(paper_text)

                    if 'ORG' in entities:
                        st.write("**Organizations:**")
                        st.write(", ".join(list(set(entities['ORG']))[:10]))

                    if 'PERSON' in entities:
                        st.write("**People:**")
                        st.write(", ".join(list(set(entities['PERSON']))[:10]))
                    
                    # Extract references
                    st.subheader("References & Citations")
                    references, years = extract_references(paper_text)
                    if references:
                        st.write(f"**Detected {len(references)} citations:**")
                        for ref in references[:8]:  # Show first 8
                            st.write(f"- {ref}")
                        if len(references) > 8:
                            st.write(f"- ... and {len(references) - 8} more")
                        
                        # Citation year distribution
                        if years:
                            year_counts = Counter(years)
                            years_df = pd.DataFrame({
                                'Year': list(year_counts.keys()),
                                'Count': list(year_counts.values())
                            }).sort_values('Year')
                            
                            # Create citation histogram
                            fig, ax = plt.subplots(figsize=(8, 3))
                            sns.barplot(x='Year', y='Count', data=years_df, ax=ax)
                            ax.set_title('Citation Year Distribution')
                            ax.set_xlabel('Year')
                            ax.set_ylabel('Number of Citations')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.write("No citations detected.")

                with col2:
                    # Extract methods
                    st.subheader("AI Methods & Techniques")
                    methods = extract_methods(paper_text, "Uploaded Paper", store=True)
                    if methods:
                        for method in methods:
                            st.write(f"- {method}")
                    else:
                        st.write("No specific AI methods detected.")

                    # Extract metrics
                    st.subheader("Evaluation Metrics")
                    metrics = extract_metrics(paper_text)
                    if metrics:
                        for metric in metrics:
                            st.write(f"- {metric}")
                    else:
                        st.write("No evaluation metrics detected.")

                # Topic analysis
                st.subheader("Topic Analysis")
                topic_id, topic_words = predict_topic(paper_text)

                if topic_id != -1:  # Not an outlier
                    st.write(f"**Main topic:** Topic {topic_id}")
                    st.write("**Topic words:**")
                    topic_df = pd.DataFrame(topic_words, columns=["Word", "Weight"])
                    st.dataframe(topic_df.head(10))
                else:
                    st.write("The paper doesn't fit well into any of the discovered topics (outlier).")
                
                # Fact checking analysis
                st.subheader("Fact Checking Analysis")
                with st.spinner("Checking factual claims..."):
                        verified_claims = fact_check_claims(paper_text)
                        if verified_claims:
                            st.write(f"**Found {len(verified_claims)} verifiable claims:**")
                            for i, vc in enumerate(verified_claims):
                                with st.expander(f"Claim {i+1}: {vc['claim'][:100]}..." if len(vc['claim']) > 100 else f"Claim {i+1}: {vc['claim']}"):
                                    st.write(f"**Full claim:** {vc['claim']}")
                                    st.write(f"**Status:** {vc['verification']['status']}")
                                    if vc['verification']['confidence'] != 'N/A':
                                        st.write(f"**Confidence:** {vc['verification']['confidence']}")
                                    st.write(f"**Explanation:** {vc['verification']['explanation']}")
                        else:
                            st.write("No specific verifiable claims detected.")
                # Research gap analysis
                st.subheader("Research Gap Analysis")
                addressed_gaps, unaddressed_gaps = analyze_research_gaps(paper_text)

                if addressed_gaps:
                    st.write("**Research gaps addressed by this paper:**")
                    for gap in addressed_gaps:
                        st.write(f"- {gap['topic'].replace('_', ' ').title()} (Priority: {gap['priority']})")

                if unaddressed_gaps:
                    st.write("**Research gaps NOT addressed in this paper:**")
                    for i, gap in enumerate(unaddressed_gaps[:3]):  # Show top 3 unaddressed gaps
                        st.write(f"- {gap['topic'].replace('_', ' ').title()} (Priority: {gap['priority']})")

                # Tokenize for word count
                tokens = word_tokenize(paper_text)

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
            st.error("Please enter paper content or upload a PDF before analyzing.")

with tab2:
    st.header("Research Gap Analysis")

    st.info("This tab analyzes research gaps in your uploaded paper or text. Upload a paper in the 'Paper Analysis' tab first.")
    
    # Check if paper text exists in session state
    if 'paper_text' in st.session_state and st.session_state['paper_text']:
        paper_text = st.session_state['paper_text']
        
        # Analyze gaps
        with st.spinner("Analyzing research gaps..."):
            addressed_gaps, unaddressed_gaps = analyze_research_gaps(paper_text)
            
            # Calculate coverage percentage for each gap
            all_gaps = addressed_gaps + unaddressed_gaps
            total_impact = sum(gap['impact_score'] for gap in all_gaps) if all_gaps else 1
            
            for gap in all_gaps:
                gap['coverage'] = (gap['score'] / 10) * 100  # Convert score to percentage
            
            # Create data for visualization
            gap_viz_data = pd.DataFrame([
                {
                    'topic': gap['topic'],
                    'impact_score': gap['impact_score'],
                    'priority': gap['priority'],
                    'coverage': gap['coverage'],
                    'addressed': 'Addressed' if gap in addressed_gaps else 'Unaddressed'
                }
                for gap in all_gaps
            ])
            
            # Display research gap visualization
            st.subheader("Research Gaps Analysis")
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='impact_score', y='topic', hue='addressed', data=gap_viz_data, 
                       palette={'Addressed': 'green', 'Unaddressed': 'red'}, ax=ax)
            ax.set_xlabel('Impact Score (Higher = More Important Gap)')
            ax.set_ylabel('Research Topic')
            ax.set_title('Research Gaps Analysis Results')
            st.pyplot(fig)
            
            # Display gap coverage visualization
            st.subheader("Research Gap Coverage")
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='coverage', y='topic', hue='priority', data=gap_viz_data, 
                       palette={'High': 'red', 'Medium': 'orange', 'Low': 'green'}, ax=ax2)
            ax2.set_xlabel('Coverage in Paper (%)')
            ax2.set_ylabel('Research Topic')
            ax2.set_title('Research Gap Coverage')
            st.pyplot(fig2)
            
            # Display table of gaps
            st.subheader("Prioritized Research Gaps")
            
            # Format table for display
            display_gaps = pd.DataFrame([
                {
                    'Topic': gap['topic'],
                    'Coverage (%)': round(gap['coverage'], 1),
                    'Impact Score': round(gap['impact_score'], 2),
                    'Priority': gap['priority'],
                    'Status': 'Addressed' if gap in addressed_gaps else 'Unaddressed'
                }
                for gap in all_gaps
            ]).sort_values('Impact Score', ascending=False)
            
            st.dataframe(display_gaps)
            
            # Research Recommendations based on gaps
            st.subheader("Research Recommendations")
            
            if unaddressed_gaps:
                st.write("Based on our analysis of your paper, we recommend focusing on these research areas:")
                
                for i, gap in enumerate(unaddressed_gaps[:5], 1):
                    st.write(f"{i}. **{gap['topic']}** (Priority: {gap['priority']})")
                    if gap['context']:
                        with st.expander("Related context from your paper"):
                            for context in gap['context']:
                                st.write(f"- \"{context.strip()}\"")
                    
                    # Add specific recommendations based on gap type
                    if gap['topic'] == 'Ethics & Alignment':
                        st.write("   - Consider addressing ethical implications and alignment with human values")
                        st.write("   - Explore fairness and bias mitigation strategies")
                    elif gap['topic'] == 'Efficiency & Resources':
                        st.write("   - Investigate model compression techniques")
                        st.write("   - Explore hardware-efficient architectures")
                    elif gap['topic'] == 'Interpretability':
                        st.write("   - Develop methods to explain model decisions")
                        st.write("   - Consider incorporating attribution techniques")
            else:
                st.write("Your paper already addresses most research gaps well. Consider extending your work in these areas:")
                for i, gap in enumerate(addressed_gaps[-3:], 1):
                    st.write(f"{i}. **{gap['topic']}** - Potential to strengthen this aspect further")

            # Fact checking insights
            if 'paper_text' in st.session_state and st.session_state['paper_text']:
                st.subheader("Claim Verification Analysis")
            with st.spinner("Analyzing claims..."):
                verified_claims = fact_check_claims(paper_text)
                # Count claims by verification status
                status_counts = {}
                for vc in verified_claims:
                    status = vc['verification']['status']
                    if status not in status_counts:
                        status_counts[status] = 0
                    status_counts[status] += 1
        
                # Calculate percentage of claims needing verification
                total_claims = len(verified_claims)
                needs_verification = sum(count for status, count in status_counts.items() 
                                      if status in ['Needs Empirical Verification', 'Needs Source Check'])
        
                # Create data for visualization
                status_df = pd.DataFrame({
                    'Status': list(status_counts.keys()),
                    'Count': list(status_counts.values())
                })
        
                # Display visualization
                if not status_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x='Count', y='Status', data=status_df, palette='YlOrRd_r', ax=ax)
                    ax.set_title('Distribution of Claim Verification Status')
                    ax.set_xlabel('Number of Claims')
                    st.pyplot(fig)
                    
                    # Show verification rate
                    if total_claims > 0:
                        verification_needed = (needs_verification / total_claims) * 100
                        st.write(f"**Verification needed:** {verification_needed:.1f}% of claims require further verification")
                        
                        # Provide recommendations based on verification analysis
                        if verification_needed > 50:
                            st.warning("A high percentage of claims in this paper require verification. Consider checking the evidence supporting these claims.")
                        elif verification_needed > 20:
                            st.info("ℹ️ Some claims in this paper would benefit from additional verification.")
                        else:
                            st.success(" Most claims in this paper appear to be well-supported or general statements.")
                    
                    # List potentially problematic claims
                    problematic_claims = [vc for vc in verified_claims 
                                    if vc['verification']['status'] in ['Likely Overstated']]
                    
                    if problematic_claims:
                        st.write("**Potentially problematic claims:**")
                        for claim in problematic_claims:
                            st.write(f"- \"{claim['claim']}\" ({claim['verification']['status']})")
                    else:
                        st.write("No obviously problematic claims detected.")

            # Download gap analysis
            gap_df = pd.DataFrame([
                {
                    'Topic': gap['topic'],
                    'Coverage (%)': round(gap['coverage'], 1),
                    'Impact Score': round(gap['impact_score'], 2),
                    'Priority': gap['priority'],
                    'Status': 'Addressed' if gap in addressed_gaps else 'Unaddressed'
                }
                for gap in all_gaps
            ])
            
            st.markdown(get_csv_download_link(gap_df, "research_gap_analysis.csv", 
                                           "📥 Download Gap Analysis"), unsafe_allow_html=True)
    else:
        # Placeholder content when no paper is uploaded
        st.warning("Please upload a paper or enter text in the 'Paper Analysis' tab first.")
        
        # Sample visualization of potential gap areas
        sample_gaps = pd.DataFrame({
            'topic': ['Ethics & Alignment', 'Efficiency & Resources', 'Robustness & Security', 
                     'Interpretability', 'Multimodal Integration', 'Evaluation Methods',
                     'Domain Adaptation', 'Uncertainty Quantification', 'Real-world Applications'],
            'impact_score': [8.5, 7.8, 7.6, 7.2, 6.9, 6.5, 6.2, 5.8, 5.5],
            'priority': ['High', 'High', 'High', 'Medium', 'Medium', 'Medium', 'Medium', 'Low', 'Low']
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='impact_score', y='topic', hue='priority', data=sample_gaps, 
                   palette={'High': 'red', 'Medium': 'orange', 'Low': 'green'}, ax=ax)
        ax.set_xlabel('Impact Score (Higher = More Important Gap)')
        ax.set_ylabel('Research Topic')
        ax.set_title('Common Research Gaps in Generative AI (Sample Data)')
        st.pyplot(fig)
        
        st.info("Upload a paper to see personalized research gap analysis specific to your content.")

# Compare Papers Tab
with tab3:
    st.header("Compare Papers")
    st.write("Analyze and compare multiple research papers to identify similarities, differences, and research trends.")
    
    # Create a dynamic paper input system
    num_papers = st.number_input("Number of papers to compare", min_value=2, max_value=5, value=2)
    
    paper_abstracts = []
    paper_titles = []
        # Add option to load PDFs in batch
    st.subheader("Quick PDF Upload")
    uploaded_pdfs = st.file_uploader("Upload multiple PDFs (optional)", type="pdf", accept_multiple_files=True)
    
    if uploaded_pdfs:
        st.info(f"Uploaded {len(uploaded_pdfs)} PDF(s). Click 'Extract Text from PDFs' to process them.")
        if st.button("Extract Text from PDFs"):
            with st.spinner("Extracting text from PDFs..."):
                extracted_texts = []
                for pdf in uploaded_pdfs[:num_papers]:  # Limit to the number of papers selected
                    text = extract_text_from_pdf(pdf)
                    extracted_texts.append({
                        "title": pdf.name.replace(".pdf", ""),
                        "text": text
                    })
                
                # Store in session state
                st.session_state['extracted_papers'] = extracted_texts
                st.success(f"Successfully extracted text from {len(extracted_texts)} PDFs")
    
    # Display input fields for each paper
    for i in range(num_papers):
        col1, col2 = st.columns([1, 3])
        # Pre-fill with extracted PDF text if available
        default_title = ""
        default_text = ""

        if 'extracted_papers' in st.session_state and i < len(st.session_state['extracted_papers']):
            default_title = st.session_state['extracted_papers'][i]["title"]
            default_text = st.session_state['extracted_papers'][i]["text"][:1000] + "..." if len(st.session_state['extracted_papers'][i]["text"]) > 1000 else st.session_state['extracted_papers'][i]["text"]
        
        with col1:
            title = st.text_input(f"Paper {i+1} Title", value=f"Paper {i+1}", key=f"title_{i}")
        with col2:
            abstract = st.text_area(f"Paper {i+1} Abstract", height=150, key=f"abstract_{i}")
        paper_titles.append(title)

        # Use full extracted text if available, otherwise use what's in the text area
        if 'extracted_papers' in st.session_state and i < len(st.session_state['extracted_papers']):
            paper_abstracts.append(st.session_state['extracted_papers'][i]["text"])
        else:
            paper_abstracts.append(abstract)
    
    if st.button("Compare Papers"):
        # Check if all abstracts have content
        if all(paper_abstracts) and len(paper_abstracts) >= 2:
            with st.spinner("Comparing papers..."):
                # Get comparison data
                comparison_data = compare_papers(paper_abstracts)
                
                if comparison_data:
                    # Display similarity matrix
                    st.subheader("Paper Similarity")
                    sim_matrix = comparison_data["similarity_matrix"]
                    
                    # Create a DataFrame for the similarity matrix
                    sim_df = pd.DataFrame(sim_matrix, 
                                         index=paper_titles,
                                         columns=paper_titles)
                    
                    # Create a heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(sim_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
                    ax.set_title("Paper Similarity (Cosine Similarity)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display shared methods and metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Research Methods Comparison")
                        common_methods = comparison_data["methods"]["common"]
                        if common_methods:
                            st.write("**Shared Methods across all papers:**")
                            for method in common_methods:
                                st.write(f"- {method}")
                        else:
                            st.write("No methods shared across all papers.")
                        
                        st.write("**Methods by Paper:**")
                        for i, methods in enumerate(comparison_data["methods"]["by_paper"]):
                            if methods:
                                st.write(f"**{paper_titles[i]}:**")
                                for method in methods:
                                    st.write(f"- {method}")
                            else:
                                st.write(f"**{paper_titles[i]}:** No methods detected")
                    
                    with col2:
                        st.subheader("Evaluation Metrics Comparison")
                        common_metrics = comparison_data["metrics"]["common"]
                        if common_metrics:
                            st.write("**Shared Metrics across all papers:**")
                            for metric in common_metrics:
                                st.write(f"- {metric}")
                        else:
                            st.write("No metrics shared across all papers.")
                        
                        st.write("**Metrics by Paper:**")
                        for i, metrics in enumerate(comparison_data["metrics"]["by_paper"]):
                            if metrics:
                                st.write(f"**{paper_titles[i]}:**")
                                for metric in metrics:
                                    st.write(f"- {metric}")
                            else:
                                st.write(f"**{paper_titles[i]}:** No metrics detected")
                    
                    # Citation analysis
                    st.subheader("Citation Analysis")
                    
                    # Create citation network visualization
                    # In a more advanced implementation, this would show actual citation relationships
                    # Here we show a placeholder network based on common references
                    common_refs = comparison_data["references"]["common"]
                    all_refs = comparison_data["references"]["all"]
                    
                    if all_refs:
                        st.write(f"Total unique references across papers: {len(all_refs)}")
                        if common_refs:
                            st.write(f"Common references shared across papers: {len(common_refs)}")
                            st.write("**Top shared references:**")
                            for ref in common_refs[:5]:
                                st.write(f"- {ref}")
                        
                        # Citation years visualization (aggregate)
                        all_years = []
                        for years in comparison_data["citation_years"]:
                            all_years.extend(years)
                        
                        if all_years:
                            year_counts = Counter(all_years)
                            years_df = pd.DataFrame({
                                'Year': list(year_counts.keys()),
                                'Count': list(year_counts.values())
                            }).sort_values('Year')
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.barplot(x='Year', y='Count', data=years_df, ax=ax)
                            ax.set_title('Aggregated Citation Year Distribution')
                            ax.set_xlabel('Year')
                            ax.set_ylabel('Number of Citations')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.write("No references detected across papers.")
                    # Add download button for the chart
                    st.markdown(get_img_download_link(fig, f"Citation_analysis_{uuid.uuid4()}.png", 
                                       "Download Citations Chart"), unsafe_allow_html=True)
                    
                    # Create comparison summary
                    st.subheader("Comparison Summary")
                    
                    # Create summary table
                    summary_data = []
                    for i in range(len(paper_abstracts)):
                        summary_data.append({
                            "Paper": paper_titles[i],
                            "Methods Count": len(comparison_data["methods"]["by_paper"][i]),
                            "Metrics Count": len(comparison_data["metrics"]["by_paper"][i]),
                            "References Count": len(comparison_data["references"]["by_paper"][i]),
                            "Unique Methods": len(set(comparison_data["methods"]["by_paper"][i]) - 
                                                set().union(*[m for j, m in enumerate(comparison_data["methods"]["by_paper"]) if j != i])),
                            "Average Similarity": np.mean([sim_matrix[i][j] for j in range(len(paper_abstracts)) if j != i])
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                    
                    # Download comparison results
                    st.markdown(get_csv_download_link(summary_df, "paper_comparison.csv", 
                                                     "Download Comparison Results"), unsafe_allow_html=True)
                    # Fact checking comparison
                    st.subheader("Claim Verification Comparison")

                    # Check factual claims for each paper
                    all_verified_claims = []
                    for i, abstract in enumerate(paper_abstracts):
                        paper_claims = fact_check_claims(abstract)
                        for claim in paper_claims:
                            claim['paper'] = paper_titles[i]
                        all_verified_claims.extend(paper_claims)

                    # Group claims by verification status
                    claims_by_status = {}
                    for claim in all_verified_claims:
                        status = claim['verification']['status']
                        if status not in claims_by_status:
                            claims_by_status[status] = []
                        claims_by_status[status].append(claim)

                    # Display results by status
                    for status, claims in claims_by_status.items():
                        with st.expander(f"{status} ({len(claims)} claims)"):
                            for claim in claims:
                                st.write(f"**Paper:** {claim['paper']}")
                                st.write(f"**Claim:** {claim['claim']}")
                                st.write(f"**Explanation:** {claim['verification']['explanation']}")
                                st.write("---")

                    # Create visualization of verification status by paper
                    if all_verified_claims:
                        # Prepare data for visualization
                        viz_data = []
                        for claim in all_verified_claims:
                            viz_data.append({
                                'Paper': claim['paper'],
                                'Status': claim['verification']['status']
                            })
                        
                        viz_df = pd.DataFrame(viz_data)
                        
                        # Create count plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.countplot(x='Paper', hue='Status', data=viz_df, ax=ax)
                        ax.set_title('Claim Verification Status by Paper')
                        ax.set_xlabel('Paper')
                        ax.set_ylabel('Number of Claims')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.error("Please enter content for at least two papers to compare.")

# Historical Trends Tab
with tab4:
    st.header("Research Trends Analysis")
    st.write("Explore how research topics have evolved over time")
    
    # Check if we need to refresh historical data
    if st.button("Update Trend Data from Analyzed Papers"):
        # Force rebuild of historical data
        st.session_state['historical_data'] = build_historical_data_from_analyses()
        if st.session_state['historical_data'] is None or st.session_state['historical_data'].empty:
            st.warning("No methods have been extracted from papers yet. Please analyze papers in the Paper Analysis tab first.")
        else:
            st.success(f"Updated trend data with {len(st.session_state['historical_data']['topic'].unique())} methods from your analyzed papers.")
    
    # Get fresh data
    historical_papers = load_historical_papers()
    
    # Filter options - now using dynamic list of methods/topics
    topics_list = sorted(historical_papers['topic'].unique().tolist())
    default_topics = topics_list[:3] if len(topics_list) >= 3 else topics_list
    
    selected_topics = st.multiselect(
        "Select topics to visualize", 
        options=topics_list,
        default=default_topics
    )
    
    year_range = st.slider(
        "Select year range",
        min_value=int(historical_papers['year'].min()),
        max_value=int(historical_papers['year'].max()),
        value=(int(historical_papers['year'].min()), int(historical_papers['year'].max()))
    )
    
    # Filter data based on selections
    filtered_data = historical_papers[
        (historical_papers['topic'].isin(selected_topics)) &
        (historical_papers['year'] >= year_range[0]) &
        (historical_papers['year'] <= year_range[1])
    ]
    
    if not filtered_data.empty:
        # Create line chart of research trends
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for topic in selected_topics:
            topic_data = filtered_data[filtered_data['topic'] == topic]
            ax.plot(topic_data['year'], topic_data['paper_count'], marker='o', linewidth=2, label=topic)
        
        ax.set_title('Research Topic Trends Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Format x-axis to show all years
        ax.set_xticks(range(year_range[0], year_range[1]+1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add download button for the chart
        st.markdown(get_img_download_link(fig, f"research_trends_{uuid.uuid4()}.png", 
                                       "Download Trends Chart"), unsafe_allow_html=True)
        
        # Topic growth analysis
        st.subheader("Topic Growth Analysis")
        
        # Calculate growth metrics
        growth_data = []
        
        for topic in selected_topics:
            topic_data = filtered_data[filtered_data['topic'] == topic].sort_values('year')
            
            if len(topic_data) >= 2:
                start_count = topic_data.iloc[0]['paper_count']
                end_count = topic_data.iloc[-1]['paper_count']
                start_year = topic_data.iloc[0]['year']
                end_year = topic_data.iloc[-1]['year']
                
                # Calculate growth metrics
                absolute_growth = end_count - start_count
                growth_years = end_year - start_year
                
                # Avoid division by zero
                if start_count > 0 and growth_years > 0:
                    percentage_growth = (end_count / start_count - 1) * 100
                    cagr = ((end_count / start_count) ** (1 / growth_years) - 1) * 100
                else:
                    percentage_growth = 0
                    cagr = 0
                
                growth_data.append({
                    "Topic": topic,
                    "Start Count": start_count,
                    "End Count": end_count,
                    "Absolute Growth": absolute_growth,
                    "Percentage Growth": round(percentage_growth, 2),
                    "CAGR (%)": round(cagr, 2)
                })
        
        # Create growth dataframe and display
        growth_df = pd.DataFrame(growth_data)
        st.dataframe(growth_df)
        
        # Visualization of relative growth
        fig2, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Topic', y='CAGR (%)', data=growth_df, palette='viridis', ax=ax)
        ax.set_title('Compound Annual Growth Rate by Topic')
        ax.set_xlabel('Topic')
        ax.set_ylabel('CAGR (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Add download button for growth chart
        st.markdown(get_img_download_link(fig2, f"topic_growth_{uuid.uuid4()}.png", 
                                       "Download Growth Chart"), unsafe_allow_html=True)
        
        # Topic correlation analysis
        st.subheader("Topic Correlation Analysis")
        st.write("This analysis shows how research topics tend to rise and fall together.")
        
        # Create pivot table for correlation analysis
        pivot_data = filtered_data.pivot_table(
            index='year', 
            columns='topic', 
            values='paper_count',
            fill_value=0
        )
        
        # Only calculate correlation if we have enough data
        if pivot_data.shape[0] > 1 and pivot_data.shape[1] > 1:
            # Calculate correlation matrix
            correlation_matrix = pivot_data.corr()
            
            # Create heatmap
            fig3, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            ax.set_title("Correlation Between Research Topics")
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Add download button for correlation matrix
            st.markdown(get_img_download_link(fig3, f"topic_correlation_{uuid.uuid4()}.png", 
                                           "Download Correlation Chart"), unsafe_allow_html=True)
        else:
            st.info("Need more data points to calculate meaningful correlations between topics.")
        
        # Topic forecasting
        st.subheader("Research Trend Forecasting")
        st.write("Simple linear forecasting of research topic trends for the next 3 years.")
        
        # Add forecasting years
        forecast_years = list(range(year_range[1] + 1, year_range[1] + 4))
        
        # Create forecasting visualization
        fig4, ax = plt.subplots(figsize=(12, 6))
        
        for topic in selected_topics:
            topic_data = filtered_data[filtered_data['topic'] == topic]
            
            # Fit a simple linear regression for forecasting
            x = topic_data['year'].values.reshape(-1, 1)
            y = topic_data['paper_count'].values
            
            if len(x) >= 2:  # Need at least 2 points for regression
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(x, y)
                
                # Plot actual data
                ax.plot(topic_data['year'], topic_data['paper_count'], marker='o', linewidth=2, label=f"{topic} (Actual)")
                
                # Generate and plot forecasted data
                forecast_x = np.array(forecast_years).reshape(-1, 1)
                forecast_y = model.predict(forecast_x)
                
                # Ensure forecast values are non-negative
                forecast_y = np.maximum(forecast_y, 0)
                
                # Plot forecasted data with dashed line
                ax.plot(forecast_years, forecast_y, linestyle='--', linewidth=2, label=f"{topic} (Forecast)")
        
        # Add vertical line to separate actual and forecasted data
        ax.axvline(x=year_range[1], color='gray', linestyle=':', alpha=0.7)
        ax.text(year_range[1] + 0.1, ax.get_ylim()[1] * 0.9, "Forecast →", va='center', ha='left', alpha=0.7)
        
        ax.set_title('Research Topic Trends and Forecasts')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize='small')
        
        # Format x-axis to show all years including forecast
        all_years = list(range(year_range[0], forecast_years[-1] + 1))
        ax.set_xticks(all_years)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Add download button for forecast chart
        st.markdown(get_img_download_link(fig4, f"topic_forecast_{uuid.uuid4()}.png", 
                                       "Download Forecast Chart"), unsafe_allow_html=True)
        
        # Download trend data
        st.markdown(get_csv_download_link(filtered_data, "research_trends.csv", 
                                     "Download Trend Data"), unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        
with tab5:
    st.header("About SciReX")
    st.write("""
    **SciReX** is an AI-powered research paper analysis tool developed to help researchers and academics with literature review and gap analysis in the field of Generative AI.

    ### Key Features:
    - **Paper Summarization**: Using fine-tuned BART model to create concise summaries
    - **Information Extraction**: Identifying methods, metrics, datasets and entities
    - **Topic Analysis**: Using BERTopic to identify research themes
    - **Gap Analysis**: Identifying under-explored research areas
    - **Paper Comparison**: Comparing multiple papers to identify similarities and differences
    - **Reference Analysis**: Extracting and analyzing citation networks
    - **Trend Analysis**: Tracking research trends over time with forecasting
    - **Claim Verification**: Check the factual accuracy of claims made in papers

    ### Technologies Used:
    - **PyTorch & Hugging Face** for model fine-tuning and inference
    - **BERTopic** for topic modeling
    - **spaCy** for named entity recognition
    - **NetworkX** for citation network analysis
    - **scikit-learn** for trend forecasting
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
st.sidebar.subheader("Try Sample Abstract")
sample_abstracts = {
    #"Diffusion Models": """Diffusion models are a class of generative models that have recently shown impressive results in image synthesis tasks. These models work by gradually adding Gaussian noise to training data and then learning to reverse this process to generate new data samples. In this paper, we present a novel approach to improving diffusion models through adaptive noise scheduling and guided generation. Our approach achieves state-of-the-art results on several image generation benchmarks, including CIFAR-10 and ImageNet. We evaluate our method using FID scores and human evaluation studies. Additionally, we explore applications in text-to-image generation and image editing tasks.""",

    #"Multimodal Learning": """Multimodal learning has become increasingly important as AI systems need to process and understand information from different modalities such as text, images, and audio. In this work, we introduce a new transformer-based architecture for multimodal representation learning that effectively aligns features across modalities. Our model employs a novel cross-attention mechanism and contrastive learning objective. We evaluate our approach on vision-language tasks including image-text retrieval and visual question answering, demonstrating significant improvements over existing methods. Our experiments show that our model achieves better generalization to unseen data and exhibits stronger zero-shot capabilities.""",
    
    # Add a paper with references for the new reference extraction feature
    "Paper with References": """In recent years, Transformer models (Vaswani et al., 2017) have revolutionized natural language processing, with models like BERT (Devlin et al., 2019) and GPT-3 (Brown et al., 2020) achieving remarkable performance across various tasks. Building on these advances, researchers have explored new directions like multimodal learning (Radford et al., 2021) and efficient training methods (Hoffmann et al., 2022). 
    
    Our work follows the diffusion model paradigm first introduced by Sohl-Dickstein et al. (2015) and later refined by Ho et al. (2020) and Nichol and Dhariwal (2021). We extend these approaches by incorporating elements from the score-based generative modeling framework of Song and Ermon (2019).
    
    We evaluate our approach using standard metrics like FID (Heusel et al., 2017) and Inception Score (Salimans et al., 2016), and compare against leading methods including GANs (Goodfellow et al., 2014), VAEs (Kingma and Welling, 2013), and other diffusion-based techniques (Song et al., 2021; Dhariwal and Nichol, 2021).
    """
}

selected_sample = st.sidebar.selectbox("Select a sample abstract", list(sample_abstracts.keys()))
if st.sidebar.button("Load Sample"):
    if 'paper_text' not in st.session_state:
        st.session_state['paper_text'] = sample_abstracts[selected_sample]
    
    # Make sure to update the text area in tab1
    if 'paper_abstract' not in st.session_state:
        st.session_state['paper_abstract'] = sample_abstracts[selected_sample]

    st.session_state['abstract_input'] = sample_abstracts[selected_sample]
    st.rerun()

# Add a section in the sidebar for requirements info
st.sidebar.markdown("---")
st.sidebar.subheader("PDF Support")
st.sidebar.info("""
    SciReX uses PyPDF2 and PyMuPDF (fitz) for PDF processing.
    
    Required packages:
    - PyPDF2
    - PyMuPDF
    - Pillow
    
    If you're using this locally, install with:
    ```
    pip install PyPDF2 PyMuPDF Pillow
    ```
""")
# Add this code after the "PDF Support" section in your sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("README")

# Expandable README content
with st.sidebar.expander("View README"):
    st.markdown("""
    # SciReX: AI-Powered Research Paper Reviewer

    ## Overview
    SciReX is an advanced tool for researchers and academics that leverages AI to analyze, summarize, and extract insights from scientific papers, with a focus on generative AI and machine learning research.

    ## Features
    - **Paper Analysis**: Summarize papers and extract key methods, metrics, and entities
    - **Research Gap Analysis**: Identify under-explored areas in research
    - **Multi-Paper Comparison**: Compare methods and findings across papers
    - **Citation Analysis**: Analyze reference networks and citation patterns
    - **Trend Analysis**: Track research topic evolution over time with forecasting
    - **Claim Verification**: Check the factual accuracy of claims made in papers
    
    ## Usage
    1. Upload a PDF or paste paper text in the Paper Analysis tab
    2. Use the Research Gaps tab to identify promising research directions
    3. Compare multiple papers with the Compare Papers feature
    4. Explore historical trends in the research field
    
    ## Requirements
    - Python 3.8+
    - PyTorch
    - Transformers
    - BERTopic
    - spaCy
    - PyPDF2 & PyMuPDF for PDF processing
    - Streamlit for the interface
    
    ## Installation
    ```
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```
    
    ## Citation
    If you use SciReX in your research, please cite:
    ```
    @software{scirex2025,
      author = {SciReX Team},
      title = {SciReX: AI-Powered Research Paper Reviewer},
      year = {2025},
      url = {https://github.com/scirex/scirex}
    }
    ```
    """)

# Add download button for README
readme_text = """# SciReX: AI-Powered Research Paper Reviewer

## Overview
SciReX is an advanced tool for researchers and academics that leverages AI to analyze, summarize, and extract insights from scientific papers, with a focus on generative AI and machine learning research.

## Features
- **Paper Analysis**: Summarize papers and extract key methods, metrics, and entities
- **Research Gap Analysis**: Identify under-explored areas in research
- **Multi-Paper Comparison**: Compare methods and findings across papers
- **Citation Analysis**: Analyze reference networks and citation patterns
- **Trend Analysis**: Track research topic evolution over time with forecasting
- **Claim Verification**: Check the factual accuracy of claims made in papers

## Usage
1. Upload a PDF or paste paper text in the Paper Analysis tab
2. Use the Research Gaps tab to identify promising research directions
3. Compare multiple papers with the Compare Papers feature
4. Explore historical trends in the research field

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- BERTopic
- spaCy
- PyPDF2 & PyMuPDF for PDF processing
- Streamlit for the interface

## Installation
-pip install -r requirements.txt
-python -m spacy download en_core_web_sm

## Citation
If you use SciReX in your research, please cite:
@software{scirex2025,
author = {SciReX Team},
title = {SciReX: AI-Powered Research Paper Reviewer},
year = {2025},
url = {https://github.com/scirex/scirex}
}"""
# Convert string to bytes for download
def get_text_download_link(text, filename, link_text):
    """Generate a download link for a text file"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

st.sidebar.markdown(get_text_download_link(readme_text, "README.md", "Download README.md"), unsafe_allow_html=True)
# Info about the project
st.sidebar.markdown("---")
st.sidebar.info("""
    **SciReX** helps researchers stay informed about the latest developments and research gaps in their field.

    This project is part of a larger initiative to accelerate scientific discovery through AI assistance.
    """)
st.sidebar.markdown("---")