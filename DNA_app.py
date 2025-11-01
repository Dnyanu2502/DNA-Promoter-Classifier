import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time
import random

# Set page configuration
st.set_page_config(page_title="DNA Analyzer", page_icon="🧬", layout="wide")

class DNAPromoterPredictor:
    def __init__(self):
        self.model = None
        self.nucleotides = 'ATCG'
    
    def load_model(self):
        """Try to load pre-trained model"""
        try:
            self.model = joblib.load('dna_promoter_model.joblib')
            return True
        except:
            return False
    
    def extract_features(self, sequence):
        """Feature extraction for DNA sequences"""
        sequence = sequence.upper()
        features = []
        
        # Basic sequence features (6 features that match the trained model)
        features.append(len(sequence))
        
        # Nucleotide frequencies
        for nt in self.nucleotides:
            features.append(sequence.count(nt) / len(sequence))
        
        # GC content
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        features.append(gc_content)
        
        return features
    
    def predict(self, sequences):
        """Make predictions using heuristic method"""
        return self._get_heuristic_results(sequences)
    
    def get_sequence_properties(self, sequence):
        """Get comprehensive sequence properties"""
        sequence = sequence.upper()
        return {
            'length': len(sequence),
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence) * 100,
            'a_content': (sequence.count('A') / len(sequence)) * 100,
            't_content': (sequence.count('T') / len(sequence)) * 100,
            'c_content': (sequence.count('C') / len(sequence)) * 100,
            'g_content': (sequence.count('G') / len(sequence)) * 100,
            'tata_count': sequence.count('TATA'),
            'at_ratio': (sequence.count('A') + sequence.count('T')) / len(sequence)
        }
    
    def _get_heuristic_results(self, sequences):
        """Advanced heuristic predictions"""
        results = []
        for seq in sequences:
            props = self.get_sequence_properties(seq)
            
            # Smart promoter scoring
            promoter_score = 0
            
            # Strong promoter indicators
            if props['tata_count'] > 0:
                promoter_score += 0.4
            
            # GC content in optimal promoter range
            if 40 <= props['gc_content'] <= 60:
                promoter_score += 0.3
            
            # Length consideration
            if 40 <= len(seq) <= 80:
                promoter_score += 0.2
            
            # AT-rich regions
            if props['at_ratio'] > 0.6:
                promoter_score += 0.1
            
            # Normalize
            promoter_score = min(0.95, promoter_score)
            promoter_score += random.uniform(-0.05, 0.05)
            promoter_score = max(0.05, promoter_score)
            
            # Determine prediction
            is_promoter = promoter_score >= 0.5
            confidence = 0.4 + (abs(promoter_score - 0.5) * 1.2)
            
            results.append({
                'sequence': seq,
                'prediction': 'Promoter' if is_promoter else 'Non-Promoter',
                'confidence': min(confidence, 0.95),
                'promoter_probability': promoter_score,
                'properties': props
            })
        return results

def parse_fasta(file):
    """Parse FASTA file format"""
    sequences = []
    current_seq = ""
    for line in file:
        line = line.decode().strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        else:
            current_seq += line
    if current_seq:
        sequences.append(current_seq)
    return sequences

def generate_random_dna(length=50):
    """Generate random DNA sequence"""
    return ''.join(random.choice('ATCG') for _ in range(length))

def analyze_sequence_properties(sequence):
    """Analyze sequence properties for display"""
    seq = sequence.upper()
    return {
        'length': len(seq),
        'gc_content': (seq.count('G') + seq.count('C')) / len(seq) * 100,
        'a_content': (seq.count('A') / len(seq)) * 100,
        't_content': (seq.count('T') / len(seq)) * 100,
        'c_content': (seq.count('C') / len(seq)) * 100,
        'g_content': (seq.count('G') / len(seq)) * 100,
        'at_ratio': (seq.count('A') + seq.count('T')) / len(seq),
        'tata_count': seq.count('TATA')
    }

# Enhanced CSS with animated logos for each tab
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .stApp {
        background: #0f172a;
        font-family: 'Arial', sans-serif;
        color: #e2e8f0;
    }
    .prediction-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #e2e8f0;
        animation: fadeInScale 0.6s ease-out;
        border: 1px solid #334155;
    }
    .prediction-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .promoter-card {
        border-left-color: #10b981;
        background: #1e293b;
    }
    .nonpromoter-card {
        border-left-color: #ef4444;
        background: #1e293b;
    }
    .feature-box {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #334155;
        color: #e2e8f0;
        animation: slideInUp 0.5s ease-out;
    }
    .stButton>button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Animation Types */
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes gentleBounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-5px);
        }
        60% {
            transform: translateY(-3px);
        }
    }
    
    @keyframes smoothRotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    @keyframes dnaPulse {
        0% {
            transform: scale(1) rotate(0deg);
        }
        50% {
            transform: scale(1.1) rotate(180deg);
        }
        100% {
            transform: scale(1) rotate(360deg);
        }
    }
    
    @keyframes microscopeMove {
        0% {
            transform: translateX(0) scale(1);
        }
        25% {
            transform: translateX(-3px) scale(1.05);
        }
        50% {
            transform: translateX(0) scale(1);
        }
        75% {
            transform: translateX(3px) scale(1.05);
        }
        100% {
            transform: translateX(0) scale(1);
        }
    }
    
    @keyframes chartGrow {
        0% {
            transform: scaleY(0.8);
        }
        50% {
            transform: scaleY(1.1);
        }
        100% {
            transform: scaleY(0.8);
        }
    }
    
    @keyframes infoSpin {
        0% {
            transform: rotate(0deg) scale(1);
        }
        50% {
            transform: rotate(180deg) scale(1.1);
        }
        100% {
            transform: rotate(360deg) scale(1);
        }
    }
    
    @keyframes gentlePulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.03);
        }
        100% {
            transform: scale(1);
        }
    }

    /* Animated Logos for Each Tab */
    .dna-logo {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: dnaPulse 4s ease-in-out infinite;
        color: #10b981;
    }
    
    .microscope-logo {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: microscopeMove 3s ease-in-out infinite;
        color: #3b82f6;
    }
    
    .chart-logo {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: chartGrow 2s ease-in-out infinite;
        color: #f59e0b;
    }
    
    .info-logo {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: infoSpin 5s linear infinite;
        color: #8b5cf6;
    }
    
    .success-animation {
        text-align: center;
        font-size: 3rem;
        animation: gentleBounce 2s ease-in-out;
        color: #10b981;
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
        animation: slideInUp 0.8s ease-out;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #cbd5e1;
    }
    
    .tab-content {
        background: #1e293b;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #334155;
        animation: fadeInScale 0.7s ease-out;
    }
    
    .metric-card {
        background: #1e293b;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        border: 1px solid #334155;
        animation: slideInUp 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        animation: gentlePulse 2s ease-in-out infinite;
    }
    
    .progress-container {
        height: 8px;
        background: #334155;
        border-radius: 4px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        animation: progressGrow 1.5s ease-out;
    }
    
    @keyframes progressGrow {
        from {
            width: 0%;
        }
        to {
            width: var(--progress-width);
        }
    }
    
    .promoter-progress {
        background: #10b981;
    }
    
    .nonpromoter-progress {
        background: #ef4444;
    }
    
    /* Text readability enhancements */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    p, span, div {
        color: #e2e8f0 !important;
    }
    
    .stRadio > label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    .stTextArea textarea {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    .stTextInput input {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    .stSelectbox select {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe th {
        background: #334155 !important;
        color: #f8fafc !important;
    }
    
    .dataframe td {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    /* Tab-specific header animations */
    .tab-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor
predictor = DNAPromoterPredictor()
model_loaded = predictor.load_model()

# Initialize session state
if 'example_seq' not in st.session_state:
    st.session_state.example_seq = ""

# Main App with clean animations
st.markdown("""
<div class="title-container">
    <div class="dna-logo">🧬</div>
    <div class="main-title">DNA Sequence Analyzer</div>
    <div class="subtitle">Professional DNA Sequence Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 Promoter Analysis", 
    "🔬 Sequence Tools", 
    "📊 Batch Processing", 
    "ℹ️ About"
])

with tab1:
    # Animated DNA logo for Promoter Analysis tab
    st.markdown('<div class="dna-logo">🧬</div>', unsafe_allow_html=True)
    st.markdown("### DNA Promoter Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        input_method = st.radio("Select Input Method:", ["Paste Sequence", "Upload FASTA File"])
        
        sequences = []
        
        if input_method == "Paste Sequence":
            sequences_input = st.text_area(
                "Enter DNA Sequences (one per line):", 
                height=150,
                value=st.session_state.example_seq,
                placeholder="TATAATgccgGCGCGccatg\nATGCATGCATGCATGCATGC"
            )
            if sequences_input:
                sequences = [seq.strip() for seq in sequences_input.split('\n') if seq.strip()]
        else:
            uploaded_file = st.file_uploader("Upload FASTA File:", type=['fasta', 'fa', 'txt'])
            if uploaded_file:
                sequences = parse_fasta(uploaded_file)
                st.success(f"📁 {len(sequences)} sequences found")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Quick Examples")
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if st.button("🧬 Promoter Sequence", use_container_width=True):
            st.session_state.example_seq = "TATAATgccgGCGCGccatgTATAATgccgGCGCGccatg"
            st.rerun()
        
        if st.button("🧬 Gene Sequence", use_container_width=True):
            st.session_state.example_seq = "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
            st.rerun()
        
        st.markdown("### System Info")
        st.info("Using Advanced Heuristic Analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Analyze Sequences", type="primary", use_container_width=True) and sequences:
        with st.spinner("Analyzing DNA sequences..."):
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent + 1)
            
            results = predictor.predict(sequences)
        
        # Success animation
        st.markdown('<div class="success-animation">✓</div>', unsafe_allow_html=True)
        st.success("### Analysis Complete!")
        
        # Results summary
        promoter_count = sum(1 for r in results if r['prediction'] == 'Promoter')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Sequences", len(results))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Promoters Found", promoter_count)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Non-Promoters", len(results) - promoter_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results with clean animations
        for result in results:
            props = result['properties']
            progress_style = f"width: {result['confidence']*100}%"
            
            if result['prediction'] == 'Promoter':
                st.markdown(f"""
                <div class="prediction-card promoter-card">
                    <h3>🎯 PROMOTER DETECTED</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                    <p><strong>Sequence:</strong> {result['sequence'][:50]}...</p>
                    <p><strong>Length:</strong> {props['length']} bp | <strong>GC Content:</strong> {props['gc_content']:.1f}%</p>
                    <p><strong>TATA Boxes:</strong> {props['tata_count']}</p>
                    <div class="progress-container">
                        <div class="progress-bar promoter-progress" style="{progress_style}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card nonpromoter-card">
                    <h3>📝 NON-PROMOTER SEQUENCE</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                    <p><strong>Sequence:</strong> {result['sequence'][:50]}...</p>
                    <p><strong>Length:</strong> {props['length']} bp | <strong>GC Content:</strong> {props['gc_content']:.1f}%</p>
                    <p><strong>TATA Boxes:</strong> {props['tata_count']}</p>
                    <div class="progress-container">
                        <div class="progress-bar nonpromoter-progress" style="{progress_style}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    # Animated Microscope logo for Sequence Tools tab
    st.markdown('<div class="microscope-logo">🔬</div>', unsafe_allow_html=True)
    st.markdown("### Sequence Analysis Tools")
    
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    seq_input = st.text_area("Enter DNA sequence for analysis:", height=100)
    
    if seq_input:
        props = analyze_sequence_properties(seq_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sequence Properties")
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.metric("Sequence Length", f"{props['length']} bp")
            st.metric("GC Content", f"{props['gc_content']:.1f}%")
            st.metric("AT Ratio", f"{props['at_ratio']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Nucleotide Composition")
            nucleotides = {
                'A': props['a_content'],
                'T': props['t_content'], 
                'C': props['c_content'],
                'G': props['g_content']
            }
            fig = px.pie(values=list(nucleotides.values()), names=list(nucleotides.keys()),
                        title="Nucleotide Distribution",
                        color_discrete_sequence=['#3b82f6', '#ef4444', '#10b981', '#f59e0b'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0'
            )
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Animated Chart logo for Batch Processing tab
    st.markdown('<div class="chart-logo">📊</div>', unsafe_allow_html=True)
    st.markdown("### Batch Sequence Processing")
    
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    batch_input = st.text_area("Enter multiple sequences (one per line):", height=200)
    
    if st.button("Process Batch", use_container_width=True) and batch_input:
        sequences = [seq.strip() for seq in batch_input.split('\n') if seq.strip()]
        
        results = []
        for seq in sequences:
            props = analyze_sequence_properties(seq)
            
            promoter_score = 0
            if props['tata_count'] > 0: promoter_score += 1
            if 40 <= props['gc_content'] <= 60: promoter_score += 1
            if 40 <= len(seq) <= 80: promoter_score += 1
            
            results.append({
                'Sequence': seq[:30] + '...' if len(seq) > 30 else seq,
                'Length': props['length'],
                'GC%': f"{props['gc_content']:.1f}%",
                'TATA_Boxes': props['tata_count'],
                'Promoter_Potential': f"{promoter_score}/3"
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### Batch Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Sequences", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_gc = df['GC%'].str.rstrip('%').astype(float).mean()
            st.metric("Average GC%", f"{avg_gc:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            high_potential = len(df[df['Promoter_Potential'] >= '2/3'])
            st.metric("High Potential", high_potential)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    # Animated Info logo for About tab
    st.markdown('<div class="info-logo">ℹ️</div>', unsafe_allow_html=True)
    st.markdown("### About DNA Sequence Analyzer")
    
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("""
    **Project Overview:**
    This platform provides advanced DNA sequence analysis with promoter detection capabilities.

    **Features:**
    - DNA promoter sequence detection
    - Sequence property analysis
    - Batch sequence processing
    - GC content calculation

    **Technical Stack:**
    - Python with Streamlit
    - Advanced heuristic algorithms
    - Plotly for visualization

    **Developed By:**
    - Dnyaneshwari Bankar (Bioinformatics)
    - Shivani Navandar (Bioinformatics)

    **Guided By:**
    Dr. Kushagra Kashyap
    Assistant Professor (Bioinformatics)
    Department of Life Sciences
    DES Pune University
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>DNA Sequence Analyzer | Department of Life Sciences, DES Pune University</p>", unsafe_allow_html=True)