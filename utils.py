import streamlit as st
from typing import Dict, Any

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # RAG system state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    
    # Image upload state
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    if 'image_uploaded' not in st.session_state:
        st.session_state.image_uploaded = False
    
    if 'image_data' not in st.session_state:
        st.session_state.image_data = None
    
    # Analysis state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level display"""
    color_map = {
        'High': '#28a745',    # Green
        'Medium': '#ffc107',  # Yellow
        'Low': '#dc3545'      # Red
    }
    return color_map.get(confidence, '#6c757d')  # Default gray

def get_grade_color(grade: str) -> str:
    """Get color for differentiation grade display"""
    color_map = {
        'Well Differentiated': '#28a745',        # Green
        'Moderately Differentiated': '#ffc107',  # Yellow
        'Poorly Differentiated': '#dc3545',      # Red
        'Unknown': '#6c757d',                     # Gray
        'Analysis Error': '#dc3545'              # Red
    }
    return color_map.get(grade, '#6c757d')

def display_results(result: Dict[str, Any]):
    """Display analysis results in a formatted way"""
    
    # Primary grade with color coding
    grade = result.get('primary_grade', 'Unknown')
    confidence = result.get('confidence_level', 'Low')
    
    grade_color = get_grade_color(grade)
    confidence_color = get_confidence_color(confidence)
    
    # Main result display
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; border: 2px solid {grade_color}; margin-bottom: 20px;'>
            <h2 style='color: {grade_color}; margin-top: 0;'>🔬 Primary Grade: {grade}</h2>
            <p style='color: {confidence_color}; font-size: 18px; margin-bottom: 0;'>
                <strong>Confidence Level: {confidence}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Key supporting features
    key_features = result.get('key_features', [])
    if key_features:
        st.subheader("🔍 Key Supporting Features")
        for feature in key_features:
            if feature.strip():
                st.write(f"• {feature}")
    
    # Additional observations
    additional_obs = result.get('additional_observations', '')
    if additional_obs and additional_obs.strip():
        st.subheader("📝 Additional Observations")
        st.write(additional_obs)
    
    # Context information
    context_used = result.get('context_used', False)
    if context_used:
        st.success("✅ Analysis based on medical literature from knowledge base")
    else:
        st.warning("⚠️ Limited context from knowledge base")
    
    # Expandable section for full analysis
    with st.expander("📄 View Full Analysis Report"):
        raw_analysis = result.get('raw_analysis', 'No detailed analysis available')
        st.text_area("Complete Analysis", raw_analysis, height=300, disabled=True)
    
    # Medical disclaimer
    st.markdown(
        """
        <div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin-top: 20px;'>
            <h4 style='color: #856404; margin-top: 0;'>⚠️ Medical Disclaimer</h4>
            <p style='color: #856404; margin-bottom: 0;'>
                This analysis is for research and educational purposes only. 
                Clinical decisions should always be made by qualified pathologists and medical professionals.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def format_medical_text(text: str) -> str:
    """Format medical text for better readability"""
    
    # Replace common medical abbreviations with full forms for clarity
    replacements = {
        'CSCC': 'Cutaneous Squamous Cell Carcinoma',
        'SCC': 'Squamous Cell Carcinoma',
        'WHO': 'World Health Organization',
        'NCCN': 'National Comprehensive Cancer Network'
    }
    
    formatted_text = text
    for abbrev, full_form in replacements.items():
        formatted_text = formatted_text.replace(abbrev, full_form)
    
    return formatted_text

def validate_image_upload(uploaded_file) -> bool:
    """Validate uploaded image file"""
    
    if uploaded_file is None:
        return False
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("Image file too large. Please upload an image smaller than 10MB.")
        return False
    
    # Check file type
    allowed_types = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_types:
        st.error(f"Unsupported file type. Please upload: {', '.join(allowed_types)}")
        return False
    
    return True

def display_system_info():
    """Display system information and status"""
    
    st.markdown(
        """
        ### System Components
        - **RAG System**: LangChain + ChromaDB for medical literature retrieval
        - **Image Analysis**: Claude 4 Opus multimodal AI
        - **Knowledge Base**: Medical literature on SCC differentiation grading
        - **Interface**: Streamlit web application
        
        ### Supported Image Formats
        - PNG, JPEG, JPG, TIFF, BMP
        - Maximum file size: 10MB
        
        ### Analysis Process
        1. Image upload and validation
        2. Retrieval of relevant medical literature
        3. Multimodal analysis using Claude 4 Opus
        4. Structured grading output with confidence assessment
        """
    )

def create_download_report(result: Dict[str, Any], image_name: str) -> str:
    """Create a downloadable text report of the analysis"""
    
    report = f"""
SQUAMOUS CELL CARCINOMA DIFFERENTIATION GRADING REPORT
=====================================================

Image: {image_name}
Analysis Date: {st.session_state.get('analysis_date', 'Not specified')}

PRIMARY ASSESSMENT
-----------------
Grade: {result.get('primary_grade', 'Unknown')}
Confidence Level: {result.get('confidence_level', 'Low')}

KEY SUPPORTING FEATURES
----------------------
"""
    
    key_features = result.get('key_features', [])
    for i, feature in enumerate(key_features, 1):
        if feature.strip():
            report += f"{i}. {feature}\n"
    
    report += f"""
ADDITIONAL OBSERVATIONS
----------------------
{result.get('additional_observations', 'None specified')}

FULL ANALYSIS
------------
{result.get('raw_analysis', 'No detailed analysis available')}

DISCLAIMER
----------
This analysis is for research and educational purposes only.
Clinical decisions should always be made by qualified pathologists and medical professionals.
This system is not intended for clinical diagnosis.
"""
    
    return report
