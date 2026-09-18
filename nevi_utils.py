import streamlit as st
from typing import Dict, Any

def get_traditional_grade_color(grade: str) -> str:
    """Get color for traditional 3-tier dysplasia grade display"""
    color_map = {
        'Mild Dysplasia': '#28a745',        # Green
        'Moderate Dysplasia': '#ffc107',    # Yellow
        'Severe Dysplasia': '#dc3545',      # Red
        'Unknown': '#6c757d',               # Gray
        'Analysis Error': '#dc3545'         # Red
    }
    return color_map.get(grade, '#6c757d')

def get_mpath_grade_color(grade: str) -> str:
    """Get color for MPATH-Dx 2-tier grade display"""
    color_map = {
        'Low-Grade Dysplasia': '#28a745',   # Green
        'High-Grade Dysplasia': '#dc3545', # Red
        'Unknown': '#6c757d',               # Gray
        'Analysis Error': '#dc3545'         # Red
    }
    return color_map.get(grade, '#6c757d')

def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level display"""
    color_map = {
        'High': '#28a745',    # Green
        'Medium': '#ffc107',  # Yellow
        'Low': '#dc3545'      # Red
    }
    return color_map.get(confidence, '#6c757d')  # Default gray

def display_nevi_results(result: Dict[str, Any]):
    """Display nevi analysis results in a formatted way"""
    
    # Get grades and confidence
    traditional_grade = result.get('traditional_grade', 'Unknown')
    mpath_grade = result.get('mpath_grade', 'Unknown')
    confidence = result.get('confidence_level', 'Low')
    
    # Get colors
    traditional_color = get_traditional_grade_color(traditional_grade)
    mpath_color = get_mpath_grade_color(mpath_grade)
    confidence_color = get_confidence_color(confidence)
    
    # Main result display with both grading systems
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; border: 2px solid {traditional_color}; margin-bottom: 20px;'>
            <h2 style='color: {traditional_color}; margin-top: 0;'>🔬 Traditional Grade: {traditional_grade}</h2>
            <h3 style='color: {mpath_color}; margin: 10px 0;'>📊 MPATH-Dx Grade: {mpath_grade}</h3>
            <p style='color: {confidence_color}; font-size: 18px; margin-bottom: 0;'>
                <strong>Confidence Level: {confidence}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create two columns for features
    col1, col2 = st.columns(2)
    
    with col1:
        # Architectural features
        architectural_features = result.get('architectural_features', [])
        if architectural_features:
            st.subheader("🏗️ Architectural Features")
            for feature in architectural_features:
                if feature.strip():
                    st.write(f"• {feature}")
    
    with col2:
        # Cytological features
        cytological_features = result.get('cytological_features', [])
        if cytological_features:
            st.subheader("🧬 Cytological Features")
            for feature in cytological_features:
                if feature.strip():
                    st.write(f"• {feature}")
    
    # Additional observations
    additional_obs = result.get('additional_observations', '')
    if additional_obs and additional_obs.strip():
        st.subheader("📝 Additional Observations")
        st.write(additional_obs)
    
    # Clinical significance
    clinical_sig = result.get('clinical_significance', '')
    if clinical_sig and clinical_sig.strip():
        st.subheader("⚕️ Clinical Significance")
        st.write(clinical_sig)
    
    # Context information
    context_used = result.get('context_used', False)
    if context_used:
        st.success("✅ Analysis based on medical literature from nevi knowledge base")
    else:
        st.warning("⚠️ Limited context from nevi knowledge base")
    
    # Grading system explanation
    with st.expander("📚 Grading System Reference"):
        st.markdown("""
        ### Traditional 3-Tier System:
        - **Mild Dysplasia**: Minimal architectural disorder, mild nuclear atypia
        - **Moderate Dysplasia**: Moderate architectural disorder, moderate nuclear atypia  
        - **Severe Dysplasia**: Significant architectural disorder, marked nuclear atypia
        
        ### MPATH-Dx 2-Tier System:
        - **Low-Grade Dysplasia**: Minimal risk, corresponds to mild dysplasia
        - **High-Grade Dysplasia**: Higher clinical significance, corresponds to moderate-severe dysplasia
        """)
    
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
                Clinical decisions should always be made by qualified dermatopathologists and medical professionals.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def initialize_nevi_session_state():
    """Initialize Streamlit session state variables for nevi analysis"""
    
    # Nevi RAG system state
    if 'nevi_rag_system' not in st.session_state:
        st.session_state.nevi_rag_system = None
    
    if 'nevi_rag_initialized' not in st.session_state:
        st.session_state.nevi_rag_initialized = False
    
    # Nevi image upload state
    if 'nevi_uploaded_image' not in st.session_state:
        st.session_state.nevi_uploaded_image = None
    
    if 'nevi_image_uploaded' not in st.session_state:
        st.session_state.nevi_image_uploaded = False
    
    if 'nevi_image_data' not in st.session_state:
        st.session_state.nevi_image_data = None
    
    # Nevi analysis state
    if 'nevi_analysis_result' not in st.session_state:
        st.session_state.nevi_analysis_result = None
    
    if 'nevi_analysis_complete' not in st.session_state:
        st.session_state.nevi_analysis_complete = False

def create_nevi_download_report(result: Dict[str, Any], image_name: str) -> str:
    """Create a downloadable text report of the nevi analysis"""
    
    report = f"""
ATYPICAL MELANOCYTIC NEVI GRADING REPORT
======================================

Image: {image_name}
Analysis Date: {st.session_state.get('nevi_analysis_date', 'Not specified')}

PRIMARY ASSESSMENT
-----------------
Traditional 3-Tier Grade: {result.get('traditional_grade', 'Unknown')}
MPATH-Dx 2-Tier Grade: {result.get('mpath_grade', 'Unknown')}
Confidence Level: {result.get('confidence_level', 'Low')}

ARCHITECTURAL FEATURES
---------------------
"""
    
    architectural_features = result.get('architectural_features', [])
    for i, feature in enumerate(architectural_features, 1):
        if feature.strip():
            report += f"{i}. {feature}\n"
    
    report += f"""
CYTOLOGICAL FEATURES
-------------------
"""
    
    cytological_features = result.get('cytological_features', [])
    for i, feature in enumerate(cytological_features, 1):
        if feature.strip():
            report += f"{i}. {feature}\n"
    
    report += f"""
ADDITIONAL OBSERVATIONS
----------------------
{result.get('additional_observations', 'None specified')}

CLINICAL SIGNIFICANCE
--------------------
{result.get('clinical_significance', 'Not specified')}

FULL ANALYSIS
------------
{result.get('raw_analysis', 'No detailed analysis available')}

GRADING SYSTEMS REFERENCE
-------------------------
Traditional 3-Tier System:
- Mild Dysplasia: Minimal architectural disorder, mild nuclear atypia
- Moderate Dysplasia: Moderate architectural disorder, moderate nuclear atypia
- Severe Dysplasia: Significant architectural disorder, marked nuclear atypia

MPATH-Dx 2-Tier System:
- Low-Grade Dysplasia: Minimal risk, corresponds to mild dysplasia
- High-Grade Dysplasia: Higher clinical significance, corresponds to moderate-severe dysplasia

DISCLAIMER
----------
This analysis is for research and educational purposes only.
Clinical decisions should always be made by qualified dermatopathologists and medical professionals.
This system is not intended for clinical diagnosis.
"""
    
    return report