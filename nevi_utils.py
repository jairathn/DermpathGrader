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

def get_mpath_class_color(mpath_class: str) -> str:
    """Colour for an MPATH-Dx v2.0 class."""
    return {
        "0": "#6c757d",    # nondiagnostic, grey
        "I": "#28a745",    # low-grade atypia, green
        "II": "#fd7e14",   # high-grade atypia, orange
        "III": "#dc3545",  # melanoma pT1a, red
        "IV": "#a71d2a",   # melanoma >=pT1b, dark red
    }.get(str(mpath_class).strip(), "#6c757d")


def display_magnification_evidence(result: Dict[str, Any]):
    """What the model says it saw at each power.

    Worth showing rather than burying: a finding attributed to a
    magnification that cannot resolve it is the most legible way to spot
    a confabulated read.
    """
    evidence = result.get("magnification_evidence") or []
    if not evidence:
        return
    st.subheader("Evidence by magnification")
    for item in evidence:
        if isinstance(item, dict):
            st.write(f"**{item.get('magnification', '?')}** - "
                     f"{item.get('finding', '')}")


def display_nevi_results(result: Dict[str, Any]):
    """Render a melanocytic grading result.

    The MPATH-Dx v2.0 class is the headline because it is the study's
    primary label. The lesion category sits directly under it, because
    Class II covers both high-grade dysplasia and melanoma in situ and
    the class alone does not say which.
    """
    import mpath_dx

    mclass = str(result.get("mpath_dx_v2_class", "") or "")
    category = result.get("lesion_category", "unknown")
    grade = result.get("dysplasia_grade", "not_applicable")
    subtype = result.get("melanoma_subtype", "not_applicable")
    histology = result.get("melanoma_histologic_subtype", "not_applicable")
    breslow = result.get("breslow_estimate_mm")
    confidence = result.get("confidence_level", "Low")

    class_label = ""
    if mclass in mpath_dx.CLASS_DEFINITIONS:
        class_label = mpath_dx.CLASS_DEFINITIONS[mclass]["label"]

    if category == "melanoma":
        subline = "Melanoma"
        if subtype == "in_situ":
            subline += " in situ"
        elif subtype == "invasive":
            subline += " (invasive"
            subline += f", Breslow ~{breslow} mm)" if breslow else ")"
        if histology and histology != "not_applicable":
            subline += f" - {histology.replace('_', ' ')}"
    elif category == "dysplastic_nevus":
        subline = f"Dysplastic nevus, {str(grade).replace('_', ' ')} atypia"
    elif category == "benign_nevus_no_atypia":
        subline = "Benign nevus, no significant atypia"
    else:
        subline = str(category).replace("_", " ").capitalize()

    colour = get_mpath_class_color(mclass)
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px;
                    border: 2px solid {colour}; margin-bottom: 20px;'>
            <h2 style='color: {colour}; margin-top: 0;'>
                MPATH-Dx v2.0 Class {mclass} - {class_label}
            </h2>
            <h3 style='color: #444; margin: 10px 0;'>{subline}</h3>
            <p style='color: {get_confidence_color(confidence)};
                      font-size: 18px; margin-bottom: 0;'>
                <strong>Confidence: {confidence}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mclass and mpath_dx.is_valid_class(mclass):
        if mpath_dx.requires_reexcision(mclass):
            st.warning("Class II or above: re-excision indicated.")
        else:
            st.info("Class I: re-excision not indicated on this basis.")
        if mclass == "II":
            st.caption(
                "Class II covers high-grade dysplastic nevi and melanoma in "
                "situ alike. The lesion category above is what distinguishes "
                "them.")

    if category == "melanoma" and subtype == "invasive":
        details = []
        if result.get("ulceration_present") is not None:
            details.append("ulcerated" if result["ulceration_present"]
                           else "not ulcerated")
        if result.get("mitoses_per_mm2") is not None:
            details.append(f"{result['mitoses_per_mm2']} mitoses/mm2")
        if details:
            st.caption("Staging features: " + ", ".join(details))

    flags = result.get("consistency_flags") or []
    if flags:
        st.error(
            "This output contradicts itself, so the class should not be "
            "taken at face value:\n"
            + "\n".join(f"- {flag}" for flag in flags))

    col1, col2 = st.columns(2)
    with col1:
        features = result.get("architectural_features", [])
        if features:
            st.subheader("Architectural features")
            for feature in features:
                if str(feature).strip():
                    st.write(f"- {feature}")
    with col2:
        features = result.get("cytological_features", [])
        if features:
            st.subheader("Cytological features")
            for feature in features:
                if str(feature).strip():
                    st.write(f"- {feature}")

    display_magnification_evidence(result)

    rationale = result.get("grading_rationale", "")
    if str(rationale).strip():
        st.subheader("Grading rationale")
        st.write(rationale)

    significance = result.get("clinical_significance", "")
    if str(significance).strip():
        st.subheader("Clinical significance")
        st.write(significance)

    if result.get("context_used"):
        st.success("Analysis drew on the retrieved literature context.")
    else:
        st.warning("No literature context was retrieved.")

    with st.expander("MPATH-Dx v2.0 reference"):
        st.markdown(
            "Version 2.0 replaced the five-class v1.0 schema with four "
            "classes and removed the standalone moderate-atypia category. "
            "Class I is low-grade (mild-to-moderate) atypia; Class II is "
            "high-grade (high-end moderate-to-severe) atypia and includes "
            "melanoma in situ; Class III is invasive melanoma under "
            "0.8 mm; Class IV is 0.8 mm or greater.")
        for cls in mpath_dx.CLASSES:
            definition = mpath_dx.CLASS_DEFINITIONS[cls]
            st.write(f"**Class {cls} - {definition['label']}**: "
                     f"{definition['definition']}")
        st.caption(mpath_dx.CITATION)

    with st.expander("Raw model output"):
        st.text_area("JSON", result.get("raw_analysis", ""), height=300,
                     disabled=True)

    st.markdown(
        """
        <div style='background-color: #fff3cd; border: 1px solid #ffeaa7;
                    border-radius: 5px; padding: 15px; margin-top: 20px;'>
            <h4 style='color: #856404; margin-top: 0;'>Medical disclaimer</h4>
            <p style='color: #856404; margin-bottom: 0;'>
                Research and educational use only. Clinical decisions are
                made by qualified dermatopathologists.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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
    
    report += """
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