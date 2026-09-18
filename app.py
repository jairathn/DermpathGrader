import streamlit as st
import os
import sys
import json
import pathlib
from PIL import Image
import base64
import io
from rag_system import RAGSystem
from image_analyzer import ImageAnalyzer
from utils import initialize_session_state, display_results
from nevi_rag_system import NeviRAGSystem
from nevi_analyzer import NeviAnalyzer
from nevi_utils import initialize_nevi_session_state, display_nevi_results
from image_utils import validate_and_process_image, get_last_image_meta
from grading_logger import CaseLogger, sha256_bytes


def _manifest_session_id() -> str:
    """Load the session_id from run_manifest.json, or return a UI fallback."""
    try:
        return json.loads(pathlib.Path("run_manifest.json").read_text())["session_id"]
    except Exception:
        return "ui-no-manifest"


def _next_rep(pathway: str, case_id: str) -> int:
    """Return the lowest replicate number whose log file does not yet exist."""
    probe = CaseLogger(pathway, case_id, 1, "probe")
    return probe.next_free_replicate()

# Page configuration
st.set_page_config(
    page_title="Pathology Image Grading System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def scc_tab():
    """SCC grading tab content"""
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for SCC system information
    with st.sidebar:
        st.header("SCC System Information")
        st.info("This system uses RAG-based analysis with medical literature to grade SCC differentiation in pathology images.")
        
        st.header("SCC Grading Categories")
        st.write("• **Well Differentiated**: High keratinization, minimal atypia")
        st.write("• **Moderately Differentiated**: Intermediate features")
        st.write("• **Poorly Differentiated**: Minimal keratinization, high atypia")
        
        st.header("SCC System Status")
        if st.button("Initialize SCC RAG System"):
            with st.spinner("Initializing SCC RAG system..."):
                try:
                    st.session_state.rag_system = RAGSystem()
                    st.session_state.rag_initialized = True
                    st.success("SCC RAG system initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize SCC RAG system: {str(e)}")
        
        if st.session_state.rag_initialized:
            st.success("✅ SCC RAG System Ready")
        else:
            st.warning("⚠️ SCC RAG System Not Initialized")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("SCC Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload SCC Pathology Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image of squamous cell carcinoma for differentiation grading",
            key="scc_uploader"
        )
        
        if uploaded_file is not None:
            # Validate and process image
            image, image_data, error_message = validate_and_process_image(uploaded_file)
            
            if error_message:
                st.error(error_message)
                st.session_state.image_uploaded = False
                st.session_state.uploaded_image = None
                st.session_state.image_data = None
            elif image is not None and image_data is not None:
                # Display uploaded image
                st.image(image, caption="Uploaded SCC Pathology Image", use_container_width=True)
                
                # Store image in session state
                st.session_state.uploaded_image = image
                st.session_state.image_uploaded = True
                st.session_state.image_data = image_data[0]  # Extract base64 data
                st.session_state.image_media_type = image_data[1]  # Extract media type
                # Capture source metadata for logging
                st.session_state.scc_source_filename = uploaded_file.name
                raw_bytes = uploaded_file.getvalue()
                st.session_state.scc_source_sha256 = sha256_bytes(raw_bytes)
                st.session_state.scc_log_path = None  # reset on new upload
            else:
                st.error("Failed to process image")
                st.session_state.image_uploaded = False
        
        # Analysis button
        if st.button("🔍 Analyze SCC Image", type="primary", disabled=not (st.session_state.image_uploaded and st.session_state.rag_initialized)):
            if not st.session_state.rag_initialized:
                st.error("Please initialize the SCC RAG system first.")
            elif not st.session_state.image_uploaded:
                st.error("Please upload an image first.")
            else:
                with st.spinner("Analyzing SCC pathology image..."):
                    try:
                        # Check if image data is available
                        if not st.session_state.image_data:
                            st.error("No image data available for analysis")
                            return
                        
                        # Initialize image analyzer
                        analyzer = ImageAnalyzer(st.session_state.rag_system)
                        
                        # Get media type or default to jpeg
                        media_type = getattr(st.session_state, 'image_media_type', 'image/jpeg')

                        # Build case logger
                        src_name = getattr(st.session_state, 'scc_source_filename', 'unknown.jpg')
                        case_id  = "UI-" + pathlib.Path(src_name).stem.replace(" ", "_")
                        rep      = _next_rep("CSCC", case_id)
                        logger   = CaseLogger("CSCC", case_id, rep, _manifest_session_id())

                        # Populate image metadata
                        img_b64 = st.session_state.image_data
                        sent_raw = base64.b64decode(img_b64)
                        img_meta = get_last_image_meta()
                        logger.set_image_meta(
                            source_registry="ui_upload",
                            source_filename=src_name,
                            source_sha256=getattr(st.session_state, 'scc_source_sha256', ''),
                            source_dimensions_px=img_meta.get("source_dimensions_px", [0, 0]),
                            sent_media_type=media_type,
                            sent_dimensions_px=img_meta.get("sent_dimensions_px", [0, 0]),
                            sent_bytes=len(sent_raw),
                            sent_sha256=sha256_bytes(sent_raw),
                            resize_applied=img_meta.get("resize_applied", False),
                            compression_quality=img_meta.get("compression_quality"),
                        )
                        
                        # Supply manifest context SHA so analyzer can set matches_manifest_context
                        try:
                            logger._manifest_context_sha256 = json.loads(
                                pathlib.Path("run_manifest.json").read_text()
                            )["pathways"]["CSCC"]["retrieval"]["context_block_sha256"]
                        except Exception:
                            logger._manifest_context_sha256 = ""

                        # Analyze the image — analyzer calls logger.set_retrieval() internally
                        result = analyzer.analyze_image(img_b64, media_type, case_logger=logger)

                        # Save log
                        log_fp = logger.save()
                        st.session_state.scc_log_path = str(log_fp)
                        
                        # Store results
                        st.session_state.analysis_result = result
                        st.session_state.analysis_complete = True
                        
                        st.success(f"SCC analysis completed — log saved to `{log_fp}`")
                        
                    except Exception as e:
                        st.error(f"SCC analysis failed: {str(e)}")
    
    with col2:
        st.header("SCC Analysis Results")
        
        if st.session_state.analysis_complete and st.session_state.analysis_result:
            display_results(st.session_state.analysis_result)
            log_path = getattr(st.session_state, 'scc_log_path', None)
            if log_path and pathlib.Path(log_path).exists():
                with st.expander("📄 Full JSON Report"):
                    st.json(json.loads(pathlib.Path(log_path).read_text()))
        else:
            st.info("Upload an SCC image and click 'Analyze SCC Image' to see results here.")

def nevi_tab():
    """Nevi grading tab content"""
    # Initialize nevi session state
    initialize_nevi_session_state()
    
    # Sidebar for nevi system information
    with st.sidebar:
        st.header("Nevi System Information")
        st.info("This system uses RAG-based analysis with medical literature to grade atypical melanocytic nevi (dysplastic nevi).")
        
        st.header("Nevi Grading Categories")
        st.write("**Traditional 3-Tier:**")
        st.write("• **Mild Dysplasia**: Minimal architectural disorder")
        st.write("• **Moderate Dysplasia**: Moderate architectural disorder")
        st.write("• **Severe Dysplasia**: Significant architectural disorder")
        st.write("")
        st.write("**MPATH-Dx 2-Tier:**")
        st.write("• **Low-Grade**: Minimal risk")
        st.write("• **High-Grade**: Higher clinical significance")
        
        st.header("Nevi System Status")
        if st.button("Initialize Nevi RAG System"):
            with st.spinner("Initializing Nevi RAG system..."):
                try:
                    st.session_state.nevi_rag_system = NeviRAGSystem()
                    st.session_state.nevi_rag_initialized = True
                    st.success("Nevi RAG system initialized successfully!")
                except Exception as e:
                    st.error(f"Failed to initialize Nevi RAG system: {str(e)}")
        
        if st.session_state.nevi_rag_initialized:
            st.success("✅ Nevi RAG System Ready")
        else:
            st.warning("⚠️ Nevi RAG System Not Initialized")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Nevi Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Nevi Pathology Image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload a histopathology image of atypical melanocytic nevi for dysplasia grading",
            key="nevi_uploader"
        )
        
        if uploaded_file is not None:
            # Validate and process image
            image, image_data, error_message = validate_and_process_image(uploaded_file)
            
            if error_message:
                st.error(error_message)
                st.session_state.nevi_image_uploaded = False
                st.session_state.nevi_uploaded_image = None
                st.session_state.nevi_image_data = None
            elif image is not None and image_data is not None:
                # Display uploaded image
                st.image(image, caption="Uploaded Nevi Pathology Image", use_container_width=True)
                
                # Store image in session state
                st.session_state.nevi_uploaded_image = image
                st.session_state.nevi_image_uploaded = True
                st.session_state.nevi_image_data = image_data[0]  # Extract base64 data
                st.session_state.nevi_image_media_type = image_data[1]  # Extract media type
                # Capture source metadata for logging
                st.session_state.nevi_source_filename = uploaded_file.name
                raw_bytes = uploaded_file.getvalue()
                st.session_state.nevi_source_sha256 = sha256_bytes(raw_bytes)
                st.session_state.nevi_log_path = None  # reset on new upload
            else:
                st.error("Failed to process image")
                st.session_state.nevi_image_uploaded = False
        
        # Analysis button
        if st.button("🔍 Analyze Nevi Image", type="primary", disabled=not (st.session_state.nevi_image_uploaded and st.session_state.nevi_rag_initialized)):
            if not st.session_state.nevi_rag_initialized:
                st.error("Please initialize the Nevi RAG system first.")
            elif not st.session_state.nevi_image_uploaded:
                st.error("Please upload an image first.")
            else:
                with st.spinner("Analyzing nevi pathology image..."):
                    try:
                        # Check if image data is available
                        if not st.session_state.nevi_image_data:
                            st.error("No image data available for analysis")
                            return
                        
                        # Initialize nevi analyzer
                        analyzer = NeviAnalyzer(st.session_state.nevi_rag_system)
                        
                        # Get media type or default to jpeg
                        media_type = getattr(st.session_state, 'nevi_image_media_type', 'image/jpeg')

                        # Build case logger
                        src_name = getattr(st.session_state, 'nevi_source_filename', 'unknown.jpg')
                        case_id  = "UI-" + pathlib.Path(src_name).stem.replace(" ", "_")
                        rep      = _next_rep("Nevus", case_id)
                        logger   = CaseLogger("Nevus", case_id, rep, _manifest_session_id())

                        # Populate image metadata
                        img_b64 = st.session_state.nevi_image_data
                        sent_raw = base64.b64decode(img_b64)
                        img_meta = get_last_image_meta()
                        logger.set_image_meta(
                            source_registry="ui_upload",
                            source_filename=src_name,
                            source_sha256=getattr(st.session_state, 'nevi_source_sha256', ''),
                            source_dimensions_px=img_meta.get("source_dimensions_px", [0, 0]),
                            sent_media_type=media_type,
                            sent_dimensions_px=img_meta.get("sent_dimensions_px", [0, 0]),
                            sent_bytes=len(sent_raw),
                            sent_sha256=sha256_bytes(sent_raw),
                            resize_applied=img_meta.get("resize_applied", False),
                            compression_quality=img_meta.get("compression_quality"),
                        )

                        # Supply manifest context SHA so analyzer can set matches_manifest_context
                        try:
                            logger._manifest_context_sha256 = json.loads(
                                pathlib.Path("run_manifest.json").read_text()
                            )["pathways"]["Nevus"]["retrieval"]["context_block_sha256"]
                        except Exception:
                            logger._manifest_context_sha256 = ""

                        # Analyze the image — analyzer calls logger.set_retrieval() internally
                        result = analyzer.analyze_image(img_b64, media_type, case_logger=logger)

                        # Save log
                        log_fp = logger.save()
                        st.session_state.nevi_log_path = str(log_fp)
                        
                        # Store results
                        st.session_state.nevi_analysis_result = result
                        st.session_state.nevi_analysis_complete = True
                        
                        st.success(f"Nevi analysis completed — log saved to `{log_fp}`")
                        
                    except Exception as e:
                        st.error(f"Nevi analysis failed: {str(e)}")
    
    with col2:
        st.header("Nevi Analysis Results")
        
        if st.session_state.nevi_analysis_complete and st.session_state.nevi_analysis_result:
            display_nevi_results(st.session_state.nevi_analysis_result)
            log_path = getattr(st.session_state, 'nevi_log_path', None)
            if log_path and pathlib.Path(log_path).exists():
                with st.expander("📄 Full JSON Report"):
                    st.json(json.loads(pathlib.Path(log_path).read_text()))
        else:
            st.info("Upload a nevi image and click 'Analyze Nevi Image' to see results here.")

def main():
    """Main application function with tab-based interface"""
    
    # App header
    st.title("🔬 Pathology Image Grading System")
    st.markdown("**Advanced AI-powered grading for SCC differentiation and atypical nevi**")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🩸 SCC Grading", "🔵 Nevi Grading"])
    
    with tab1:
        st.header("Squamous Cell Carcinoma Differentiation Grading")
        st.markdown("Grade SCC differentiation as well, moderately, or poorly differentiated based on established pathology criteria.")
        scc_tab()
    
    with tab2:
        st.header("Atypical Melanocytic Nevi (Dysplastic Nevi) Grading")
        st.markdown("Grade atypical nevi using both traditional 3-tier and MPATH-Dx 2-tier classification systems.")
        nevi_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>For Research and Educational Purposes Only</strong> ⚠️</p>
        <p>This system is not intended for clinical diagnosis. Always consult with qualified pathologists for medical decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
