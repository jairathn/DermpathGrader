"""Streamlit UI for the dermatopathology grading system (protocol v2.0).

    streamlit run app.py --server.port 5000

Each pathway now takes four images per case - whole slide, 4x, 10x and
40x - because a single low-power field does not carry enough resolution
to grade cytology, and a single high-power field loses the architecture.
All four are required before the Analyze button enables: a case graded on
three images is not comparable to one graded on four.

Whole-slide .svs files cannot be uploaded here. The vision API does not
read them. Run extract_tiles.py to produce the four JPEGs first; the
dermatopathologist arm reads the .svs directly in their own viewer.

The UI and run_tests.py share image_utils, so a case graded here and the
same case graded in batch send byte-identical images.
"""

from __future__ import annotations

import datetime
import io
import json
import pathlib
import zipfile

import streamlit as st

import claude_transport
import config
import image_utils
import report
from grading_logger import CaseLogger
from utils import initialize_session_state, display_results
from nevi_utils import initialize_nevi_session_state, display_nevi_results


st.set_page_config(page_title="Dermatopathology Grading System",
                   page_icon="*", layout="wide",
                   initial_sidebar_state="expanded")


def _manifest_session_id() -> str:
    try:
        return json.loads(
            pathlib.Path(config.RUN_MANIFEST).read_text())["session_id"]
    except Exception:
        return "ui-no-manifest"


def _next_rep(pathway: str, case_id: str) -> int:
    return CaseLogger(pathway, case_id, 1, "probe").next_free_replicate()


def store_ready(pathway: str) -> bool:
    """Is this pathway's vector store present on disk?

    Checked before offering the Initialize button, so a missing store
    produces an explanation rather than a traceback in the researcher's
    face.
    """
    directory = pathlib.Path(config.CHROMA_DIR[pathway])
    return directory.exists() and any(directory.iterdir())


def collect_logs() -> list[pathlib.Path]:
    root = pathlib.Path(config.LOG_ROOT)
    return sorted(root.glob("*/*.json")) if root.exists() else []


def logs_zip() -> bytes:
    """Every case log written in this session, as one archive.

    Managed hosts give the app no persistent disk: a restart wipes
    analysis_logs/. Downloading is what makes a hosted session's work
    survivable, so this sits in the sidebar rather than being buried.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in collect_logs():
            archive.write(path, arcname=str(path))
    return buffer.getvalue()


def magnification_uploaders(key_prefix: str) -> dict[str, bytes]:
    """One uploader per magnification. Returns magnification -> bytes."""
    sources: dict[str, bytes] = {}
    st.caption(
        f"All {len(config.MAGNIFICATIONS)} magnifications are required. "
        f"Upload exported tiles, not .svs.")
    for mag in config.MAGNIFICATIONS:
        caption = config.MAGNIFICATION_CAPTIONS[mag]
        uploaded = st.file_uploader(
            f"{mag} - {caption.split(' - ')[0]}",
            type=[s.lstrip(".") for s in config.ACCEPTED_IMAGE_SUFFIXES],
            key=f"{key_prefix}_{mag}",
            help=caption)
        if uploaded is not None:
            error = image_utils.validate_upload(uploaded)
            if error:
                st.error(error)
            else:
                sources[mag] = uploaded.getvalue()
                st.session_state[f"{key_prefix}_{mag}_name"] = uploaded.name
    return sources


def show_prepared(prepared: list) -> None:
    """Thumbnail strip plus what was actually sent."""
    columns = st.columns(len(prepared))
    for column, image in zip(columns, prepared, strict=True):
        with column:
            st.image(f"data:image/jpeg;base64,{image.b64}",
                     caption=image.magnification, use_container_width=True)
            st.caption(
                f"{image.sent_dimensions_px[0]}x"
                f"{image.sent_dimensions_px[1]} px, "
                f"q{image.compression_quality}, "
                f"{image.sent_bytes / 1e6:.2f} MB")
    total = sum(i.sent_bytes for i in prepared) / 1e6
    st.caption(f"Total payload {total:.2f} MB across {len(prepared)} images.")


def pathway_tab(pathway: str) -> None:
    """One grading tab. The two pathways stay separately wired on purpose."""
    is_cscc = pathway == "CSCC"
    prefix = "scc" if is_cscc else "nevi"
    state_key = "rag_system" if is_cscc else "nevi_rag_system"
    init_key = "rag_initialized" if is_cscc else "nevi_rag_initialized"

    if is_cscc:
        initialize_session_state()
    else:
        initialize_nevi_session_state()

    with st.sidebar:
        st.header(f"{pathway} system")
        if is_cscc:
            st.write("Grades: well / moderately / poorly differentiated")
        else:
            st.write("Grades: mild / moderate / severe dysplasia, "
                     "and melanoma")
            st.write("Also assigns an MPATH-Dx v2.0 class (0, I, II, III, IV)")
        st.caption(f"Model {config.MODEL_ID}, effort {config.EFFORT}, "
                   f"protocol v{config.PROTOCOL_VERSION}")

        ready_to_init = store_ready(pathway)
        if not ready_to_init:
            st.error(
                f"No vector store at `{config.CHROMA_DIR[pathway]}/`. "
                f"This pathway cannot run until it is built.")
            with st.expander("How to fix this"):
                st.markdown(
                    "Run `python build_vector_stores.py --check` to see "
                    "which source PDFs are present, then "
                    "`python build_vector_stores.py` to build, and commit "
                    "the store directory. The stores are small (the CSCC "
                    "one is about 1.3 MB) and belong in the repo, because "
                    "managed hosts have no persistent disk to build them "
                    "onto.")

        if st.button(f"Initialize {pathway} RAG system", key=f"{prefix}_init",
                     disabled=not ready_to_init):
            with st.spinner("Initializing..."):
                try:
                    if is_cscc:
                        from rag_system import RAGSystem
                        st.session_state[state_key] = RAGSystem()
                    else:
                        from nevi_rag_system import NeviRAGSystem
                        st.session_state[state_key] = NeviRAGSystem()
                    st.session_state[init_key] = True
                    st.success(f"{pathway} RAG system ready")
                except Exception as exc:
                    st.error(f"Failed to initialize: {exc}")

        st.write("Ready" if st.session_state.get(init_key)
                 else "Not initialized")

        logs = collect_logs()
        if logs:
            st.divider()
            st.caption(
                f"{len(logs)} case log(s) this session. Hosted deployments "
                f"have no persistent disk, so download before you finish.")
            st.download_button(
                "Download all case logs (.zip)",
                data=logs_zip(),
                file_name=(f"dermpathgrader_logs_"
                           f"{datetime.datetime.now():%Y%m%d_%H%M}.zip"),
                mime="application/zip",
                key=f"{prefix}_dl_all")

    left, right = st.columns([1, 1])

    with left:
        st.header(f"{pathway} case images")
        sources = magnification_uploaders(prefix)

        missing = [m for m in config.REQUIRED_MAGNIFICATIONS
                   if m not in sources]
        prepared = None
        if missing:
            st.info(f"Waiting for: {', '.join(missing)}")
        else:
            try:
                prepared = image_utils.prepare_case_images(sources)
                show_prepared(prepared)
                st.session_state[f"{prefix}_prepared"] = prepared
            except image_utils.ImagePreparationError as exc:
                st.error(str(exc))

        ready = prepared is not None and st.session_state.get(init_key)
        if st.button(f"Analyze {pathway} case", type="primary",
                     disabled=not ready, key=f"{prefix}_go"):
            with st.spinner(f"Grading {len(prepared)} images..."):
                try:
                    first = st.session_state.get(
                        f"{prefix}_{config.MAGNIFICATIONS[0]}_name",
                        "unknown")
                    case_id = "UI-" + pathlib.Path(first).stem.replace(" ", "_")
                    replicate = _next_rep(pathway, case_id)
                    logger = CaseLogger(pathway, case_id, replicate,
                                        _manifest_session_id())
                    logger.set_image_set(prepared,
                                         tile_selection_method="ui_upload")

                    rag = st.session_state[state_key]
                    rag.clear_retrieval_log()
                    if is_cscc:
                        from image_analyzer import ImageAnalyzer
                        analyzer = ImageAnalyzer(rag)
                    else:
                        from nevi_analyzer import NeviAnalyzer
                        analyzer = NeviAnalyzer(rag)

                    result = analyzer.analyze_images(prepared,
                                                     case_logger=logger)
                    path = logger.save()
                    st.session_state[f"{prefix}_result"] = result
                    st.session_state[f"{prefix}_log_path"] = str(path)
                except claude_transport.GradingRefused as exc:
                    path = logger.save()
                    st.session_state[f"{prefix}_result"] = None
                    st.session_state[f"{prefix}_log_path"] = str(path)
                    st.error(
                        "The model declined to grade this case"
                        + (f" (category: {exc.category})" if exc.category else "")
                        + ". No grade was produced; the attempt is logged.")
                except claude_transport.GradingTruncated as exc:
                    path = logger.save()
                    st.session_state[f"{prefix}_result"] = None
                    st.session_state[f"{prefix}_log_path"] = str(path)
                    st.error(f"Output was truncated: {exc}")
                except claude_transport.GradingTransportError as exc:
                    path = logger.save()
                    st.session_state[f"{prefix}_result"] = None
                    st.session_state[f"{prefix}_log_path"] = str(path)
                    st.error(f"Could not reach the API after retries: {exc}")
                except Exception as exc:
                    try:
                        if logger.record.get("failure") is None:
                            logger.set_failure(exc)
                        logger.save()
                    except Exception:
                        pass
                    st.session_state[f"{prefix}_result"] = None
                    st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

    with right:
        st.header("Result")
        result = st.session_state.get(f"{prefix}_result")
        if not result:
            if st.session_state.get(f"{prefix}_log_path"):
                st.warning("The last attempt produced no grade. Its log is "
                           "below.")
            else:
                st.info("No analysis yet.")
        else:
            if is_cscc:
                display_results(result)
            else:
                display_nevi_results(result)
            log_path = st.session_state.get(f"{prefix}_log_path")
            if log_path and pathlib.Path(log_path).exists():
                st.caption(f"Logged to {log_path}")
                log_text = pathlib.Path(log_path).read_text(encoding="utf-8")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        "Download case log (.json)",
                        data=log_text,
                        file_name=pathlib.Path(log_path).name,
                        mime="application/json",
                        key=f"{prefix}_dl_one")
                with col_b:
                    st.download_button(
                        "Download synoptic report (.md)",
                        data=report.render(json.loads(log_text)),
                        file_name=pathlib.Path(log_path).stem + ".md",
                        mime="text/markdown",
                        key=f"{prefix}_dl_report")


def main() -> None:
    st.title("Dermatopathology Grading System")
    st.caption(
        f"Protocol v{config.PROTOCOL_VERSION} - "
        f"{config.TARGET_N_TOTAL} cases "
        f"({config.TARGET_N['Nevus']} melanocytic, {config.TARGET_N['CSCC']} "
        f"CSCC), {config.N_PER_STRATUM} per stratum, "
        f"{len(config.MAGNIFICATIONS)} magnifications per case")

    cscc_tab, nevus_tab = st.tabs(
        ["CSCC differentiation", "Melanocytic lesions"])
    with cscc_tab:
        pathway_tab("CSCC")
    with nevus_tab:
        pathway_tab("Nevus")

    st.divider()
    st.caption(
        "Research and educational use only. Clinical decisions are made by "
        "qualified pathologists. This system is not for clinical diagnosis.")


if __name__ == "__main__":
    main()
