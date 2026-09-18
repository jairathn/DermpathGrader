import os
import sys
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
try:
    # Current home of the splitter. `langchain.text_splitter` is a
    # re-export of this same class, so chunking is unchanged; importing
    # it directly avoids pulling in the whole langchain metapackage.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older environments
    from langchain.text_splitter import RecursiveCharacterTextSplitter
# Must precede the chromadb import: some managed hosts ship a
# SQLite older than the 3.35 chromadb requires.
import sqlite_compat  # noqa: F401
import chromadb
import hashlib as _hashlib
from chromadb.config import Settings

# ── optional Streamlit ───────────────────────────────────────────────
# v1 imported streamlit unconditionally and called st.info/st.success at
# import time, so run_tests.py had to monkey-patch four st.* functions to
# stay quiet in batch. Messages now go to Streamlit only when a session is
# actually running, and to stdout otherwise.

def _notify(kind: str, message: str) -> None:
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            getattr(st, kind)(message)
            return
    except Exception:
        pass
    if kind in ("error", "warning"):
        import sys
        print(f"[{kind}] {message}", file=sys.stderr)


class RAGSystem:
    """RAG system for processing medical literature and providing context for SCC grading"""
    
    def __init__(self):
        """Initialize the RAG system"""
        self.client = None
        self.collection = None
        self.documents = []
        self.setup_chromadb()
        self.process_documents()
        self._retrieval_log = []   # list of dicts; populated by query_documents()

    # ── logging helpers ───────────────────────────────────────────────────────
    def get_retrieval_log(self) -> list:
        """Return a copy of every retrieved chunk record across all queries."""
        return list(self._retrieval_log)

    def clear_retrieval_log(self):
        """Reset before each new case so the log only covers one analysis."""
        self._retrieval_log = []

    def setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="scc_grading_literature",
                metadata={"description": "Medical literature for SCC differentiation grading"}
            )
        except Exception as e:
            _notify("error", f"Failed to setup ChromaDB: {str(e)}")
            raise
    
    def process_documents(self):
        """Process the uploaded PDF documents and create vector store"""
        try:
            # Check if documents are already processed
            if self.collection and self.collection.count() > 0:
                _notify("info", "Documents already processed in ChromaDB")
                return
                
            # Get the PDF files from the attached_assets directory
            pdf_files = [
                "attached_assets/Grading differentiation.pdf",
                "attached_assets/Grading SCC diff.pdf"
            ]
            
            all_texts = []
            all_metadatas = []
            all_ids = []
            doc_id = 0
            
            for pdf_file in pdf_files:
                if os.path.exists(pdf_file):
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    texts = text_splitter.split_documents(docs)
                    
                    for text_doc in texts:
                        all_texts.append(text_doc.page_content)
                        all_metadatas.append({
                            "source": pdf_file,
                            "page": text_doc.metadata.get("page", 0)
                        })
                        all_ids.append(f"doc_{doc_id}")
                        doc_id += 1
                        
                else:
                    _notify("warning", f"PDF file not found: {pdf_file}")
            
            if not all_texts:
                raise Exception("No PDF documents found to process")
            
            # Add documents to ChromaDB collection
            if self.collection:
                self.collection.add(
                    documents=all_texts,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
            
            _notify("success", f"Processed {len(all_texts)} document chunks successfully!")
            
        except Exception as e:
            _notify("error", f"Failed to process documents: {str(e)}")
            raise
    
    def query_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the document collection for relevant information"""
        try:
            if self.collection is None:
                raise Exception("Collection not initialized")
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    })

            # ── logging: record each retrieved chunk with its chunk_id ─────────
            chunk_ids = (results.get('ids') or [[]])[0]
            for i, d in enumerate(documents):
                self._retrieval_log.append({
                    'query':    query,
                    'rank':     i + 1,
                    'chunk_id': chunk_ids[i] if i < len(chunk_ids) else '',
                    'distance': d['distance'],
                    'source':   d['metadata'].get('source', ''),
                    'page':     d['metadata'].get('page', 0),
                    'text':     d['content'],
                })
            # ─────────────────────────────────────────────────────────────────

            return documents
            
        except Exception as e:
            _notify("error", f"Failed to query documents: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant context from the knowledge base"""
        try:
            documents = self.query_documents(query, n_results=5)
            
            context = {
                'documents': documents,
                'combined_text': ''
            }
            
            for doc in documents:
                context['combined_text'] += doc['content'] + "\n\n"
            
            return context
            
        except Exception as e:
            _notify("error", f"Failed to retrieve context: {str(e)}")
            return {'documents': [], 'combined_text': ''}
    
    def get_grading_criteria(self) -> str:
        """Get specific grading criteria from the knowledge base"""
        queries = [
            "well differentiated squamous cell carcinoma characteristics keratinization",
            "moderately differentiated squamous cell carcinoma features",
            "poorly differentiated squamous cell carcinoma criteria atypia",
            "keratin pearls horn cysts squamous cell carcinoma grading",
            "cellular atypia pleomorphism squamous cell carcinoma differentiation",
            "Broders grading system squamous cell carcinoma",
            "WHO grading squamous cell carcinoma differentiation"
        ]
        
        all_texts = []
        for query in queries:
            docs = self.query_documents(query, n_results=5)
            for doc in docs:
                all_texts.append(doc['content'])
        # Join with "\n\n" — no trailing separator, matching manifest and verifier rebuild
        return "\n\n".join(all_texts)
