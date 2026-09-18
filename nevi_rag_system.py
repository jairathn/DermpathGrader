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


class NeviRAGSystem:
    """RAG system for processing medical literature and providing context for nevi grading"""
    
    def __init__(self):
        """Initialize the nevi RAG system"""
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
        """Setup ChromaDB client and collection for nevi"""
        try:
            # Initialize ChromaDB client with separate path for nevi
            self.client = chromadb.PersistentClient(path="./chroma_db_nevi")
            
            # Create or get collection for nevi literature
            self.collection = self.client.get_or_create_collection(
                name="nevi_grading_literature",
                metadata={"description": "Medical literature for atypical nevi and melanocytic lesion grading"}
            )
        except Exception as e:
            _notify("error", f"Failed to setup ChromaDB for nevi: {str(e)}")
            raise
    
    def process_documents(self):
        """Process the uploaded PDF documents for nevi grading"""
        try:
            # Check if documents are already processed
            if self.collection and self.collection.count() > 0:
                _notify("info", "Nevi documents already processed in ChromaDB")
                return
                
            # Get the PDF files for nevi grading
            pdf_files = [
                "attached_assets/Nevi 2001_1749902667336.pdf",
                "attached_assets/Nevi 2025_1749902667337.pdf", 
                "attached_assets/Nevi MPath 2_1749902667338.pdf",
                "attached_assets/Nevi MPath_1749902667339.pdf"
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
                        all_ids.append(f"nevi_doc_{doc_id}")
                        doc_id += 1
                        
                else:
                    _notify("warning", f"Nevi PDF file not found: {pdf_file}")
            
            if not all_texts:
                raise Exception("No nevi PDF documents found to process")
            
            # Add documents to ChromaDB collection
            if self.collection:
                self.collection.add(
                    documents=all_texts,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
            
            _notify("success", f"Processed {len(all_texts)} nevi document chunks successfully!")
            
        except Exception as e:
            _notify("error", f"Failed to process nevi documents: {str(e)}")
            raise
    
    def query_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the nevi document collection for relevant information"""
        try:
            if self.collection is None:
                raise Exception("Nevi collection not initialized")
            
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
            _notify("error", f"Failed to query nevi documents: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant context from the nevi knowledge base"""
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
            _notify("error", f"Failed to retrieve nevi context: {str(e)}")
            return {'documents': [], 'combined_text': ''}
    
    def get_grading_criteria(self) -> str:
        """Get specific grading criteria for atypical nevi from the knowledge base"""
        queries = [
            "mild dysplasia atypical melanocytic nevi characteristics",
            "moderate dysplasia atypical melanocytic nevi features",
            "severe dysplasia atypical melanocytic nevi criteria",
            "low grade dysplasia melanocytic lesions",
            "high grade dysplasia melanocytic lesions", 
            "nuclear atypia melanocytic nevi grading",
            "architectural disorder melanocytic nevi",
            "junctional melanocytic hyperplasia",
            "MPATH-Dx classification melanocytic lesions",
            "dysplastic nevus histologic criteria",
            "melanocytic atypia grading system",
            "WHO classification dysplastic nevi"
        ]
        
        all_texts = []
        for query in queries:
            docs = self.query_documents(query, n_results=5)
            for doc in docs:
                all_texts.append(doc['content'])
        # Join with "\n\n" — no trailing separator, matching manifest and verifier rebuild
        return "\n\n".join(all_texts)