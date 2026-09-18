import os
import sys
import re
import base64
import hashlib as _hashlib
import json
import time
from typing import Dict, Any
import anthropic
from anthropic import Anthropic
import streamlit as st

class NeviAnalyzer:
    """Image analyzer using Claude for multimodal atypical nevi analysis"""
    
    def __init__(self, nevi_rag_system):
        """Initialize the nevi analyzer with RAG system"""
        self.nevi_rag_system = nevi_rag_system
        self.setup_claude()
    
    def setup_claude(self):
        """Setup Claude API client"""
        try:
            # Get API key from environment
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_key:
                raise Exception("ANTHROPIC_API_KEY environment variable not set")
            
            # Initialize Claude client
            self.client = Anthropic(api_key=anthropic_key)
            self.model = "claude-opus-4-5-20251101"
            
        except Exception as e:
            st.error(f"Failed to setup Claude API for nevi analysis: {str(e)}")
            raise
    
    def create_analysis_prompt(self, context: str) -> str:
        """Create a short, focused prompt for fast atypical nevi grading"""
        
        prompt = f"""
Expert Dermatopathologist. Analyze the image using the provided context:
{context}

### GRADING CRITERIA:
- **Severe Dysplasia**: Grade as SEVERE if 3+ nuclear abnormalities OR ANY architectural high-risk feature (confluent hyperplasia, pagetoid spread, bridging nests, epidermal mitoses, nuclear size >=2x).
- **Moderate Dysplasia**: Nuclear size 1.5x, some pleomorphism/chromatin clumping, well-developed nests.
- **Mild Dysplasia**: Nuclear size 1x, minimal atypia, symmetric, normal maturation.

### RULES:
1. Nuclear pleomorphism is the KEY differentiator between mild and moderate.
2. 3+ nuclear abnormalities = SEVERE.
3. If features overlap moderate/severe, grade as SEVERE.

### OUTPUT JSON ONLY:
{{
  "traditional_grade": "Mild Dysplasia" | "Moderate Dysplasia" | "Severe Dysplasia",
  "mpath_grade": "Low-Grade Dysplasia" | "High-Grade Dysplasia",
  "confidence_level": "High" | "Medium" | "Low",
  "nuclear_abnormality_count": number,
  "architectural_features": ["feature 1", "feature 2"],
  "cytological_features": ["feature 1", "feature 2"],
  "grading_rationale": "short explanation",
  "clinical_significance": "short assessment"
}}
"""
        return prompt

    def analyze_image(self, image_data: str, media_type: str = "image/jpeg",
                      case_logger=None) -> Dict[str, Any]:
        """Analyze the nevi pathology image using Claude with RAG context"""
        try:
            # Get relevant context from nevi RAG system
            context = self.nevi_rag_system.get_grading_criteria()
            
            # Create analysis prompt
            prompt = self.create_analysis_prompt(context)
            
            # Prepare the message for Claude
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    }
                ]
            }
            
            # Call Claude API
            _t0 = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.1,
                messages=[message]
            )
            _latency_ms = int((time.time() - _t0) * 1000)

            # Parse the response
            analysis_text = response.content[0].text
            
            # Extract structured information from the response
            result = self.parse_analysis_response(analysis_text)
            result['raw_analysis'] = analysis_text
            result['context_used'] = len(context.split()) > 0

            # ── case logging ──────────────────────────────────────────────────
            if case_logger is not None:
                _img_bytes = base64.b64decode(image_data)
                _img_sha   = _hashlib.sha256(_img_bytes).hexdigest()
                case_logger.set_request(
                    model=self.model, temperature=0.1, max_tokens=1500,
                    system_text="", user_text=prompt,
                    image_sha256=_img_sha, sent_media_type=media_type,
                )
                case_logger.set_response(
                    api_request_id=getattr(response, 'id', ''),
                    model_returned=getattr(response, 'model', ''),
                    stop_reason=getattr(response, 'stop_reason', '') or '',
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=_latency_ms,
                    raw_text=analysis_text,   # byte-for-byte, before any parsing
                )
                _strategy = getattr(self, '_last_parse_strategy', 'direct_json')
                case_logger.set_parsing(
                    strategy_used=_strategy,
                    fallback_invoked=(_strategy != 'direct_json'),
                    parse_errors=getattr(self, '_last_parse_errors', []),
                    parsed={k: result.get(k) for k in [
                        'traditional_grade', 'mpath_grade', 'confidence_level',
                        'nuclear_abnormality_count', 'architectural_features',
                        'cytological_features', 'grading_rationale',
                        'clinical_significance',
                    ]},
                )
                if hasattr(self.nevi_rag_system, '_retrieval_log'):
                    _rlog = self.nevi_rag_system._retrieval_log
                    _ctx_sha = _hashlib.sha256(context.encode()).hexdigest()
                    case_logger.set_retrieval(
                        subqueries_issued=list(dict.fromkeys(r['query'] for r in _rlog)),
                        retrieved_chunk_ids_in_order=[r['chunk_id'] for r in _rlog],
                        context_block_sha256=_ctx_sha,
                        manifest_context_sha256=getattr(
                            case_logger, '_manifest_context_sha256', ''),
                    )
            # ─────────────────────────────────────────────────────────────────

            return result
            
        except Exception as e:
            st.error(f"Nevi image analysis failed: {str(e)}")
            raise
    
    def parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's JSON response into structured format"""
        self._last_parse_strategy = 'direct_json'   # logging
        self._last_parse_errors   = []              # logging
        try:
            # Try to find JSON block if Claude adds preamble
            json_str = response_text
            if "{" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
            
            data = json.loads(json_str)
            
            return {
                'traditional_grade': data.get('traditional_grade', 'Moderate Dysplasia'),
                'mpath_grade': data.get('mpath_grade', 'High-Grade Dysplasia'),
                'confidence_level': data.get('confidence_level', 'Medium'),
                'nuclear_abnormality_count': data.get('nuclear_abnormality_count', 0),
                'architectural_features': data.get('architectural_features', []),
                'cytological_features': data.get('cytological_features', []),
                'additional_observations': data.get('grading_rationale', ''),
                'clinical_significance': data.get('clinical_significance', ''),
                'raw_analysis': response_text
            }
            
        except Exception as e:
            # Fallback to simple keyword search if JSON parsing fails
            self._last_parse_strategy = 'keyword_fallback'   # logging
            self._last_parse_errors   = [str(e)]             # logging
            result = {
                'traditional_grade': 'Moderate Dysplasia',
                'mpath_grade': 'High-Grade Dysplasia',
                'confidence_level': 'Low',
                'architectural_features': ["Parsing error"],
                'cytological_features': ["Parsing error"],
                'additional_observations': f'Error parsing JSON: {str(e)}',
                'clinical_significance': 'Manual review required',
                'raw_analysis': response_text
            }
            
            if 'Severe' in response_text: result['traditional_grade'] = 'Severe Dysplasia'
            elif 'Mild' in response_text: result['traditional_grade'] = 'Mild Dysplasia'
            
            if 'Low-Grade' in response_text: result['mpath_grade'] = 'Low-Grade Dysplasia'
            
            return result
