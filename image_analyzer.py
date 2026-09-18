import os
import sys
import base64
import hashlib as _hashlib
import time
from typing import Dict, Any
import anthropic
from anthropic import Anthropic
import streamlit as st

class ImageAnalyzer:
    """Image analyzer using Claude for multimodal pathology image analysis"""
    
    def __init__(self, rag_system):
        """Initialize the image analyzer with RAG system"""
        self.rag_system = rag_system
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
            st.error(f"Failed to setup Claude API: {str(e)}")
            raise
    
    def create_analysis_prompt(self, context: str) -> str:
        """Create a comprehensive prompt for SCC differentiation grading"""
        
        prompt = f"""
You are an expert dermatopathologist analyzing a histopathology image of cutaneous squamous cell carcinoma (CSCC) for differentiation grading. 

Based on the following medical literature context from authoritative sources:

{context}

ANALYZE the provided histopathology image for these features:
- Keratinization: keratin pearls, horn cysts, intracellular keratin (present = better differentiated)
- Cellular atypia: pleomorphism, nuclear abnormalities (high = poorly differentiated)  
- Tumor architecture: organized vs infiltrative/disorganized
- Squamous maturation: gradient from basal to keratinized cells
- Mitotic activity: low, moderate, or high

GRADING CRITERIA:
- Well Differentiated: abundant keratinization, keratin pearls present, minimal atypia, organized
- Moderately Differentiated: some keratinization, moderate atypia
- Poorly Differentiated: minimal/absent keratinization, high atypia, infiltrative, basaloid

YOU MUST PROVIDE A GRADE. Respond with a JSON object in this exact format:
```json
{{
  "primary_grade": "Well Differentiated" or "Moderately Differentiated" or "Poorly Differentiated",
  "confidence_level": "High" or "Medium" or "Low",
  "keratinization_present": true or false,
  "atypia_level": "minimal" or "moderate" or "high",
  "key_features": ["feature 1", "feature 2", "feature 3"],
  "additional_observations": "any other findings"
}}
```

RULES:
1. You MUST choose one of the three grades - never leave it blank or unknown
2. If keratinization is absent/minimal AND atypia is high → grade is "Poorly Differentiated"
3. If abundant keratinization AND minimal atypia → grade is "Well Differentiated"
4. When uncertain, base grade on the predominant features observed
5. For mixed grades, report the worst (least differentiated) component
"""
        return prompt
    
    def analyze_image(self, image_data: str, media_type: str = "image/jpeg",
                      case_logger=None) -> Dict[str, Any]:
        """Analyze the pathology image using Claude with RAG context"""
        try:
            # Get relevant context from RAG system
            context = self.rag_system.get_grading_criteria()
            
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
                temperature=0.1,  # Low temperature for consistent medical analysis
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
                _img_bytes  = base64.b64decode(image_data)
                _img_sha    = _hashlib.sha256(_img_bytes).hexdigest()
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
                        'primary_grade', 'confidence_level',
                        'keratinization_present', 'atypia_level',
                        'key_features', 'additional_observations',
                    ]},
                )
                if hasattr(self.rag_system, '_retrieval_log'):
                    _rlog = self.rag_system._retrieval_log
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
            st.error(f"Image analysis failed: {str(e)}")
            raise
    
    def parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's response into structured format with multiple fallbacks"""
        import re
        import json
        
        result = {
            'primary_grade': 'Unknown',
            'confidence_level': 'Medium',
            'key_features': [],
            'additional_observations': '',
            'raw_analysis': response_text,
            'keratinization_present': None,
            'atypia_level': None
        }
        
        # LAYER 1: Try to parse JSON from response
        self._last_parse_strategy = 'direct_json'   # logging
        self._last_parse_errors   = []              # logging
        json_parsed = self._try_parse_json(response_text, result)
        
        # LAYER 2: If JSON failed or grade still unknown, try regex patterns
        if result['primary_grade'] == 'Unknown':
            self._try_regex_parsing(response_text, result)
            self._last_parse_strategy = 'regex'     # logging
        
        # LAYER 3: If still unknown, infer grade from feature keywords in text
        if result['primary_grade'] == 'Unknown':
            self._infer_grade_from_features(response_text, result)
            self._last_parse_strategy = 'keyword_inference'   # logging
        
        return result
    
    def _try_parse_json(self, response_text: str, result: Dict[str, Any]) -> bool:
        """Try to extract and parse JSON from response"""
        import re
        import json
        
        # Look for JSON in code blocks or raw JSON
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code block
            r'```\s*(\{.*?\})\s*```',       # JSON in generic code block
            r'(\{[^{}]*"primary_grade"[^{}]*\})',  # Raw JSON with primary_grade
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    json_str = match.group(1)
                    data = json.loads(json_str)
                    
                    # Extract grade
                    if 'primary_grade' in data:
                        grade = data['primary_grade']
                        result['primary_grade'] = self._normalize_grade(grade)
                    
                    # Extract other fields
                    if 'confidence_level' in data:
                        result['confidence_level'] = data['confidence_level'].capitalize()
                    if 'key_features' in data:
                        result['key_features'] = data['key_features'] if isinstance(data['key_features'], list) else [data['key_features']]
                    if 'additional_observations' in data:
                        result['additional_observations'] = data['additional_observations']
                    if 'keratinization_present' in data:
                        result['keratinization_present'] = data['keratinization_present']
                    if 'atypia_level' in data:
                        result['atypia_level'] = data['atypia_level']
                    
                    return True
                except json.JSONDecodeError:
                    continue
        return False
    
    def _try_regex_parsing(self, response_text: str, result: Dict[str, Any]) -> None:
        """Try multiple regex patterns to extract grade"""
        import re
        
        # Multiple patterns to catch different formats
        grade_patterns = [
            r'["\']?primary_grade["\']?\s*[:\s]+["\']?(Well Differentiated|Moderately Differentiated|Poorly Differentiated)["\']?',
            r'Primary Grade[:\s]+\**\s*(Well Differentiated|Moderately Differentiated|Poorly Differentiated)',
            r'\*\*Primary Grade\*\*[:\s]+(Well Differentiated|Moderately Differentiated|Poorly Differentiated)',
            r'Grade[:\s]+\**\s*(Well|Moderately|Poorly)\s+Differentiated',
            r'(Well|Moderately|Poorly)\s+Differentiated\s+(SCC|squamous cell carcinoma)',
            r'this\s+(is|appears|represents)\s+(?:a\s+)?(well|moderately|poorly)\s+differentiated',
            r'classified\s+as\s+(well|moderately|poorly)\s+differentiated',
            r'diagnosis[:\s]+(well|moderately|poorly)\s+differentiated',
        ]
        
        for pattern in grade_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                grade_text = match.group(1) if match.lastindex else match.group(0)
                result['primary_grade'] = self._normalize_grade(grade_text)
                if result['primary_grade'] != 'Unknown':
                    break
        
        # Extract confidence if not already set
        conf_patterns = [
            r'["\']?confidence_level["\']?\s*[:\s]+["\']?(High|Medium|Low)["\']?',
            r'Confidence[:\s]+\**(High|Medium|Low)',
        ]
        for pattern in conf_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result['confidence_level'] = match.group(1).capitalize()
                break
    
    def _infer_grade_from_features(self, response_text: str, result: Dict[str, Any]) -> None:
        """Infer grade from described features as last resort"""
        text_lower = response_text.lower()
        
        # Count indicators for each grade
        poorly_indicators = [
            'no keratinization', 'absent keratinization', 'minimal keratinization',
            'lack of keratinization', 'without keratinization', 'no keratin pearls',
            'basaloid', 'high atypia', 'marked atypia', 'severe atypia',
            'high-grade atypia', 'infiltrative', 'disorganized', 'anaplastic',
            'lack of maturation', 'no maturation', 'undifferentiated',
            'high mitotic', 'numerous mitoses', 'pleomorphic'
        ]
        
        well_indicators = [
            'abundant keratinization', 'keratin pearls present', 'prominent keratinization',
            'well-formed keratin', 'horn pearls', 'mature squamous',
            'minimal atypia', 'low atypia', 'organized architecture',
            'orderly maturation', 'low mitotic', 'rare mitoses',
            'well differentiated features', 'good differentiation'
        ]
        
        moderate_indicators = [
            'some keratinization', 'focal keratinization', 'partial keratinization',
            'moderate atypia', 'intermediate', 'moderately differentiated features'
        ]
        
        poorly_score = sum(1 for ind in poorly_indicators if ind in text_lower)
        well_score = sum(1 for ind in well_indicators if ind in text_lower)
        moderate_score = sum(1 for ind in moderate_indicators if ind in text_lower)
        
        # Determine grade based on scores
        if poorly_score > well_score and poorly_score > moderate_score:
            result['primary_grade'] = 'Poorly Differentiated'
            result['confidence_level'] = 'Medium'
        elif well_score > poorly_score and well_score > moderate_score:
            result['primary_grade'] = 'Well Differentiated'
            result['confidence_level'] = 'Medium'
        elif moderate_score > 0 or (poorly_score > 0 and well_score > 0):
            result['primary_grade'] = 'Moderately Differentiated'
            result['confidence_level'] = 'Medium'
        else:
            # Default to moderately if we can't determine
            result['primary_grade'] = 'Moderately Differentiated'
            result['confidence_level'] = 'Low'
    
    def _normalize_grade(self, grade_text: str) -> str:
        """Normalize grade text to standard format"""
        if not grade_text:
            return 'Unknown'
        grade_lower = grade_text.lower()
        if 'poorly' in grade_lower:
            return 'Poorly Differentiated'
        elif 'moderately' in grade_lower:
            return 'Moderately Differentiated'
        elif 'well' in grade_lower:
            return 'Well Differentiated'
        return 'Unknown'
