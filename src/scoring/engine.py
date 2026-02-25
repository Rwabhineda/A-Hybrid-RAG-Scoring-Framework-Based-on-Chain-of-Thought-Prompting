# -*- coding: utf-8 -*-
"""
Batch Scoring Engine for ASEAN Legal Clause Evaluation
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
import chromadb
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


LEGAL_KEYWORDS = {
    'obligation': ['shall', 'must', 'required', 'obliged', 'duty', 'responsible', 'commit', 'undertake', 'ensure', 'guarantee'],
    'precision': ['within', 'days', 'months', 'years', 'before', 'after', 'specific', 'detailed', 'method', 'procedure', 'target'],
    'delegation': ['authority', 'body', 'institution', 'committee', 'organization', 'secretariat', 'council', 'commission', 'party', 'parties']
}


SCORING_GUIDE = """
You are a trained ASEAN legal clause evaluator. Please rate the following clause on three dimensions: Obligation, Precision, and Delegation. For each dimension, the score must be one of: 0.0, 0.25, 0.5, 0.75, or 1.0.

Strictly follow the stepwise reasoning and criteria below for your scoring. Do NOT score by intuition or general impression.

Obligation (the strength of legal or institutional commitment imposed by the clause)
Stepwise criteria:
1. Does the clause contain any binding or committal language?
   - No → Score 0.0
   - Yes → Step 2
2. Does it only use recommendatory language, or is it a political commitment?
   - Yes → Score 0.25
   - No → Step 3
3. Is the obligation conditional, or is the obligated actor not the party itself?
   - Yes → Score 0.5
   - No → Step 4
4. Does the clause state that non-compliance will lead to legal/institutional consequences (e.g., sanctions, penalties, loss of benefits, legal liability)?
   - No → Score 0.75
   - Yes → Score 1.0

Precision (the extent to which the clause is specific and concrete regarding actions and responsible parties)
Stepwise criteria:
1. Does it contain any action content?
   - No → Score 0.0
   - Yes → Step 2
2. Does it contain any concrete, executable action?
   - No → Score 0.25
   - Yes → Step 3
3. Does it only specify actors and actions, lacking all concrete details (e.g., frequency, timeframe, methods, targets)?
   - Yes → Score 0.5
   - No → Step 4
4. Does it lack some, but not all, such details (i.e., missing one or two among frequency, methods, targets)?
   - Yes → Score 0.75
   - No → Score 1.0

Delegation (whether the clause delegates real adjudicatory, supervisory, executive, or substantial decision-making power to a third party)
Stepwise criteria:
1a. Does it mention any concrete institution or party?
   - No → Score 0.0
   - Yes → 1b
1b. Is any concrete institution or party given support functions (coordination, technical support, information, advice, etc.) or authoritative functions (dispute settlement, execution, sanctions, supervision, binding interpretation, or final, non-appealable decisions)?
   - No → Score 0.0
   - Yes → Step 2
2. Is the empowered entity one of the parties (including subordinate bodies)?
   - Yes → Score 0.25
   - No → Step 3
3. Does the institution hold "decisive control" over treaty implementation (key approval, veto, procedural design, resource allocation, certification, etc.)?
   - No → Score 0.5
   - Yes → Step 4
4. Is the institution granted authoritative powers (as above) that are legally binding on the parties?
   - No → Score 0.75
   - Yes → Step 5
5. Are those authoritative powers subject to prerequisites (e.g., consensus, party consent, procedural trigger, application)?
   - No → Score 0.75
   - Yes → Score 1.0

Please score strictly according to the above criteria and the clause text only.
"""


def extract_key_terms(text: str) -> List[str]:
    text_lower = text.lower()
    found_terms = []
    for category, terms in LEGAL_KEYWORDS.items():
        for term in terms:
            if term in text_lower:
                found_terms.append(term)
    institutions = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b', text)
    found_terms.extend(institutions[:3])
    return list(set(found_terms))[:10]


def build_prompt(clause_text: str, similar_examples: List[Dict], use_cot_guide: bool = True) -> str:
    clause_keywords = extract_key_terms(clause_text)
    example_section = ""
    for i, item in enumerate(similar_examples[:3]):
        metadata = item['metadata']
        conf_info = []
        for dim in ['obligation', 'precision', 'delegation']:
            score = metadata.get(dim, 'N/A')
            conf = metadata.get(f'confidence_{dim}', 'N/A')
            conf_info.append(f"{dim.capitalize()}: {score} (confidence: {conf})")
        distance_info = f"(similarity: {1-item.get('distance', 0):.2f})" if 'distance' in item else ""
        example_section += f"""Example {i+1} {distance_info}:
Clause: {item['document']}
Key terms identified: {', '.join(extract_key_terms(item['document'])[:5])}
{chr(10).join(conf_info)}
Explanation: {metadata.get('explanation_text', 'N/A')}
---
"""
    base_prompt = f"""
CRITICAL INSTRUCTIONS:
1. You MUST follow the stepwise criteria EXACTLY - evaluate each step explicitly
2. Pay special attention to these key terms in the clause: {', '.join(clause_keywords)}
3. Each dimension is INDEPENDENT - do not let one score influence another
4. If uncertain between two scores, provide detailed reasoning and choose the more conservative (lower) score
5. Your reasoning MUST explicitly reference the specific steps in the criteria

Here are some HIGH-CONFIDENCE examples with similar characteristics:
{example_section}

Now, analyze the following clause step by step:

Clause: {clause_text}

IMPORTANT: For each dimension below, you must:
- Explicitly state which step you are evaluating
- Quote the relevant part of the clause
- Explain why you move to the next step or stop
- State the final score clearly

Obligation:
[Follow steps 1-4 explicitly, showing your reasoning at each step]

Precision:
[Follow steps 1-4 explicitly, showing your reasoning at each step]

Delegation:
[Follow steps 1a, 1b, 2-5 explicitly, showing your reasoning at each step]

Explanation:
Based on the above step-by-step analysis, provide a brief summary of your scoring rationale. Focus on the key factors that determined each score.

FINAL SCORES (must be exactly one of: 0.0, 0.25, 0.5, 0.75, or 1.0):
{{"obligation": [score], "precision": [score], "delegation": [score]}}
"""
    if use_cot_guide:
        return f"{SCORING_GUIDE}\n{base_prompt}"
    return base_prompt.strip()


def extract_scores(output_str: str) -> Dict[str, Optional[float]]:
    json_patterns = [
        r'\{[^}]*"obligation"[^}]+\}',
        r'FINAL SCORES[^{]*(\{[^}]+\})',
        r'"obligation":\s*\[?\s*([\d.]+)\s*\]?[^}]+\}'
    ]
    for pattern in json_patterns:
        match = re.search(pattern, output_str, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                json_str = match.group(0) if '{' in match.group(0) else match.group(1)
                json_str = json_str.replace('[', '').replace(']', '')
                scores_raw = json.loads(json_str)
                return {k: float(v) for k, v in scores_raw.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    scores = {}
    for dim in ["obligation", "precision", "delegation"]:
        patterns = [
            rf"{dim}.*?(?:final\s+)?score[:\s]*(0(?:\.0)?|0\.25|0\.5|0\.75|1(?:\.0)?)",
            rf"{dim}.*?→\s*Score\s+(0(?:\.0)?|0\.25|0\.5|0\.75|1(?:\.0)?)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output_str, re.IGNORECASE | re.DOTALL)
            if matches:
                scores[dim] = float(matches[-1])
                break
        else:
            scores[dim] = None
    return scores


def normalize_score(s: Optional[float]) -> float:
    if s is None:
        return 0.0
    allowed = [0.0, 0.25, 0.5, 0.75, 1.0]
    return min(allowed, key=lambda x: abs(x-s))


def is_exception(scores: Dict[str, float], output: str = "") -> bool:
    if not scores or any(v is None for v in scores.values()):
        return True
    values = [scores.get("obligation"), scores.get("precision"), scores.get("delegation")]
    if all(v == 1.0 for v in values) or all(v == 0.0 for v in values):
        return True
    if scores.get("obligation") == 0.0 and scores.get("delegation") >= 0.75:
        return True
    if scores.get("precision") <= 0.25 and scores.get("delegation") >= 0.75:
        return True
    if all(v == 0.5 for v in values):
        return True
    return False


class ClauseCache:
    def __init__(self, cache_dir: str, similarity_threshold: float = 1.0, model_tag: str = "default"):
        self.cache_dir = Path(cache_dir) / model_tag
        self.similarity_threshold = similarity_threshold
        self.memory_cache = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, clause_text: str) -> str:
        normalized = re.sub(r'\s+', ' ', clause_text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _calculate_similarity(self, text1: str, text2: str, model) -> float:
        try:
            vec1 = model.encode([text1], convert_to_numpy=True)[0]
            vec2 = model.encode([text2], convert_to_numpy=True)[0]
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def get_cached_result(self, clause_text: str, model) -> Optional[Dict]:
        cache_key = self._get_cache_key(clause_text)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.memory_cache[cache_key] = cached_data
                return cached_data
            except Exception:
                pass
        return None
    
    def save_result(self, clause_text: str, scores: Dict, llm_output: str):
        cache_key = self._get_cache_key(clause_text)
        cache_data = {
            'text': clause_text,
            'scores': scores,
            'llm_output': llm_output,
            'timestamp': time.time()
        }
        self.memory_cache[cache_key] = cache_data
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass


class BatchScorer:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.collection = None
        self.session = None
        self.filter_model = None

        # Load configuration
        self.openai_model = config["models"]["openai"]["model"]
        self.openai_api_key = config["models"]["openai"]["api_key"]
        self.openai_api_url = config["models"]["openai"]["api_url"]
        self.temperature = config["models"]["openai"].get("temperature", 0.0)
        self.embedding_model = config["models"]["embedding"]["model"]
        self.chroma_dir = config["vector_db"]["chroma_dir"]
        self.collection_name = config["vector_db"]["collection_name"]
        self.top_k = config["retrieval"]["top_k"]
        self.max_concurrent = config["runtime"]["max_concurrent"]
        self.batch_size = config["runtime"]["batch_size"]
        self.request_timeout = config["runtime"]["request_timeout"]
        self.input_file = config["paths"]["input_file"]
        self.output_file = config["paths"]["output_file"]
        self.exception_log = config["paths"]["exception_log"]
        self.cache_dir = config["paths"]["cache_dir"]
        self.use_rag = config["features"]["use_rag"]
        self.use_cot_guide = config["features"].get("use_cot_guide", True)
        self.wrd_enabled = config["features"].get("wrd_enabled", False)
        self.relevance_threshold = config["retrieval"].get("relevance_threshold", 0.6)
        self.filter_model_name = config["models"]["filter"]["model"]

        self.cache = ClauseCache(self.cache_dir, model_tag=self.openai_model)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def initialize(self):
        logger.info("Initializing embedding model...")
        self.model = SentenceTransformer(self.embedding_model)
        
        if self.use_rag:
            logger.info("Connecting to vector database...")
            chroma_client = PersistentClient(path=self.chroma_dir)
            self.collection = chroma_client.get_or_create_collection(name=self.collection_name)
        if self.use_rag and self.wrd_enabled:
            logger.info("Loading WRD relevance model (E5): %s", self.filter_model_name)
            self.filter_model = SentenceTransformer(self.filter_model_name)

        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    def get_similar_examples(self, clause_text: str) -> List[Dict]:
        if not self.use_rag or not self.collection:
            return []
        
        vec = self.model.encode([clause_text], convert_to_numpy=True).tolist()[0]
        results = self.collection.query(
            query_embeddings=[vec], n_results=20, include=["metadatas", "documents", "distances"]
        )
        examples = []
        candidates = []
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            confidence_score = sum(
                metadata.get(f'confidence_{dim}', 0.5)
                for dim in ['obligation', 'precision', 'delegation']
            ) / 3
            distance = results["distances"][0][i] if "distances" in results else 1.0
            candidates.append({
                "document": results["documents"][0][i],
                "metadata": metadata,
                "distance": distance,
                "confidence_score": confidence_score,
                "quality": (1 - distance) * confidence_score
            })
        candidates.sort(key=lambda x: x['quality'], reverse=True)
        for candidate in candidates:
            if candidate['confidence_score'] >= 0.5:
                examples.append({
                    "document": candidate["document"],
                    "metadata": candidate["metadata"],
                    "distance": candidate["distance"]
                })
            if len(examples) >= self.top_k:
                break
        if len(examples) < self.top_k:
            existing_docs = {ex["document"] for ex in examples}
            for candidate in candidates:
                if candidate["document"] not in existing_docs:
                    examples.append({
                        "document": candidate["document"],
                        "metadata": candidate["metadata"],
                        "distance": candidate["distance"]
                    })
                if len(examples) >= self.top_k:
                    break
        examples = examples[:self.top_k]
        if self.wrd_enabled and self.filter_model is not None and len(examples) > 0:
            query_text = "query: " + clause_text
            passage_texts = ["passage: " + ex["document"] for ex in examples]
            query_vec = self.filter_model.encode([query_text], convert_to_numpy=True)
            passage_vecs = self.filter_model.encode(passage_texts, convert_to_numpy=True)
            q_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)
            p_norm = passage_vecs / (np.linalg.norm(passage_vecs, axis=1, keepdims=True) + 1e-9)
            similarities = np.dot(p_norm, q_norm.T).flatten()
            filtered_indices = [i for i in range(len(examples)) if similarities[i] >= self.relevance_threshold]
            filtered_indices.sort(key=lambda i: similarities[i], reverse=True)
            examples = [examples[i] for i in filtered_indices]
        return examples
    
    async def call_llm(self, prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4096,
            "temperature": self.temperature
        }
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with self.session.post(self.openai_api_url, headers=headers, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            wait_time = min(2 ** attempt * 2, 30)
                            logger.warning(f"Rate limit reached. Retrying after {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_text = await response.text()
                            logger.error(f"API call failed with status {response.status}: {error_text}")
                            return None
                except Exception as e:
                    logger.warning(f"API request exception on attempt {attempt+1}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)
                    continue
            return None
    
    async def process_clause(self, clause: Dict) -> Tuple:
        clause_text = clause["text"].replace("\\n", "\n").strip()
        cached_result = self.cache.get_cached_result(clause_text, self.model)
        if cached_result:
            logger.info(f"Cache hit for ID: {clause['id']}")
            return clause, cached_result['llm_output'], cached_result['scores']
        
        similar_examples = self.get_similar_examples(clause_text)
        prompt = build_prompt(clause_text, similar_examples, self.use_cot_guide)
        llm_output = await self.call_llm(prompt)
        
        if not llm_output:
            return clause, None, {"obligation": None, "precision": None, "delegation": None}
        
        scores = extract_scores(llm_output)
        scores = {k: normalize_score(v) for k, v in scores.items()}
        
        if is_exception(scores, llm_output):
            logger.warning(f"Exceptional scores for ID: {clause['id']}. Retrying...")
            enhanced_prompt = prompt.replace(
                "CRITICAL INSTRUCTIONS:",
                "CRITICAL INSTRUCTIONS:\n0. PREVIOUS ATTEMPT MAY HAVE ISSUES - Please follow EVERY step precisely!"
            )
            llm_output2 = await self.call_llm(enhanced_prompt)
            scores2 = extract_scores(llm_output2) if llm_output2 else None
            
            if scores2:
                scores2 = {k: normalize_score(v) for k, v in scores2.items()}
                if is_exception(scores2, llm_output2):
                    logger.warning(f"Re-scoring still exceptional for ID: {clause['id']}. Using fallback.")
                    final_scores = {"obligation": 0.5, "precision": 0.5, "delegation": 0.25}
                    final_output = llm_output2
                else:
                    final_scores = scores2
                    final_output = llm_output2
            else:
                final_scores = scores
                final_output = llm_output
            
            self.cache.save_result(clause_text, final_scores, final_output)
            return clause, final_output, final_scores
        
        self.cache.save_result(clause_text, scores, llm_output)
        return clause, llm_output, scores
    
    async def process_batch(self, clauses: List[Dict]) -> List[Tuple]:
        tasks = [self.process_clause(clause) for clause in clauses]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def read_completed_ids(self, filename: str) -> set:
        if not os.path.exists(filename):
            return set()
        completed = set()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed.add(data['id'])
                except json.JSONDecodeError:
                    continue
        return completed
    
    async def run(self):
        logger.info("Initializing batch scoring system...")
        await self.initialize()
        
        with open(self.input_file, "r", encoding="utf-8") as f:
            clauses = json.load(f)
        
        completed_ids = self.read_completed_ids(self.output_file)
        remaining_clauses = [c for c in clauses if c["id"] not in completed_ids]
        
        logger.info(f"Total: {len(clauses)}, Completed: {len(completed_ids)}, Remaining: {len(remaining_clauses)}")
        
        if not remaining_clauses:
            logger.info("All clauses have been processed.")
            await self.close()
            return
        
        start_time = time.time()
        total_processed = 0
        exception_count = 0
        
        # Ensure output directories exist
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.exception_log).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, "a", encoding="utf-8") as fout, \
             open(self.exception_log, "a", encoding="utf-8") as log_fout:
            
            num_batches = math.ceil(len(remaining_clauses) / self.batch_size)
            for i, batch_start in enumerate(range(0, len(remaining_clauses), self.batch_size)):
                batch_end = min(batch_start + self.batch_size, len(remaining_clauses))
                batch = remaining_clauses[batch_start:batch_end]
                
                logger.info(f"Processing Batch {i + 1}/{num_batches} (Clauses {batch_start + 1}-{batch_end})")
                
                batch_start_time = time.time()
                results = await self.process_batch(batch)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Exception while processing clause: {result}")
                        continue
                    
                    clause, llm_output, scores = result
                    
                    if not llm_output or any(v is None for v in scores.values()):
                        logger.error(f"API call failed for clause ID: {clause['id']}. Skipping.")
                        continue
                    
                    if is_exception(scores, llm_output):
                        exception_count += 1
                        log_obj = {
                            "id": clause["id"],
                            "document_title": clause.get("document_title", clause.get("title", "")),
                            "text": clause["text"],
                            "llm_output": llm_output,
                            "scores": scores,
                            "timestamp": datetime.now().isoformat()
                        }
                        log_fout.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                    
                    result_obj = {
                        "id": clause["id"],
                        "document_title": clause.get("document_title", clause.get("title", clause.get("article_number", ""))),
                        "text": clause["text"],
                        "obligation": scores["obligation"],
                        "precision": scores["precision"],
                        "delegation": scores["delegation"]
                    }
                    fout.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                    total_processed += 1
                
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / total_processed if total_processed > 0 else 0
                
                logger.info(f"Batch {i+1} complete in {batch_time:.2f}s. Progress: {total_processed}/{len(remaining_clauses)} ({(total_processed/len(remaining_clauses))*100:.1f}%) | Avg: {avg_time_per_item:.2f}s/clause | Exceptions: {exception_count}")
                fout.flush()
                log_fout.flush()
        
        await self.close()
        
        total_time = time.time() - start_time
        avg_time_per_clause = total_time / total_processed if total_processed > 0 else 0
        exception_rate = (exception_count / total_processed * 100) if total_processed > 0 else 0
        
        logger.info("=" * 50)
        logger.info("Batch processing complete.")
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        logger.info(f"Processed: {total_processed}")
        logger.info(f"Avg time per clause: {avg_time_per_clause:.2f} seconds")
        logger.info(f"Exceptions: {exception_count} ({exception_rate:.1f}%)")
        logger.info(f"Output: {self.output_file}")
