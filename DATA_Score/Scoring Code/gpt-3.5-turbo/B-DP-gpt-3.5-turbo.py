# ===== Standard Library =====
import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===== Third-Party =====
import aiohttp
import numpy as np
import torch
import chromadb
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder, SentenceTransformer


# ==== 基本参数 ====
INPUT_FILE = r"C:\Users\patrick\Desktop\111\Test_Article.json"
OUTPUT_FILE = r"C:\Users\patrick\Desktop\111\Test_Article_scored.jsonl"
EXCEPTION_LOG_FILE = r"C:\Users\patrick\Desktop\111\Test_Article_exception_llm_output.log"
CHROMA_DIR = r"C:\Users\patrick\Desktop\RAG Databases\RAG_Data\chroma_db"
COLLECTION_NAME = "asean_scoring"
EMBEDDING_MODEL = "all-mpnet-base-v2"
FILTER_MODEL = "nlpaueb/legal-bert-base-uncased"
TOP_K = 5

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-3.5-turbo"
PROXY_URL = None

MAX_CONCURRENT = 3
BATCH_SIZE = 9
REQUEST_TIMEOUT = 300
SIMILARITY_THRESHOLD = 1.0
CACHE_DIR = "clause_cache"


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

def build_prompt(clause_text: str, similar_examples: List[Dict]) -> str:
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
    return f"""{SCORING_GUIDE}

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

def extract_scores(output_str: str) -> Dict[str, Optional[float]]:
    json_patterns = [
        r'\{[^}]*"obligation"[^}]+\}',
        r'FINAL SCORES[^{]*(\{[^}]+\})',
        r'"obligation":\s*([\d.]+)[^}]+}'
    ]
    for pattern in json_patterns:
        match = re.search(pattern, output_str, re.I | re.S)
        if match:
            try:
                json_str = match.group() if '{' in match.group() else match.group(1)
                scores = json.loads(json_str)
                return {k: float(v) for k, v in scores.items()}
            except:
                continue
    scores = {}
    for dim in ["obligation", "precision", "delegation"]:
        patterns = [
            rf"{dim}.*?(?:final\s+)?score[:\s]*(0(?:\.0)?|0\.25|0\.5|0\.75|1(?:\.0)?)",
            rf"{dim}.*?→\s*Score\s+(0(?:\.0)?|0\.25|0\.5|0\.75|1(?:\.0)?)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output_str, re.I | re.S)
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

def filter_with_legal_ce(
    query: str,
    items: List[Dict],
    *,
    tau: float = 0.5,
    add_prob: bool = False,
    keep_order: bool = True,
    top_k: Optional[int] = None,
    model_name: Optional[str] = None,
    max_length: int = 512,
    quiet: bool = True
) -> List[Dict]:
    if quiet:
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def _get_device_str() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _to_source_schema(lst: List[Dict]) -> List[Dict]:
        return [{"document": it.get("document", ""),
                 "metadata": it.get("metadata", {}),
                 "distance": it.get("distance", 1.0)} for it in lst]

    if not hasattr(filter_with_legal_ce, "_CE_CACHE"):
        filter_with_legal_ce._CE_CACHE = {}
    _CE_CACHE: Dict[Tuple[str, int, str], CrossEncoder] = getattr(filter_with_legal_ce, "_CE_CACHE")

    def _load_cross_encoder(name: Optional[str], max_len: int) -> CrossEncoder:
        _name = name or os.getenv("LEGAL_CE_MODEL") or FILTER_MODEL
        device = _get_device_str()
        key = (_name, max_len, device)
        if key in _CE_CACHE:
            return _CE_CACHE[key]
        ce = CrossEncoder(_name, max_length=max_len, device=device, num_labels=1)
        _CE_CACHE[key] = ce
        return ce

    if not items:
        return []

    ce = _load_cross_encoder(model_name, max_length)
    pairs = [(query, it.get("document", "")) for it in items]
    scores = ce.predict(pairs, show_progress_bar=False)
    probs = [_sigmoid(s) for s in scores]

    annotated = []
    for it, p in zip(items, probs):
        x = dict(it)
        if add_prob:
            x["prob"] = float(p)
        annotated.append(x)

    filtered = [x for x, p in zip(annotated, probs) if p >= tau]
    kept = filtered if len(filtered) >= len(items) else annotated

    if keep_order:
        out = kept
    else:
        if add_prob:
            pinned, tail = kept[:1], kept[1:]
            tail.sort(key=lambda x: x.get("prob", 0.0), reverse=True)
            out = pinned + tail
        else:
            out = kept

    if isinstance(top_k, int) and top_k > 0:
        out = out[:top_k]

    return out if add_prob else _to_source_schema(out)


class ClauseCache:
    def __init__(self, cache_dir: str = CACHE_DIR, similarity_threshold: float = SIMILARITY_THRESHOLD, model_tag: str = "default"):
        self.cache_dir = os.path.join(cache_dir, model_tag)
        self.similarity_threshold = similarity_threshold
        self.memory_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)
    def _get_cache_key(self, clause_text: str) -> str:
        normalized = re.sub(r'\s+', ' ', clause_text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
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
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.memory_cache[cache_key] = cached_data
                return cached_data
            except:
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
        except:
            pass

class BatchScorer:
    def __init__(self, model_tag="gpt-4o"):
        self.model = None
        self.collection = None
        self.session = None
        self.cache = ClauseCache(model_tag=model_tag)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    async def initialize(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        chroma_client = PersistentClient(path=CHROMA_DIR)
        self.collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    async def close(self):
        if self.session:
            await self.session.close()

    def get_similar_examples(self, clause_text: str) -> List[Dict]:
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
            if len(examples) >= TOP_K:
                break
        if len(examples) < TOP_K:
            for candidate in candidates:
                if candidate not in examples:
                    examples.append({
                        "document": candidate["document"],
                        "metadata": candidate["metadata"],
                        "distance": candidate["distance"]
                    })
                if len(examples) >= TOP_K:
                    break

        reviewed = filter_with_legal_ce(
            query=clause_text,
            items=examples
        )

        return  (reviewed or examples)[:TOP_K]


    async def call_llm(self, prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4096
        }
        async with self.semaphore:
            for attempt in range(3):
                try:
                    kwargs = {}
                    if PROXY_URL:
                        kwargs['proxy'] = PROXY_URL
                    async with self.session.post(OPENAI_API_URL, headers=headers, json=payload, **kwargs) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            wait_time = min(2 ** attempt * 2, 30)
                            print(f"API限流，等待{wait_time}秒后重试...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_text = await response.text()
                            print(f"API ERROR: {response.status} {error_text}")
                            return None
                except Exception as e:
                    print(f"API Exception (attempt {attempt+1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)
                    continue
            return None

    async def process_clause(self, clause: Dict) -> Tuple:
        clause_text = clause["text"].replace("\\n", "\n").strip()
        # cached_result = self.cache.get_cached_result(clause_text, self.model)
        # if cached_result:
        #     print(f"缓存命中: {clause['id']}")
        #     return clause, cached_result['llm_output'], cached_result['scores']
        similar_examples = self.get_similar_examples(clause_text)
        prompt = build_prompt(clause_text, similar_examples)
        llm_output = await self.call_llm(prompt)
        if not llm_output:
            return clause, None, {"obligation": None, "precision": None, "delegation": None}
        scores = extract_scores(llm_output)
        scores = {k: normalize_score(v) for k, v in scores.items()}
        if is_exception(scores, llm_output):
            print(f"⚠️ 异常条款二次评分: {clause['id']}")
            enhanced_prompt = prompt.replace(
                "CRITICAL INSTRUCTIONS:",
                "CRITICAL INSTRUCTIONS:\n0. PREVIOUS ATTEMPT MAY HAVE ISSUES - Please follow EVERY step precisely!"
            )
            llm_output2 = await self.call_llm(enhanced_prompt)
            scores2 = extract_scores(llm_output2) if llm_output2 else None
            scores2 = {k: normalize_score(v) for k, v in scores2.items()} if scores2 else scores
            if is_exception(scores2, llm_output2):
                print(f"⚠️⚠️ 二次评分仍异常，使用保守评分: {clause['id']}")
                final_scores = {"obligation": 0.5, "precision": 0.5, "delegation": 0.25}
            else:
                final_scores = scores2
            return clause, llm_output2, final_scores
        self.cache.save_result(clause_text, scores, llm_output)
        return clause, llm_output, scores

    async def process_batch(self, clauses: List[Dict]) -> List[Tuple]:
        tasks = [self.process_clause(clause) for clause in clauses]
        return await asyncio.gather(*tasks, return_exceptions=True)

def read_completed_ids(filename: str) -> set:
    if not os.path.exists(filename):
        return set()
    completed = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                completed.add(data['id'])
            except:
                continue
    return completed

async def main():
    print("🚀 启动批量自动评分系统...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        clauses = json.load(f)
    completed_ids = read_completed_ids(OUTPUT_FILE)
    remaining_clauses = [c for c in clauses if c["id"] not in completed_ids]
    print(f"已完成 {len(completed_ids)} 条，剩余 {len(remaining_clauses)} 条待处理")
    if not remaining_clauses:
        print("✅ 所有条款已处理完成！")
        return
    start_time = time.time()
    total_processed = 0
    exception_count = 0
    scorer = BatchScorer(model_tag=OPENAI_MODEL)
    await scorer.initialize()
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout, \
         open(EXCEPTION_LOG_FILE, "a", encoding="utf-8") as log_fout:
        for batch_start in range(0, len(remaining_clauses), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(remaining_clauses))
            batch = remaining_clauses[batch_start:batch_end]
            print(f"\n📦 处理批次 {batch_start//BATCH_SIZE + 1}: 条款 {batch_start+1}-{batch_end}")
            batch_start_time = time.time()
            results = await scorer.process_batch(batch)
            for result in results:
                if isinstance(result, Exception):
                    print(f"❌ 处理出错: {result}")
                    continue
                clause, llm_output, scores = result
                if not llm_output:
                    print(f"⚠️ API调用失败: {clause['id']}")
                    continue
                final_scores = scores
                if is_exception(scores, llm_output):
                    exception_count += 1
                    log_obj = {
                        "id": clause["id"],
                        "document_title": clause.get("document_title", clause.get("title", "")),
                        "text": clause["text"],
                        "llm_output": llm_output,
                        "scores": scores,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    log_fout.write(json.dumps(log_obj, ensure_ascii=False) + "\n")
                result_obj = {
                    "id": clause["id"],
                    "document_title": clause.get("document_title", clause.get("title", clause.get("article_number", ""))),
                    "text": clause["text"],
                    "obligation": final_scores["obligation"],
                    "precision": final_scores["precision"],
                    "delegation": final_scores["delegation"]
                }
                fout.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
                total_processed += 1
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / total_processed if total_processed > 0 else 0
            print(f"✅ 批次完成，用时: {batch_time:.1f}s")
            print(f"📊 总进度: {total_processed}/{len(remaining_clauses)} ({total_processed/len(remaining_clauses)*100:.1f}%)")
            print(f"⚡ 平均每条: {avg_time_per_item:.1f}s")
            print(f"⚠️  异常条款数: {exception_count}")
            fout.flush()
            log_fout.flush()
    await scorer.close()
    total_time = time.time() - start_time
    print(f"\n🎉 全部处理完成！")
    print(f"⏱️  总用时: {total_time/60:.1f}分钟")
    print(f"⚡ 平均每条: {total_time/len(remaining_clauses):.1f}秒")
    print(f"⚠️  异常条款数: {exception_count} ({exception_count/total_processed*100:.1f}%)")
    print(f"📁 输出文件: {OUTPUT_FILE}")
    print(f"📋 异常日志: {EXCEPTION_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
