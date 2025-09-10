import json
import asyncio
import aiohttp
import os
import hashlib
import pickle
import time
import re
from typing import Dict, Optional, Tuple
from datetime import datetime

# ==== 基本参数 ====
INPUT_FILE = r"C:\Users\zzh69\Desktop\国际文书的法律化程度：基于检索增强生成技术（RAG）的文书条款评分方法\RAG Databases\Scoring Code\Test_Article.json"
OUTPUT_FILE = r"C:\Users\zzh69\Desktop\国际文书的法律化程度：基于检索增强生成技术（RAG）的文书条款评分方法\RAG Databases\Scoring Code\gpt-5\Test_Article-gpt-5-WRD-Zero-shot.jsonl"
EXCEPTION_LOG_FILE = r"C:\Users\zzh69\Desktop\国际文书的法律化程度：基于检索增强生成技术（RAG）的文书条款评分方法\RAG Databases\Scoring Code\gpt-5\Test_Article_exception_llm_output_WRD_Zero-shot.log"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-5"
PROXY_URL = None

MAX_CONCURRENT = 3
BATCH_SIZE = 9
REQUEST_TIMEOUT = 300
CACHE_DIR = "clause_cache_noRAG_Zero-shot"

SCORING_GUIDE = """

You are an expert ASEAN legal clause evaluator.  
Your task: Given a single legal clause, assign it a score from the set {0.0, 0.25, 0.5, 0.75, 1.0} on each of the following three dimensions:  
  • Obligation (strength of commitment)  
  • Precision (level of detail and concreteness)  
  • Delegation (degree of decision-making power granted to a third party)  

"""

def build_prompt(clause_text: str) -> str:
    return f"""{SCORING_GUIDE}

Now, analyze the following clause:

Clause: {clause_text}

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

class ClauseCache:
    def __init__(self, cache_dir: str = CACHE_DIR, model_tag: str = "default"):
        self.cache_dir = os.path.join(cache_dir, model_tag)
        self.memory_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)
    def _get_cache_key(self, clause_text: str) -> str:
        normalized = re.sub(r'\s+', ' ', clause_text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
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
        self.session = None
        self.cache = ClauseCache(model_tag=model_tag)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def initialize(self):
        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=30)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def close(self):
        if self.session:
            await self.session.close()

    async def call_llm(self, prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 4096
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
        cached_result = self.cache.get_cached_result(clause_text, None)
        if cached_result:
            print(f"缓存命中: {clause['id']}")
            return clause, cached_result['llm_output'], cached_result['scores']
        prompt = build_prompt(clause_text)
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

    async def process_batch(self, clauses):
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
    print("🚀 启动批量自动评分系统（无RAG范例对照）...")
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
