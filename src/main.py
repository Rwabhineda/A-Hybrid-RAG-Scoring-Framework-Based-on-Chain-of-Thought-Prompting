# -*- coding: utf-8 -*-
"""
ASEAN Legal Clause Scoring - Unified Entry Point
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Optional[str] = None) -> dict:
    root = get_repo_root()
    
    if config_path:
        cfg_file = Path(config_path)
    elif os.getenv("SCORING_CONFIG"):
        cfg_file = Path(os.getenv("SCORING_CONFIG"))
    else:
        cfg_file = root / "configs" / "gpt-4o" / "gpt-4o.yaml"
    
    if not cfg_file.is_absolute():
        cfg_file = root / cfg_file
    
    if not cfg_file.exists():
        print(f"[ERROR] Config file not found: {cfg_file}")
        sys.exit(1)
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    config = _normalize_config(config, root)
    
    # API Key from env only
    api_provider = os.getenv("API_PROVIDER", "openai").lower()
    if api_provider == "deepseek":
        if os.getenv("DEEPSEEK_API_KEY"):
            config["models"]["openai"]["api_key"] = os.getenv("DEEPSEEK_API_KEY")
        if config["models"]["openai"]["model"].startswith("gpt-"):
            config["models"]["openai"]["_original_model"] = config["models"]["openai"]["model"]
            config["models"]["openai"]["model"] = "deepseek-chat"
    else:
        if os.getenv("OPENAI_API_KEY"):
            config["models"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
    
    config["_api_provider"] = api_provider
    return config


def _normalize_config(config: dict, root: Path) -> dict:
    if "paths" in config and "models" in config:
        for section in ["paths", "vector_db"]:
            if section in config:
                for key, val in config[section].items():
                    if key == "collection_name":  # Skip non-path fields
                        continue
                    if isinstance(val, str) and not Path(val).is_absolute():
                        config[section][key] = str(root / val)
        return config
    
    # Convert old format
    return {
        "experiment": config.get("experiment", {"name": "default", "description": ""}),
        "paths": {
            "input_file": _resolve_path(config.get("input_file", "data/processed/test_articles.json"), root),
            "output_file": _resolve_path(config.get("output_file", "outputs/results.jsonl"), root),
            "exception_log": _resolve_path(config.get("exception_log_file", "logs/exceptions.log"), root),
            "cache_dir": _resolve_path(config.get("cache_dir", "data/cache"), root)
        },
        "vector_db": {
            "chroma_dir": _resolve_path(config.get("chroma_dir", "data/rag/chroma_db"), root),
            "collection_name": config.get("collection_name", "asean_scoring")
        },
        "models": {
            "openai": {
                "model": config.get("openai_model", "gpt-4o"),
                "api_url": config.get("openai_api_url", "https://api.openai.com/v1/chat/completions"),
                "api_key": ""
            },
            "embedding": {"model": config.get("embedding_model", "intfloat/e5-large-v2")},
            "filter": {"model": config.get("filter_model", "nlpaueb/legal-bert-base-uncased")}
        },
        "retrieval": {"top_k": config.get("top_k", 5), "similarity_threshold": config.get("similarity_threshold", 1.0)},
        "runtime": {"max_concurrent": config.get("max_concurrent", 3), "batch_size": config.get("batch_size", 9), "request_timeout": config.get("request_timeout", 300)},
        "features": {
            "mode": config.get("features", {}).get("mode", "rag"),
            "use_rag": config.get("features", {}).get("use_rag", True),
            "use_cot_guide": config.get("features", {}).get("use_cot_guide", True),
            "wrd_enabled": config.get("wrd_enabled", False),
            "zero_shot": config.get("zero_shot", False)
        }
    }


def _resolve_path(path_str: str, root: Path) -> str:
    if not path_str:
        return str(root)
    path = Path(path_str)
    return str(path if path.is_absolute() else root / path)


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_config(config: dict) -> bool:
    if "paths" not in config or "models" not in config:
        print("[ERROR] Missing required config sections")
        return False
    
    for key in ["input_file", "output_file"]:
        if key not in config["paths"]:
            print(f"[ERROR] Missing required path: {key}")
            return False
    
    if not config["models"]["openai"].get("api_key"):
        print("[ERROR] API key not found! Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env")
        return False
    
    if not Path(config["paths"]["input_file"]).exists():
        print(f"[ERROR] Input file not found: {config['paths']['input_file']}")
        return False
    
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="ASEAN Legal Clause Scoring",
        epilog="""
Examples:
  uv run python src/main.py
  uv run python src/main.py --config configs/gpt-4o/gpt-4o.yaml
  uv run python src/main.py --config configs/gpt-3.5-turbo/gpt-3.5-turbo.yaml
        """
    )
    parser.add_argument("--config", "-c", default=None, help="Config file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    mode = config.get("features", {}).get("mode", "rag")
    
    if mode != "random":
        if not validate_config(config):
            sys.exit(1)
    
    logger.info(f"MODE: {mode}")
    if mode != "random":
        logger.info(f"API: {config.get('_api_provider', 'openai').upper()}")
        logger.info(f"Model: {config['models']['openai']['model']}")
    
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Input: {config['paths']['input_file']}")
    logger.info(f"Output: {config['paths']['output_file']}")
    
    Path(config['paths']['output_file']).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from scoring.engine import BatchScorer
        scorer = BatchScorer(config)
        await scorer.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Scoring completed!")


if __name__ == "__main__":
    asyncio.run(main())
