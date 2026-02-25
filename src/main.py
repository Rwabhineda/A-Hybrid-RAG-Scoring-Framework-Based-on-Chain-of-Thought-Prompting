# -*- coding: utf-8 -*-
"""
ASEAN Legal Clause Scoring - Unified Entry Point

Usage:
    # Run with default config (gpt-4o)
    uv run python src/main.py
    
    # Run with specific config
    uv run python src/main.py --config configs/gpt-4o.yaml
    
    # Run other models
    uv run python src/main.py --config configs/gpt-3.5-turbo.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    Supports both new hierarchical format and old flat format.
    
    Priority:
    1. Command line argument (--config)
    2. Environment variable (SCORING_CONFIG)
    3. Default config (configs/gpt-4o.yaml)
    """
    root = get_repo_root()
    
    if config_path:
        cfg_file = Path(config_path)
    elif os.getenv("SCORING_CONFIG"):
        cfg_file = Path(os.getenv("SCORING_CONFIG"))
    else:
        cfg_file = root / "configs" / "gpt-4o.yaml"
    
    if not cfg_file.is_absolute():
        cfg_file = root / cfg_file
    
    if not cfg_file.exists():
        print(f"[ERROR] Config file not found: {cfg_file}")
        print(f"[HINT] Create a config file or use: --config configs/<model>.yaml")
        sys.exit(1)
    
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Normalize config to new hierarchical format
    config = _normalize_config(config, root)
    
    # Configure API provider from environment variable
    api_provider = os.getenv("API_PROVIDER", "openai").lower()
    
    # API URL from config file (not from env)
    # API Key only from environment variable (not from config)
    if api_provider == "deepseek":
        # Use DeepSeek API - only set API key from env
        if os.getenv("DEEPSEEK_API_KEY"):
            config["models"]["openai"]["api_key"] = os.getenv("DEEPSEEK_API_KEY")
        else:
            config["models"]["openai"]["api_key"] = ""
        # DeepSeek uses different model names, map if needed
        model = config["models"]["openai"]["model"]
        if model.startswith("gpt-"):
            # Map OpenAI model names to DeepSeek equivalents
            config["models"]["openai"]["_original_model"] = model
            config["models"]["openai"]["model"] = "deepseek-chat"
    else:
        # Use OpenAI/ChatGPT API (default) - only set API key from env
        if os.getenv("OPENAI_API_KEY"):
            config["models"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
        else:
            config["models"]["openai"]["api_key"] = ""
    
    # Store provider info for logging
    config["_api_provider"] = api_provider
    
    return config


def _normalize_config(config: dict, root: Path) -> dict:
    """Normalize old flat config format to new hierarchical format."""
    # Check if already in new format
    if "paths" in config and "models" in config:
        # Convert relative paths to absolute
        for section in ["paths", "vector_db"]:
            if section in config:
                for key, val in config[section].items():
                    if isinstance(val, str) and not Path(val).is_absolute():
                        config[section][key] = str(root / val)
        return config
    
    # Convert old format to new format
    new_config = {
        "experiment": {
            "name": config.get("experiment", {}).get("name", "default"),
            "description": config.get("experiment", {}).get("description", "")
        },
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
                "api_key": config.get("openai_api_key", "")
            },
            "embedding": {
                "model": config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
            },
            "filter": {
                "model": config.get("filter_model", "nlpaueb/legal-bert-base-uncased")
            }
        },
        "retrieval": {
            "top_k": config.get("top_k", 5),
            "similarity_threshold": config.get("similarity_threshold", 1.0)
        },
        "runtime": {
            "max_concurrent": config.get("max_concurrent", 3),
            "batch_size": config.get("batch_size", 9),
            "request_timeout": config.get("request_timeout", 300)
        },
        "features": {
            "wrd_enabled": config.get("wrd_enabled", False),
            "zero_shot": config.get("zero_shot", False),
            "use_rag": config.get("use_rag", True)
        }
    }
    return new_config


def _resolve_path(path_str: str, root: Path) -> str:
    """Resolve relative path to absolute path."""
    if not path_str:
        return str(root)
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(root / path)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_config(config: dict) -> bool:
    """Validate configuration."""
    # Check required sections
    required_sections = ["paths", "models"]
    for section in required_sections:
        if section not in config:
            print(f"[ERROR] Missing required config section: {section}")
            return False
    
    # Check required keys in paths
    path_keys = ["input_file", "output_file"]
    for key in path_keys:
        if key not in config["paths"] or not config["paths"][key]:
            print(f"[ERROR] Missing required config key: paths.{key}")
            return False
    
    # Check model configuration
    if "openai" not in config["models"]:
        print("[ERROR] Missing OpenAI model configuration")
        return False
    
    if not config["models"]["openai"].get("model"):
        print("[ERROR] Missing OpenAI model name")
        return False
    
    # Check API key
    api_provider = config.get("_api_provider", "openai")
    if not config["models"]["openai"].get("api_key"):
        if api_provider == "deepseek":
            print("[ERROR] DeepSeek API key not found!")
            print("[HINT] Set DEEPSEEK_API_KEY in .env file or environment variable")
        else:
            print("[ERROR] OpenAI API key not found!")
            print("[HINT] Set OPENAI_API_KEY in .env file or environment variable")
        return False
    
    # Check input file exists
    input_file = Path(config["paths"]["input_file"])
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return False
    
    return True


async def main():
    parser = argparse.ArgumentParser(
        description="ASEAN Legal Clause Scoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run python src/main.py
  
  # Run with specific model config
  uv run python src/main.py --config configs/gpt-3.5-turbo.yaml
  
  # Use environment variable for config
  $env:SCORING_CONFIG="configs/gpt-4o.yaml"; uv run python src/main.py
        """
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to configuration file (default: configs/gpt-4o.yaml)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Validate configuration
    if not validate_config(config):
        sys.exit(1)
    
    api_provider = config.get("_api_provider", "openai")
    model_name = config['models']['openai']['model']
    original_model = config['models']['openai'].get('_original_model', model_name)
    
    logger.info(f"API Provider: {api_provider.upper()}")
    logger.info(f"Using model: {model_name}" + (f" (mapped from {original_model})" if original_model != model_name else ""))
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Input file: {config['paths']['input_file']}")
    logger.info(f"Output file: {config['paths']['output_file']}")
    
    # Ensure output directory exists
    output_dir = Path(config['paths']['output_file']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and run the scoring module
    try:
        from scoring.engine import BatchScorer
        
        scorer = BatchScorer(config)
        await scorer.run()
        
    except ImportError as e:
        logger.error(f"Failed to import scoring module: {e}")
        logger.error("Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        sys.exit(1)
    
    logger.info("Scoring completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
