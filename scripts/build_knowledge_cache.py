#!/usr/bin/env python3
"""
Knowledge Cache Builder Script

용어집/사양값DB에서 동적 지식 캐시를 생성합니다.

Usage:
    # 기본 사용 (data/glossary.xlsx 사용)
    python scripts/build_knowledge_cache.py

    # 특정 용어집 파일 지정
    python scripts/build_knowledge_cache.py --glossary path/to/glossary.xlsx

    # 강제 재생성 (기존 캐시 무시)
    python scripts/build_knowledge_cache.py --force
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v53_extractor import KnowledgeCacheBuilder
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Build knowledge cache from glossary/spec_db'
    )
    parser.add_argument(
        '--glossary',
        type=str,
        default='data/glossary.xlsx',
        help='Path to glossary Excel file (default: data/glossary.xlsx)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if cache exists'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='knowledge_base/data',
        help='Output directory for cache files (default: knowledge_base/data)'
    )

    args = parser.parse_args()

    # Check glossary file
    if not os.path.exists(args.glossary):
        print(f"[ERROR] Glossary file not found: {args.glossary}")
        print(f"[INFO] Please provide a valid glossary file path")
        sys.exit(1)

    # Set cache directory
    KnowledgeCacheBuilder.CACHE_DIR = args.output_dir

    # Check existing cache
    if not args.force:
        cache_files = [
            'synonyms_cache.json',
            'units_cache.json',
            'abbreviations_cache.json'
        ]

        all_exist = all(
            KnowledgeCacheBuilder.is_cache_valid(f, max_age_hours=24*7)  # 1 week
            for f in cache_files
        )

        if all_exist:
            print("[INFO] Valid cache already exists (less than 1 week old)")
            print("[INFO] Use --force to rebuild")

            # Show cache info
            for cache_file in cache_files:
                cache_path = KnowledgeCacheBuilder.get_cache_path(cache_file)
                if os.path.exists(cache_path):
                    size = os.path.getsize(cache_path)
                    mtime = os.path.getmtime(cache_path)
                    import time
                    age_hours = (time.time() - mtime) / 3600
                    print(f"  - {cache_file}: {size:,} bytes, {age_hours:.1f} hours old")

            return

    # Load glossary
    print(f"[INFO] Loading glossary from: {args.glossary}")
    try:
        glossary_df = pd.read_excel(args.glossary)
        print(f"[INFO] Loaded {len(glossary_df)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to load glossary: {e}")
        sys.exit(1)

    # Show sample data
    print("\n[INFO] Sample glossary data:")
    print(glossary_df.head(3)[['umgv_desc', 'pos_umgv_desc', 'umgv_uom', 'pos_umgv_uom']].to_string())

    # Build caches
    print("\n" + "="*60)
    print("Building Knowledge Caches")
    print("="*60 + "\n")

    KnowledgeCacheBuilder.rebuild_all_caches(glossary_df=glossary_df)

    # Show results
    print("\n" + "="*60)
    print("Cache Build Complete")
    print("="*60 + "\n")

    cache_files = [
        'synonyms_cache.json',
        'units_cache.json',
        'abbreviations_cache.json'
    ]

    for cache_file in cache_files:
        cache_path = KnowledgeCacheBuilder.get_cache_path(cache_file)
        if os.path.exists(cache_path):
            size = os.path.getsize(cache_path)
            print(f"  ✓ {cache_file}: {size:,} bytes")

            # Load and show stats
            cache_data = KnowledgeCacheBuilder.load_cache(cache_file)
            if 'forward' in cache_data:
                print(f"    - {len(cache_data['forward'])} standard terms")
                print(f"    - {len(cache_data['reverse'])} variant mappings")
            elif isinstance(cache_data, dict):
                print(f"    - {len(cache_data)} entries")
        else:
            print(f"  ✗ {cache_file}: NOT FOUND")

    print(f"\n[INFO] Cache directory: {os.path.abspath(args.output_dir)}")
    print("[INFO] Next: Run v53_extractor.py to use the cache")


if __name__ == "__main__":
    main()
