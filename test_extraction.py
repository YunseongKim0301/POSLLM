#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POS 추출 테스트 스크립트 (Windows PC용)
"""

import os
import sys
import json
import logging
from pathlib import Path

# v53_extractor 임포트
from v53_extractor import (
    POSExtractorV52,
    Config,
    SpecItem
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_config():
    """테스트용 Config 생성"""
    config = Config()

    # 기본 설정
    config.extraction_mode = "light"
    config.data_source_mode = "file"  # 파일 모드 (PostgreSQL 없이 테스트)

    # 경로 설정
    base_dir = Path(__file__).parent
    config.base_folder = str(base_dir)
    config.light_mode_pos_folder = str(base_dir / "uploaded_files")
    config.glossary_path = str(base_dir / "test_data" / "pos_dict.txt")
    config.specdb_path = str(base_dir / "test_data" / "umgv_fin.txt")
    config.spec_path = str(base_dir / "test_data" / "ext_tmpl.txt")
    config.output_path = str(base_dir / "output")

    # LLM 설정
    config.use_llm = True
    config.enable_llm_fallback = True
    config.ollama_model = "qwen2.5:32b"  # 또는 "gemma2:27b"
    config.ollama_host = "127.0.0.1"
    config.ollama_ports = [11434]
    config.ollama_timeout = 180

    # Voting 설정
    config.vote_enabled = True
    config.vote_k = 2  # 테스트용으로 2로 설정
    config.vote_min_agreement = 2

    # 출력 설정
    config.save_json = True
    config.save_csv = True
    config.save_debug_csv = True

    # 임베딩 비활성화 (PostgreSQL 없이 테스트)
    config.use_precomputed_embeddings = False
    config.enable_semantic_search = False

    return config


def load_test_specs(spec_file):
    """테스트용 사양 목록 로드"""
    specs = []

    if not os.path.exists(spec_file):
        logger.warning(f"템플릿 파일 없음: {spec_file}")
        return specs

    # 간단한 CSV 파싱 (헤더 스킵)
    with open(spec_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 0:  # 헤더 스킵
            continue

        parts = line.strip().split('\t')  # TSV 형식 가정
        if len(parts) >= 3:
            spec = SpecItem(
                spec_name=parts[0].strip(),
                spec_code=parts[1].strip() if len(parts) > 1 else "",
                equipment=parts[2].strip() if len(parts) > 2 else "",
                expected_unit=parts[3].strip() if len(parts) > 3 else "",
                hull="",
                matnr=""
            )
            specs.append(spec)

    logger.info(f"로드된 사양 수: {len(specs)}")
    return specs


def test_single_extraction():
    """단일 POS 파일 테스트"""
    logger.info("=" * 80)
    logger.info("POS 추출 테스트 시작")
    logger.info("=" * 80)

    # Config 생성
    config = create_test_config()

    # 출력 디렉토리 생성
    os.makedirs(config.output_path, exist_ok=True)

    # Extractor 초기화
    logger.info("Extractor 초기화 중...")
    extractor = POSExtractorV52(config, mode="light")
    extractor.initialize()

    # POS 파일 찾기
    pos_folder = Path(config.light_mode_pos_folder)
    html_files = list(pos_folder.glob("*.html"))

    if not html_files:
        logger.error(f"POS 파일을 찾을 수 없습니다: {pos_folder}")
        return

    test_file = html_files[0]
    logger.info(f"테스트 파일: {test_file.name}")

    # 사양 목록 로드
    specs = load_test_specs(config.spec_path)
    if not specs:
        logger.error("사양 목록을 로드할 수 없습니다")
        return

    # 추출 테스트 (처음 5개만)
    results = []
    test_specs = specs[:5]

    logger.info(f"추출 시작: {len(test_specs)}개 사양")

    for i, spec in enumerate(test_specs, 1):
        logger.info(f"[{i}/{len(test_specs)}] 추출 중: {spec.spec_name}")

        try:
            result = extractor.extract_single(str(test_file), spec)
            results.append(result)

            # 결과 출력
            if result.get('pos_umgv_value'):
                logger.info(f"  ✓ 성공: {result['pos_umgv_value']} {result.get('pos_umgv_uom', '')}")
            else:
                logger.warning(f"  ✗ 실패: {result.get('_method', 'UNKNOWN')}")

        except Exception as e:
            logger.error(f"  ✗ 오류: {e}")

    # 결과 저장
    output_file = Path(config.output_path) / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"결과 저장: {output_file}")

    # 통계 출력
    success_count = sum(1 for r in results if r.get('pos_umgv_value'))
    logger.info("=" * 80)
    logger.info(f"추출 완료: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = test_single_extraction()
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
