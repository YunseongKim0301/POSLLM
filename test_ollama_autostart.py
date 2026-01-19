#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ollama 자동 시작 기능 테스트

v77_extractor.py의 UnifiedLLMClient Ollama 자동 시작 기능을 테스트합니다.
"""

import sys
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def test_ollama_autostart():
    """
    Ollama 자동 시작 테스트
    """
    logger.info("=" * 80)
    logger.info("Ollama 자동 시작 기능 테스트")
    logger.info("=" * 80)

    # v77_extractor에서 UnifiedLLMClient 임포트
    try:
        from v77_extractor import UnifiedLLMClient
        logger.info("✓ v77_extractor 임포트 성공")
    except ImportError as e:
        logger.error(f"✗ v77_extractor 임포트 실패: {e}")
        return False

    # UnifiedLLMClient 초기화 (자동 시작 활성화)
    logger.info("")
    logger.info("-" * 80)
    logger.info("1. UnifiedLLMClient 초기화 (Ollama 자동 시작)")
    logger.info("-" * 80)

    try:
        client = UnifiedLLMClient(
            model="gemma3:27b",
            ollama_ports=[11434],  # 일단 1개 포트만 테스트
            auto_start_ollama=True,
            timeout=30,
            logger=logger
        )
        logger.info("✓ UnifiedLLMClient 초기화 성공")
    except Exception as e:
        logger.error(f"✗ UnifiedLLMClient 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 간단한 LLM 호출 테스트
    logger.info("")
    logger.info("-" * 80)
    logger.info("2. LLM 호출 테스트")
    logger.info("-" * 80)

    test_prompt = """Extract the value for 'CAPACITY(SWL)' from this text:

Text: "Hoisting capacity | SWL 6 tonnes"

Return only the value (number and unit), nothing else."""

    logger.info(f"프롬프트:\n{test_prompt}")

    try:
        response, in_tok, out_tok = client.generate(test_prompt)

        if response:
            logger.info(f"✓ LLM 응답 성공:")
            logger.info(f"  응답: {response.strip()}")
            logger.info(f"  입력 토큰: {in_tok}, 출력 토큰: {out_tok}")
            return True
        else:
            logger.warning("✗ LLM 응답 없음 (Ollama가 시작되었지만 모델이 없을 수 있음)")
            logger.warning("  해결: ollama pull gemma3:27b")
            return False

    except Exception as e:
        logger.error(f"✗ LLM 호출 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    success = test_ollama_autostart()

    logger.info("")
    logger.info("=" * 80)
    if success:
        logger.info("✓ 모든 테스트 통과!")
        logger.info("  v77_extractor.py에서 Ollama가 자동으로 시작되고 정상 작동합니다.")
    else:
        logger.warning("✗ 일부 테스트 실패")
        logger.warning("  위의 오류 메시지를 확인하고 OLLAMA_AUTO_START_GUIDE.md를 참조하세요.")
    logger.info("=" * 80)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
