#!/usr/bin/env python3
"""
v61_extractor.py 소규모 테스트 (100개 샘플)

전체 추출 프로세스(Rule + LLM) 테스트
"""

import os
import sys
import json
import logging
from datetime import datetime

# v61_extractor 임포트
sys.path.insert(0, os.path.dirname(__file__))
from v61_extractor import POSExtractorV61, build_config, SpecItem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_samples(limit=100):
    """PostgreSQL에서 테스트 샘플 로드"""
    import psycopg2

    conn = psycopg2.connect(
        host="10.131.132.116",
        port=5432,
        dbname="managesys",
        user="postgres",
        password="pmg_umg!@"
    )

    cur = conn.cursor()

    # ext_tmpl에서 샘플 추출 (hull별로 다양하게)
    query = """
        SELECT DISTINCT
            t.matnr,
            t.file_name,
            t.umgv_desc,
            t.umgv_code,
            t.umgv_uom,
            u.umgv_value as ground_truth_value,
            u.pos_umgv_desc as ground_truth_pos_desc,
            u.pos_chunk as ground_truth_chunk
        FROM ext_tmpl t
        LEFT JOIN umgv_fin u ON t.matnr = u.matnr AND t.umgv_code = u.umgv_code
        WHERE t.file_name IS NOT NULL
          AND t.file_name != ''
          AND t.umgv_desc IS NOT NULL
          AND u.umgv_value IS NOT NULL
          AND u.umgv_value != ''
        ORDER BY t.matnr, t.umgv_code
        LIMIT %s
    """

    cur.execute(query, (limit,))
    rows = cur.fetchall()

    samples = []
    for row in rows:
        samples.append({
            'matnr': row[0],
            'file_name': row[1],
            'umgv_desc': row[2],
            'umgv_code': row[3],
            'umgv_uom': row[4],
            'ground_truth_value': row[5],
            'ground_truth_pos_desc': row[6],
            'ground_truth_chunk': row[7]
        })

    cur.close()
    conn.close()

    return samples


def extract_hull_from_matnr(matnr: str) -> str:
    """matnr에서 hull 추출"""
    if not matnr or len(matnr) < 10:
        return ""
    return matnr[:10]


def evaluate_result(extracted, ground_truth):
    """추출 결과 평가"""
    issues = []

    # 1. pos_umgv_desc 체크: umgv_desc가 그대로 들어갔는지
    if extracted.get('pos_umgv_desc') == ground_truth.get('umgv_desc'):
        issues.append("pos_umgv_desc_equals_umgv_desc")

    # 2. pos_chunk 길이 체크: 너무 짧은지 (50자 미만)
    pos_chunk = extracted.get('pos_chunk', '')
    if pos_chunk and len(pos_chunk) < 50:
        issues.append("pos_chunk_too_short")

    # 3. pos_chunk에 맥락이 있는지 (구분자만 있고 문장이 없는지)
    if pos_chunk:
        # |나 ()만 많고 실제 단어가 적으면 문제
        delimiter_count = pos_chunk.count('|') + pos_chunk.count('(') + pos_chunk.count(')')
        word_count = len([w for w in pos_chunk.split() if len(w) >= 3])
        if delimiter_count > word_count:
            issues.append("pos_chunk_delimiter_heavy")

    # 4. 값 정확도 체크
    extracted_value = extracted.get('pos_umgv_value', '')
    gt_value = ground_truth.get('ground_truth_value', '')

    value_match = False
    if extracted_value and gt_value:
        # 숫자 추출해서 비교
        import re
        ext_nums = re.findall(r'\d+\.?\d*', extracted_value)
        gt_nums = re.findall(r'\d+\.?\d*', gt_value)

        if ext_nums and gt_nums:
            # 첫 번째 숫자가 같으면 매칭
            value_match = (ext_nums[0] == gt_nums[0])

    return {
        'value_match': value_match,
        'issues': issues
    }


def run_test():
    """테스트 실행"""
    logger.info("=" * 80)
    logger.info("v61 소규모 테스트 시작 (100개 샘플)")
    logger.info("=" * 80)

    # 샘플 로드
    logger.info("테스트 샘플 로드 중...")
    samples = load_test_samples(limit=100)
    logger.info(f"테스트 샘플: {len(samples)}개")

    if not samples:
        logger.error("테스트 샘플이 없습니다.")
        return

    # 추출기 초기화
    logger.info("POSExtractorV61 초기화 중...")
    config = build_config()
    extractor = POSExtractorV61(config=config)

    # 테스트 실행
    results = []
    issue_counter = {}
    matched_count = 0

    logger.info("추출 시작...")
    start_time = datetime.now()

    for idx, sample in enumerate(samples, 1):
        # HTML 경로
        html_path = os.path.join("/workspace/server/uploaded_files", sample['file_name'])

        if not os.path.exists(html_path):
            logger.warning(f"[{idx}/{len(samples)}] 파일 없음: {sample['file_name']}")
            continue

        # SpecItem 생성
        spec = SpecItem(
            spec_name=sample['umgv_desc'],
            spec_code=sample['umgv_code'],
            expected_unit=sample['umgv_uom'],
            hull=extract_hull_from_matnr(sample['matnr'])
        )

        # 추출
        logger.info(f"[{idx}/{len(samples)}] 추출: {sample['umgv_desc']}")
        extracted = extractor.extract_single(html_path, spec)

        # 평가
        evaluation = evaluate_result(extracted, sample)

        if evaluation['value_match']:
            matched_count += 1

        for issue in evaluation['issues']:
            issue_counter[issue] = issue_counter.get(issue, 0) + 1

        results.append({
            'sample': sample,
            'extracted': extracted,
            'evaluation': evaluation
        })

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # 결과 요약
    logger.info("=" * 80)
    logger.info("테스트 결과")
    logger.info("=" * 80)
    logger.info(f"총 샘플: {len(results)}개")
    logger.info(f"값 매칭: {matched_count}개 ({matched_count/len(results)*100:.1f}%)")
    logger.info(f"실행 시간: {elapsed:.1f}초 ({elapsed/len(results):.1f}초/개)")
    logger.info("")
    logger.info("발견된 문제점:")
    for issue, count in sorted(issue_counter.items(), key=lambda x: -x[1]):
        logger.info(f"  - {issue}: {count}건 ({count/len(results)*100:.1f}%)")

    # 결과 저장
    output_path = "/home/user/POSLLM/test_v61_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(results),
                'matched': matched_count,
                'accuracy': matched_count / len(results) * 100,
                'elapsed_seconds': elapsed,
                'seconds_per_item': elapsed / len(results),
                'issues': issue_counter
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n결과 저장: {output_path}")

    # 정확도가 90% 미만이면 개선 필요
    accuracy = matched_count / len(results) * 100
    if accuracy < 90.0:
        logger.warning(f"\n⚠️ 정확도 {accuracy:.1f}% < 90% - 개선이 필요합니다")
        return False
    else:
        logger.info(f"\n✓ 정확도 {accuracy:.1f}% ≥ 90% - 목표 달성!")
        return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
