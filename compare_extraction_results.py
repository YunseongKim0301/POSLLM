#!/usr/bin/env python3
"""
v61과 v70 추출 결과 비교

비교 항목:
1. 정확도 (found vs not_found)
2. 성능 (처리 시간)
3. LLM 호출 횟수
4. 신뢰도 점수 분포
"""

import json
import sys
from typing import Dict, List
from collections import Counter


def load_results(filepath: str) -> Dict:
    """결과 JSON 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_results(results: Dict, version_name: str) -> Dict:
    """결과 분석"""
    print(f"\n{'='*80}")
    print(f"{version_name} 결과 분석")
    print(f"{'='*80}")

    stats = {
        'total': 0,
        'found': 0,
        'not_found': 0,
        'extraction_failed': 0,
        'methods': Counter(),
        'avg_confidence': 0.0,
        'processing_time': 0.0
    }

    # results가 list인 경우
    if isinstance(results, dict) and 'results' in results:
        result_list = results['results']
        stats['processing_time'] = results.get('total_time', 0.0)
    elif isinstance(results, list):
        result_list = results
    else:
        print("✗ 결과 형식 오류")
        return stats

    stats['total'] = len(result_list)

    confidence_scores = []

    for item in result_list:
        # Method 카운트
        method = item.get('_method', 'unknown')
        stats['methods'][method] += 1

        # Found/Not found 카운트
        if item.get('pos_umgv_value'):
            stats['found'] += 1
        else:
            if method == 'EXTRACTION_FAILED' or method == 'FILE_NOT_FOUND':
                stats['extraction_failed'] += 1
            else:
                stats['not_found'] += 1

        # Confidence 점수
        conf = item.get('_confidence', 0.0)
        if conf > 0:
            confidence_scores.append(conf)

    if confidence_scores:
        stats['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)

    # 출력
    print(f"\n총 사양 수: {stats['total']}")
    print(f"  ✓ 추출 성공: {stats['found']} ({stats['found']/stats['total']*100:.1f}%)")
    print(f"  ✗ 추출 실패: {stats['extraction_failed']} ({stats['extraction_failed']/stats['total']*100:.1f}%)")
    print(f"  ? 값 없음: {stats['not_found']}")

    print(f"\n추출 방법별 분포:")
    for method, count in stats['methods'].most_common():
        print(f"  {method}: {count} ({count/stats['total']*100:.1f}%)")

    if confidence_scores:
        print(f"\n평균 신뢰도: {stats['avg_confidence']:.3f}")
        print(f"최소 신뢰도: {min(confidence_scores):.3f}")
        print(f"최대 신뢰도: {max(confidence_scores):.3f}")

    if stats['processing_time'] > 0:
        print(f"\n처리 시간: {stats['processing_time']:.1f}초")
        avg_time = stats['processing_time'] / stats['total']
        print(f"평균 시간/spec: {avg_time:.2f}초")
        total_300k = avg_time * 300000
        print(f"300K 추정 시간: {total_300k:.0f}초 = {total_300k/3600:.1f}시간 = {total_300k/86400:.1f}일")

    return stats


def compare_results(v61_results: Dict, v70_results: Dict):
    """v61과 v70 비교"""
    print(f"\n{'='*80}")
    print("v61 vs v70 비교")
    print(f"{'='*80}")

    v61_stats = analyze_results(v61_results, "v61")
    v70_stats = analyze_results(v70_results, "v70")

    print(f"\n{'='*80}")
    print("비교 요약")
    print(f"{'='*80}")

    # 정확도 비교
    v61_accuracy = v61_stats['found'] / v61_stats['total'] * 100 if v61_stats['total'] > 0 else 0
    v70_accuracy = v70_stats['found'] / v70_stats['total'] * 100 if v70_stats['total'] > 0 else 0

    print(f"\n정확도:")
    print(f"  v61: {v61_accuracy:.2f}%")
    print(f"  v70: {v70_accuracy:.2f}%")
    print(f"  차이: {v70_accuracy - v61_accuracy:+.2f}%")

    # 신뢰도 비교
    print(f"\n평균 신뢰도:")
    print(f"  v61: {v61_stats['avg_confidence']:.3f}")
    print(f"  v70: {v70_stats['avg_confidence']:.3f}")
    print(f"  차이: {v70_stats['avg_confidence'] - v61_stats['avg_confidence']:+.3f}")

    # 성능 비교
    if v61_stats['processing_time'] > 0 and v70_stats['processing_time'] > 0:
        v61_avg = v61_stats['processing_time'] / v61_stats['total']
        v70_avg = v70_stats['processing_time'] / v70_stats['total']

        print(f"\n평균 시간/spec:")
        print(f"  v61: {v61_avg:.2f}초")
        print(f"  v70: {v70_avg:.2f}초")
        print(f"  차이: {v70_avg - v61_avg:+.2f}초 ({(v70_avg/v61_avg - 1)*100:+.1f}%)")

        # 300K 추정
        v61_300k = v61_avg * 300000 / 86400  # days
        v70_300k = v70_avg * 300000 / 86400  # days

        print(f"\n300K 추정 시간:")
        print(f"  v61: {v61_300k:.1f}일")
        print(f"  v70: {v70_300k:.1f}일")
        print(f"  목표 (2-3일): {'✓' if v70_300k <= 3 else '✗'}")

    # Method 비교
    print(f"\n추출 방법 비교:")
    all_methods = set(v61_stats['methods'].keys()) | set(v70_stats['methods'].keys())
    for method in sorted(all_methods):
        v61_count = v61_stats['methods'].get(method, 0)
        v70_count = v70_stats['methods'].get(method, 0)
        v61_pct = v61_count / v61_stats['total'] * 100 if v61_stats['total'] > 0 else 0
        v70_pct = v70_count / v70_stats['total'] * 100 if v70_stats['total'] > 0 else 0

        print(f"  {method}:")
        print(f"    v61: {v61_count} ({v61_pct:.1f}%)")
        print(f"    v70: {v70_count} ({v70_pct:.1f}%)")


def find_differences(v61_results: Dict, v70_results: Dict):
    """두 버전에서 다르게 추출된 사양 찾기"""
    print(f"\n{'='*80}")
    print("차이점 분석")
    print(f"{'='*80}")

    # results 리스트 추출
    if isinstance(v61_results, dict) and 'results' in v61_results:
        v61_list = v61_results['results']
    elif isinstance(v61_results, list):
        v61_list = v61_results
    else:
        print("✗ v61 결과 형식 오류")
        return

    if isinstance(v70_results, dict) and 'results' in v70_results:
        v70_list = v70_results['results']
    elif isinstance(v70_results, list):
        v70_list = v70_results
    else:
        print("✗ v70 결과 형식 오류")
        return

    # umgv_code를 키로 매핑
    v61_map = {item['umgv_code']: item for item in v61_list}
    v70_map = {item['umgv_code']: item for item in v70_list}

    differences = []

    for code in v61_map.keys():
        if code not in v70_map:
            continue

        v61_item = v61_map[code]
        v70_item = v70_map[code]

        v61_value = v61_item.get('pos_umgv_value', '')
        v70_value = v70_item.get('pos_umgv_value', '')

        # 값이 다른 경우
        if v61_value != v70_value:
            differences.append({
                'code': code,
                'name': v61_item.get('umgv_desc', ''),
                'v61_value': v61_value,
                'v70_value': v70_value,
                'v61_method': v61_item.get('_method', ''),
                'v70_method': v70_item.get('_method', ''),
                'v61_confidence': v61_item.get('_confidence', 0),
                'v70_confidence': v70_item.get('_confidence', 0)
            })

    if differences:
        print(f"\n차이점 {len(differences)}개 발견:")
        print()

        for i, diff in enumerate(differences[:20], 1):  # 최대 20개만 표시
            print(f"{i}. {diff['code']}: {diff['name']}")
            print(f"   v61: '{diff['v61_value']}' ({diff['v61_method']}, conf={diff['v61_confidence']:.2f})")
            print(f"   v70: '{diff['v70_value']}' ({diff['v70_method']}, conf={diff['v70_confidence']:.2f})")
            print()

        if len(differences) > 20:
            print(f"   ... 외 {len(differences) - 20}개")
    else:
        print("\n✓ 모든 사양의 추출 결과가 동일합니다")


def main():
    """메인 함수"""
    if len(sys.argv) != 3:
        print("사용법: python compare_extraction_results.py <v61_result.json> <v70_result.json>")
        sys.exit(1)

    v61_path = sys.argv[1]
    v70_path = sys.argv[2]

    print("="*80)
    print("추출 결과 비교")
    print("="*80)
    print(f"\nv61 결과: {v61_path}")
    print(f"v70 결과: {v70_path}")

    # 로드
    try:
        v61_results = load_results(v61_path)
        v70_results = load_results(v70_path)
    except FileNotFoundError as e:
        print(f"\n✗ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n✗ JSON 파싱 오류: {e}")
        sys.exit(1)

    # 비교
    compare_results(v61_results, v70_results)

    # 차이점 분석
    find_differences(v61_results, v70_results)

    print(f"\n{'='*80}")
    print("비교 완료")
    print("="*80)


if __name__ == "__main__":
    main()
