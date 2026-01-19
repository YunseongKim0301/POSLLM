#!/usr/bin/env python3
"""
v76 extractor 테스트 스크립트
"""
import sys
import os

# v76_extractor를 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v76_extractor import POSExtractor, Config

def main():
    # Config 설정 (LIGHT 모드)
    config = Config()
    config.extract_mode = "light"
    config.data_source_mode = "db"
    config.output_json = True
    config.output_csv = False
    config.output_db = False

    # PostgreSQL 연결 정보 (로그에서 확인)
    config.postgres_host = "10.131.132.116"
    config.postgres_port = 5432
    config.postgres_db = "managesys"
    config.postgres_user = "postgres"  # 기본값
    config.postgres_password = ""  # 환경변수에서 가져올 것으로 예상

    # Extractor 초기화
    print("=" * 70)
    print("v76 Extractor 테스트 시작")
    print("=" * 70)

    extractor = POSExtractor(config=config)

    # 샘플 파일 경로
    sample_file = "2530-POS-0033101_001_00_A4{12}.html"

    if not os.path.exists(sample_file):
        print(f"ERROR: 파일을 찾을 수 없습니다: {sample_file}")
        return

    # 추출 실행 (소량 모드)
    results = extractor.extract_light([sample_file])

    # 결과 출력
    print("\n" + "=" * 70)
    print("추출 결과")
    print("=" * 70)

    import json
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)

if __name__ == "__main__":
    main()
