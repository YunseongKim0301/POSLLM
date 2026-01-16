# PostgreSQL 기반 동적 지식 시스템 마이그레이션 가이드

## 개요

파일 기반(glossary.xlsx, spec_db.xlsx) 시스템을 PostgreSQL 전용 시스템으로 전환합니다.

보안망 환경을 고려하여:
- ❌ SQLite/JSON 캐시 파일 생성 (보안 이슈 가능)
- ✅ 인메모리 dict 캐시 (프로그램 시작 시 1회 로드)

---

## 1. 구현 완료 사항

### ✅ `PostgresKnowledgeLoader` 클래스 (v53_extractor.py: L2641~)

**기능**:
- PostgreSQL `pos_dict`, `umgv_fin` 테이블에서 동적 지식 로드
- 인메모리 dict로 O(1) 조회
- 프로세스 종료 시 자동 삭제 (보안 안전)

**메서드**:
```python
loader = PostgresKnowledgeLoader(config=config)
loader.load_all()  # 1회 호출, 수초 소요

# 동의어 조회
synonyms = loader.get_synonyms("OUTPUT")  # ["POWER OUTPUT", "RATED OUTPUT", ...]

# 단위 정규화
standard = loader.normalize_unit("OC")  # "°C"

# 약어 확장
expansions = loader.get_abbreviation_expansions("M/E OUTPUT")
# ["M/E OUTPUT", "Main Engine OUTPUT", "Marine Engine OUTPUT"]
```

### ✅ 래퍼 클래스 업데이트

- **`UnitNormalizer`**: PostgresKnowledgeLoader 래핑
- **`FuzzyMatcher`**: PostgresKnowledgeLoader 래핑
- **`AbbreviationExpander`**: PostgresKnowledgeLoader 래핑

기존 인터페이스 유지하면서 PostgreSQL 기반으로 전환.

---

## 2. 필요한 추가 작업

### TODO 1: POSExtractorV52 초기화 수정

**위치**: `v53_extractor.py` - `POSExtractorV52.__init__()` 및 `_init_light_mode()`

**수정 내용**:

```python
def _init_light_mode(self, glossary_path: str, specdb_path: str):
    """Light 모드 초기화"""
    start = time.time()

    # === DB 모드 전용 (파일 모드 제거) ===
    if self.config.data_source_mode != "db":
        self.log.error("v53 Enhanced는 DB 모드만 지원합니다.")
        raise RuntimeError("파일 모드는 더 이상 지원되지 않습니다. data_source_mode='db'로 설정하세요.")

    self.log.info("데이터 소스: DB 모드 (PostgreSQL)")

    # PostgreSQL 연결
    try:
        self.pg_loader = PostgresEmbeddingLoader(self.config, self.log)
    except Exception as e:
        self.log.error(f"PostgreSQL 연결 실패: {e}")
        raise RuntimeError(f"PostgreSQL 연결 필수: {e}")

    # 용어집 로드 (pos_dict)
    glossary_df = self.pg_loader.load_glossary_from_db()
    if glossary_df.empty:
        raise RuntimeError("용어집(pos_dict) 로드 실패")
    self.glossary = LightweightGlossaryIndex(df=glossary_df)

    # 사양값DB 로드 (umgv_fin)
    specdb_df = self.pg_loader.load_specdb_from_db()
    if specdb_df.empty:
        raise RuntimeError("사양값DB(umgv_fin) 로드 실패")
    self.specdb = LightweightSpecDBIndex(df=specdb_df)

    # === 동적 지식 로더 초기화 (v53 Enhanced) ===
    self.pg_knowledge_loader = PostgresKnowledgeLoader(
        conn=self.pg_loader.conn,  # 기존 연결 재사용
        logger=self.log
    )

    # 지식 로드 (1회, 수초 소요)
    if not self.pg_knowledge_loader.load_all():
        self.log.warning("동적 지식 로드 실패, 기본 기능만 사용")

    # ReferenceHintEngine 초기화
    self.hint_engine = ReferenceHintEngine(
        glossary=self.glossary,
        specdb=self.specdb,
        pg_loader=self.pg_loader,
        logger=self.log
    )

    elapsed = time.time() - start
    self.log.info(f"Light 모드 초기화 완료: {elapsed:.2f}초")
```

### TODO 2: LLMFallbackExtractor에 PostgresKnowledgeLoader 전달

**위치**: `v53_extractor.py` - `POSExtractorV52.__init__()`의 LLMFallbackExtractor 초기화 부분

**수정 내용**:

```python
# LLM Fallback 초기화
self.llm_fallback = None
if self.config.use_llm and self.config.enable_llm_fallback:
    self.llm_fallback = LLMFallbackExtractor(
        ollama_host=self.config.ollama_host,
        ollama_ports=self.config.ollama_ports,
        model=self.config.ollama_model,
        timeout=self.config.ollama_timeout,
        logger=self.log,
        llm_client=self.llm_client,
        use_voting=self.config.vote_enabled,
        glossary=self.glossary,
        enable_enhanced_chunk_selection=True,
        use_dynamic_knowledge=True  # v53 Enhanced
    )

    # PostgresKnowledgeLoader 주입
    self.llm_fallback.pg_knowledge_loader = self.pg_knowledge_loader

    # UnitNormalizer 초기화 (후처리용)
    if self.pg_knowledge_loader:
        self.llm_fallback.unit_normalizer = UnitNormalizer(self.pg_knowledge_loader)
```

### TODO 3: ChunkCandidateGenerator에 PostgresKnowledgeLoader 전달

**위치**: `v53_extractor.py` - `LLMFallbackExtractor._init_enhanced_components()`

**수정 내용**:

```python
def _init_enhanced_components(self, parser: HTMLChunkParser):
    """Enhanced chunk selection 컴포넌트 초기화"""
    if self._enhanced_components_initialized:
        return

    try:
        # HTMLSectionParser 생성
        self.section_parser = HTMLSectionParser(
            html_content=parser.html_content,
            file_path=parser.file_path,
            chunk_parser=parser
        )

        # ChunkCandidateGenerator 생성 (PostgresKnowledgeLoader 전달)
        self.candidate_generator = ChunkCandidateGenerator(
            section_parser=self.section_parser,
            chunk_parser=parser,
            glossary=self.glossary,
            logger=self.log,
            use_dynamic_knowledge=True,
            pg_knowledge_loader=self.pg_knowledge_loader  # v53 Enhanced
        )

        # ChunkQualityScorer 생성
        self.quality_scorer = ChunkQualityScorer(
            glossary=self.glossary,
            logger=self.log
        )

        # LLMChunkSelector 생성
        if self.llm_client:
            self.llm_chunk_selector = LLMChunkSelector(
                llm_client=self.llm_client,
                logger=self.log
            )

        # ChunkExpander 생성
        self.chunk_expander = ChunkExpander(
            section_parser=self.section_parser,
            chunk_parser=parser,
            logger=self.log
        )

        self._enhanced_components_initialized = True
        self.log.debug("Enhanced components initialized (PostgreSQL-based)")

    except Exception as e:
        self.log.warning(f"Failed to initialize enhanced components: {e}")
        self.enable_enhanced_chunk_selection = False
```

---

## 3. 성능 테스트

### 테스트 시나리오

```python
import time
from v53_extractor import Config, POSExtractorV52

# 설정
config = Config()
config.data_source_mode = "db"
config.db_host = "your_db_host"
config.db_port = 5432
config.db_name = "your_db"
config.db_user = "your_user"
config.db_password = "your_password"

# 초기화 시간 측정
start = time.time()
extractor = POSExtractorV52(config=config)
init_time = time.time() - start

print(f"초기화 시간: {init_time:.2f}초")
print(f"동의어 개수: {len(extractor.pg_knowledge_loader.synonym_reverse)}")
print(f"단위 변형 개수: {len(extractor.pg_knowledge_loader.unit_reverse)}")
print(f"약어 개수: {len(extractor.pg_knowledge_loader.abbreviations)}")

# 추출 성능 테스트
spec = SpecItem(hull="2377", spec_name="OUTPUT", equipment="MAIN ENGINE")

start = time.time()
result = extractor.extract_single("test.html", spec)
extract_time = time.time() - start

print(f"추출 시간: {extract_time:.2f}초")
print(f"추출 결과: {result['value']} {result['unit']}")
```

**예상 결과**:
- 초기화 시간: 3-5초 (PostgreSQL 쿼리 + 인메모리 캐싱)
- 추출 시간: 0.5-10초 (Rule-based 0.5초, LLM 10초)

---

## 4. 병렬 처리 최적화 (30만 사양, 2-3일 목표)

### 현재 성능 분석

**목표**:
- 30만 사양 ÷ 2일 = 15만/일 = 1.74개/초 처리 필요

**최적화 전략**:

#### 1. 프로세스 병렬화 (ProcessPoolExecutor)

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_spec_batch(spec_batch, config):
    """단일 프로세스에서 배치 처리"""
    extractor = POSExtractorV52(config=config)
    results = []
    for spec in spec_batch:
        result = extractor.extract_single(spec.html_path, spec)
        results.append(result)
    return results

# 메인 프로세스
if __name__ == "__main__":
    num_workers = 10  # 10개 프로세스 병렬

    # 30만 사양을 10개로 나눔
    all_specs = load_all_specs()  # 30만개
    batch_size = len(all_specs) // num_workers

    batches = [all_specs[i:i+batch_size] for i in range(0, len(all_specs), batch_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_spec_batch, batch, config) for batch in batches]

        for future in as_completed(futures):
            batch_results = future.result()
            save_results(batch_results)
```

#### 2. LLM Batch 처리 활용

```python
# LLMFallbackExtractor.extract_batch() 활용
# 15개 사양을 1회 LLM 호출로 처리
# → LLM 호출 횟수 1/15로 감소
```

**예상 성능**:
- Rule-based 성공률: 32% (0.5초/사양) → 즉시 처리
- LLM Fallback: 68%
  - Batch 처리: 15개/15초 = 1초/사양
- 평균: 0.32 × 0.5 + 0.68 × 1 = **0.84초/사양**
- 병렬 10 프로세스: **8.4개/초 처리 가능** ✅

**계산**:
- 30만 사양 ÷ 8.4개/초 = 35,714초 = 9.9시간 ✅ (목표: 2일)

---

## 5. 보안망 고려사항

### ✅ 안전한 설계
1. **파일 생성 없음**: SQLite/JSON 캐시 파일 생성 제거
2. **인메모리 전용**: Python dict만 사용
3. **프로세스 종료 시 자동 삭제**: 메모리 해제로 데이터 자동 삭제

### ⚠️ 주의사항
1. **PostgreSQL 연결 필수**: DB 모드만 지원
2. **메모리 사용량**: 동의어/단위 캐시가 메모리에 상주 (예상: 10-50MB)
3. **재시작 시 재로드**: 프로세스 재시작마다 PostgreSQL에서 재로드 (3-5초)

---

## 6. 다음 단계

### 즉시 작업

1. ✅ PostgresKnowledgeLoader 구현 완료
2. ✅ 래퍼 클래스 업데이트 완료
3. ⏳ POSExtractorV52 통합 (TODO 1-3)
4. ⏳ 성능 테스트 실행
5. ⏳ 병렬 처리 스크립트 작성

### 장기 계획

1. Human-in-the-loop 피드백 수집
2. pos_dict 테이블에 피드백 반영
3. 자동 재학습 (프로세스 재시작 시)

---

## 7. FAQ

**Q: 파일 모드는 완전히 제거되나요?**
A: 네, v53 Enhanced는 DB 모드만 지원합니다. `data_source_mode='db'` 필수.

**Q: 기존 JSON 캐시는 어떻게 되나요?**
A: 더 이상 사용되지 않습니다. `knowledge_base/data/` 폴더는 삭제 가능합니다.

**Q: PostgreSQL 연결이 실패하면?**
A: 프로그램이 시작되지 않습니다. DB 연결은 필수입니다.

**Q: 성능이 파일 모드보다 느리지 않나요?**
A: 초기화는 느리지만(파일: 1초, DB: 3-5초), 추출 성능은 동일하거나 더 빠릅니다 (인메모리 O(1) 조회).

---

## 8. 참고 자료

- [DYNAMIC_LEARNING_ARCHITECTURE.md](DYNAMIC_LEARNING_ARCHITECTURE.md): 전체 아키텍처
- [USAGE_GUIDE.md](USAGE_GUIDE.md): 사용 가이드 (파일 기반, 업데이트 필요)
- [v53_extractor.py](v53_extractor.py): 구현 코드
