# 4-Stage Chunk Selection Architecture Design
## POS Specification Value Extraction - v53 Enhancement

### 문제 정의 (Problem Statement)

**현재 문제:**
- Chunk 선택이 단순한 키워드 매칭으로 이루어짐
- 서로 다른 사양항목에 대해 동일한 잘못된 chunk가 반복적으로 선택됨
- 예: "Minimum cargo temperature | (-163OC)" → MAX/MIN SERVICE TEMPERATURE specs에 사용
- 예: "GENERAL section" → BILGE OUTLET specs에 사용
- 실제 올바른 값은 Section 2 TECHNICAL PARTICULARS에 존재하나 발견 못함
- 짧은 chunk (<50 chars)가 선택되어 LLM이 추출 불가능
- 74% LLM Fallback, 12% Rule success, 14% failure

**목표:**
- 정확한 section (Section 2 우선)에서 chunk 추출
- 다단계 검증으로 chunk 품질 보장
- 5-10초/spec 유지하면서 정확도 85-90% 달성
- POS 문서당 배치 처리로 전체 시간 단축

---

## 전체 아키텍처 (Overall Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. HTMLSectionParser                                                │
│    - HTML 섹션 구조 파싱                                             │
│    - Section 2 (TECHNICAL PARTICULARS) 식별                        │
│    - 섹션별 컨텐츠 인덱싱                                            │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. ChunkCandidateGenerator + ChunkQualityScorer                     │
│    - 다양한 소스에서 후보 chunk 생성                                 │
│      * Section 2 tables                                            │
│      * Section 2 text                                              │
│      * Keyword search results                                      │
│      * Hint-based search results                                   │
│    - 각 후보에 품질 점수 부여 (0-1.0)                                │
│      * 길이 검증 (100-3000 chars 선호)                              │
│      * 키워드 존재 여부                                              │
│      * 숫자 패턴 (숫자형 spec인 경우)                                │
│      * 테이블 구조 (structured data 선호)                            │
│      * 섹션 관련성 (Section 2 > Section 1)                          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. LLMChunkSelector                                                 │
│    - Top N 후보 (예: top 5)를 LLM에 제시                            │
│    - LLM이 가장 관련성 높은 chunk 선택                               │
│    - Confidence score 반환                                          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. ChunkExpander                                                    │
│    - 선택된 chunk가 너무 짧으면 확장 (<100 chars)                    │
│    - 주변 컨텍스트 추가 (±500 chars)                                 │
│    - 테이블 구조 보존                                                │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ↓
       [기존 RuleBasedExtractor / LLMFallbackExtractor]
```

---

## Stage 1: HTMLSectionParser

### 목적
- HTML 문서의 섹션 구조를 파싱하여 계층적으로 관리
- Section 2 (TECHNICAL PARTICULARS)를 우선적으로 식별
- 섹션별 인덱스 구축으로 빠른 조회 가능

### 클래스 설계

```python
@dataclass
class HTMLSection:
    """HTML 섹션 정보"""
    section_num: str           # "1", "2", "2.A", "2.2.1", etc.
    section_title: str         # "GENERAL", "TECHNICAL PARTICULARS", etc.
    section_level: int         # 1, 2, 3 (depth)
    start_pos: int            # HTML에서의 시작 위치
    end_pos: int              # HTML에서의 끝 위치
    content: str              # 섹션 전체 컨텐츠
    tables: List[List[List[str]]]  # 섹션 내 테이블들
    text_chunks: List[str]    # 섹션 내 텍스트 청크들
    subsections: List['HTMLSection']  # 하위 섹션들

class HTMLSectionParser:
    """
    HTML 문서를 섹션 단위로 파싱

    POS 문서는 일반적으로 다음과 같은 구조:
    1. GENERAL
    2. TECHNICAL PARTICULARS (또는 2.-A, 2.-B 등)
       2.1 Main particulars
       2.2 Detail specifications
    3. APPENDIX
    ...
    """

    def __init__(self, html_content: str = "", file_path: str = ""):
        self.html_content = html_content
        self.file_path = file_path
        self.soup = None
        self.sections: List[HTMLSection] = []
        self.section_index: Dict[str, HTMLSection] = {}  # 빠른 조회용

        if file_path and os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.html_content = f.read()

        if self.html_content:
            self._parse()

    def _parse(self):
        """HTML 파싱 및 섹션 추출"""
        self.soup = BeautifulSoup(self.html_content, 'html.parser')
        self._extract_sections()
        self._build_section_index()

    def _extract_sections(self):
        """
        섹션 추출 전략:
        1. <h1>, <h2>, <h3> 태그로 섹션 헤더 식별
        2. <div class="section" data-section="N"> 속성 활용
        3. 섹션 번호 패턴 인식: "1.", "2.-A", "2.2.1", etc.
        4. 각 섹션의 컨텐츠 범위 결정
        """
        pass  # 구현 예정

    def get_section_by_number(self, section_num: str) -> Optional[HTMLSection]:
        """섹션 번호로 조회 (예: "2", "2.A", "2.2.1")"""
        return self.section_index.get(section_num)

    def get_technical_sections(self) -> List[HTMLSection]:
        """
        Section 2 (TECHNICAL PARTICULARS) 및 하위 섹션 반환

        가장 중요한 메서드: 대부분의 spec은 여기서 찾아야 함
        """
        technical = []
        for section in self.sections:
            # Section 2로 시작하는 모든 섹션
            if section.section_num.startswith('2'):
                technical.append(section)
        return technical

    def search_in_section(
        self,
        section_num: str,
        keywords: List[str]
    ) -> List[str]:
        """특정 섹션 내에서 키워드 검색"""
        section = self.get_section_by_number(section_num)
        if not section:
            return []

        # 섹션 컨텐츠에서 키워드 포함 텍스트 추출
        matches = []
        for keyword in keywords:
            # 구현 예정
            pass

        return matches

    def get_section_tables(self, section_num: str) -> List[List[List[str]]]:
        """특정 섹션의 테이블들 반환"""
        section = self.get_section_by_number(section_num)
        if section:
            return section.tables
        return []
```

---

## Stage 2: ChunkCandidate & ChunkQualityScorer

### 목적
- 다양한 소스에서 chunk 후보 생성
- 각 후보의 품질을 정량적으로 평가
- Top N 후보만 다음 단계로 전달 (효율성)

### 클래스 설계

```python
@dataclass
class ChunkCandidate:
    """Chunk 후보 정보"""
    text: str                  # Chunk 텍스트
    source: str                # 출처: "section_2a_table", "section_2_text", "keyword_search", etc.
    section_num: str           # 소속 섹션 번호
    quality_score: float       # 품질 점수 (0-1.0)
    keywords_found: List[str]  # 발견된 키워드들
    has_numeric: bool          # 숫자 포함 여부
    is_table: bool             # 테이블 출처 여부
    start_pos: int             # 원본에서의 시작 위치 (확장용)
    end_pos: int               # 원본에서의 끝 위치 (확장용)
    metadata: Dict[str, Any]   # 추가 메타데이터

class ChunkCandidateGenerator:
    """
    다양한 소스에서 chunk 후보 생성

    생성 전략:
    1. Section 2 우선 검색
    2. Hint의 section_num 활용
    3. 사양명/장비명 키워드 검색
    4. 동의어 확장 검색
    5. 과거값 패턴 검색
    """

    def __init__(
        self,
        section_parser: HTMLSectionParser,
        chunk_parser: HTMLChunkParser,
        glossary: LightweightGlossaryIndex = None
    ):
        self.section_parser = section_parser
        self.chunk_parser = chunk_parser
        self.glossary = glossary

    def generate_candidates(
        self,
        spec: SpecItem,
        hint: ExtractionHint = None,
        max_candidates: int = 10
    ) -> List[ChunkCandidate]:
        """
        Spec에 대한 chunk 후보들 생성

        Returns:
            최대 max_candidates개의 후보 (중복 제거)
        """
        candidates = []
        seen_texts = set()  # 중복 방지

        # 1. Section 2 테이블에서 검색 (최우선)
        candidates.extend(
            self._search_in_technical_tables(spec, hint, seen_texts)
        )

        # 2. Hint의 section_num 활용
        if hint and hint.section_num:
            candidates.extend(
                self._search_in_hint_section(spec, hint, seen_texts)
            )

        # 3. 사양명 키워드 검색 (Section 2 우선)
        candidates.extend(
            self._keyword_search(spec, hint, seen_texts)
        )

        # 4. 동의어 확장 검색
        if self.glossary:
            candidates.extend(
                self._synonym_search(spec, hint, seen_texts)
            )

        # 5. 중복 제거 및 제한
        unique_candidates = self._deduplicate(candidates)
        return unique_candidates[:max_candidates]

    def _search_in_technical_tables(
        self,
        spec: SpecItem,
        hint: ExtractionHint,
        seen_texts: Set[str]
    ) -> List[ChunkCandidate]:
        """Section 2의 테이블에서 검색"""
        candidates = []
        technical_sections = self.section_parser.get_technical_sections()

        for section in technical_sections:
            for table in section.tables:
                # 테이블에서 spec_name 검색
                # 구현 예정
                pass

        return candidates

    # 기타 메서드들...

class ChunkQualityScorer:
    """
    Chunk 후보의 품질 평가

    평가 기준 (0-1.0 점수):
    1. 길이 적정성: 100-3000 chars 선호
    2. 키워드 존재: spec_name, equipment, synonyms
    3. 숫자 패턴: 숫자형 spec인 경우 숫자 포함 필수
    4. 테이블 구조: 테이블 출처 가산점
    5. 섹션 관련성: Section 2 가산점, Section 1 감점
    6. 컨텍스트 품질: 주변 텍스트 관련성
    """

    def __init__(
        self,
        glossary: LightweightGlossaryIndex = None
    ):
        self.glossary = glossary

    def score_candidate(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint = None
    ) -> float:
        """
        후보 chunk의 품질 점수 계산

        Returns:
            0.0-1.0 사이의 점수
        """
        score = 0.0

        # 1. 길이 점수 (최대 0.2)
        score += self._score_length(candidate)

        # 2. 키워드 점수 (최대 0.3)
        score += self._score_keywords(candidate, spec, hint)

        # 3. 숫자 패턴 점수 (최대 0.15)
        score += self._score_numeric_pattern(candidate, spec, hint)

        # 4. 구조 점수 (최대 0.15)
        score += self._score_structure(candidate, hint)

        # 5. 섹션 관련성 점수 (최대 0.2)
        score += self._score_section_relevance(candidate, hint)

        return min(1.0, score)

    def _score_length(self, candidate: ChunkCandidate) -> float:
        """
        길이 점수

        - 100-3000 chars: 0.2 (최적)
        - 50-100 chars: 0.1 (짧음)
        - < 50 chars: 0.0 (너무 짧음)
        - > 3000 chars: 0.15 (너무 김)
        """
        length = len(candidate.text)

        if 100 <= length <= 3000:
            return 0.2
        elif 50 <= length < 100:
            return 0.1
        elif length > 3000:
            return 0.15
        else:
            return 0.0

    def _score_keywords(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint
    ) -> float:
        """
        키워드 점수 (0-0.3)

        - spec_name 포함: +0.15
        - equipment 포함: +0.1
        - 동의어 포함: +0.05
        """
        score = 0.0
        text_upper = candidate.text.upper()

        # Spec name
        if spec.spec_name.upper() in text_upper:
            score += 0.15

        # Equipment
        if spec.equipment and spec.equipment.upper() in text_upper:
            score += 0.1

        # 동의어 (hint의 pos_umgv_desc)
        if hint and hint.pos_umgv_desc:
            if hint.pos_umgv_desc.upper() in text_upper:
                score += 0.05

        return score

    def _score_numeric_pattern(
        self,
        candidate: ChunkCandidate,
        spec: SpecItem,
        hint: ExtractionHint
    ) -> float:
        """
        숫자 패턴 점수 (0-0.15)

        숫자형 spec인데 숫자가 없으면 감점
        """
        value_format = hint.value_format if hint else ""
        is_numeric = is_numeric_spec(spec.spec_name, value_format)

        has_number = re.search(r'\d', candidate.text)

        if is_numeric:
            if has_number:
                return 0.15  # 숫자형 spec + 숫자 있음
            else:
                return 0.0   # 숫자형 spec + 숫자 없음 (치명적)
        else:
            # 텍스트형 spec
            if has_number:
                return 0.1   # 숫자 있어도 괜찮음
            else:
                return 0.1   # 숫자 없어도 괜찮음

    def _score_structure(
        self,
        candidate: ChunkCandidate,
        hint: ExtractionHint
    ) -> float:
        """
        구조 점수 (0-0.15)

        - 테이블 출처: +0.1
        - table_text hint 일치: +0.05
        """
        score = 0.0

        if candidate.is_table:
            score += 0.1

            # table_text hint와 일치하면 추가 점수
            if hint and hint.table_text and hint.table_text.upper() == "Y":
                score += 0.05

        return score

    def _score_section_relevance(
        self,
        candidate: ChunkCandidate,
        hint: ExtractionHint
    ) -> float:
        """
        섹션 관련성 점수 (0-0.2)

        - Section 2: +0.15 (가장 중요)
        - Hint section 일치: +0.05
        - Section 1 (GENERAL): -0.1 (감점)
        """
        score = 0.0
        section_num = candidate.section_num

        # Section 2 가산점
        if section_num and section_num.startswith('2'):
            score += 0.15

        # Section 1 감점
        if section_num and section_num.startswith('1'):
            score -= 0.1

        # Hint section 일치
        if hint and hint.section_num:
            hint_section = hint.section_num.split()[0]  # "2.2.1 Main ..." → "2.2.1"
            if section_num and section_num in hint_section:
                score += 0.05

        return max(0.0, score)
```

---

## Stage 3: LLMChunkSelector

### 목적
- Top N 후보 중 LLM이 가장 관련성 높은 chunk 선택
- 단순 점수 비교가 아닌 semantic 이해 기반 선택
- Confidence score 제공

### 클래스 설계

```python
class LLMChunkSelector:
    """
    LLM 기반 최적 chunk 선택

    Strategy:
    1. Top 3-5 후보를 LLM에 제시
    2. 각 후보에 번호 부여
    3. LLM이 가장 관련성 높은 번호 선택
    4. Confidence score 반환
    """

    def __init__(
        self,
        llm_client: 'UnifiedLLMClient',
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.log = logger or logging.getLogger(__name__)

    def select_best_chunk(
        self,
        candidates: List[ChunkCandidate],
        spec: SpecItem,
        hint: ExtractionHint = None,
        top_k: int = 5
    ) -> Optional[ChunkCandidate]:
        """
        최적 chunk 선택

        Args:
            candidates: 후보 리스트 (이미 quality score로 정렬됨)
            spec: 추출 대상 사양
            hint: 참조 힌트
            top_k: LLM에 제시할 후보 수

        Returns:
            선택된 chunk (없으면 None)
        """
        if not candidates:
            return None

        # 후보가 1개면 그대로 반환
        if len(candidates) == 1:
            return candidates[0]

        # Top K만 선택
        top_candidates = candidates[:top_k]

        # 프롬프트 생성
        prompt = self._build_selection_prompt(top_candidates, spec, hint)

        # LLM 호출
        try:
            response = self.llm_client.generate(prompt)
            selected_idx = self._parse_selection_response(response, len(top_candidates))

            if selected_idx is not None and 0 <= selected_idx < len(top_candidates):
                selected = top_candidates[selected_idx]
                self.log.debug(
                    "LLM selected chunk %d: source=%s, score=%.2f",
                    selected_idx, selected.source, selected.quality_score
                )
                return selected
            else:
                # 파싱 실패 시 최고 점수 후보 반환
                self.log.warning("LLM selection parse failed, using top score")
                return top_candidates[0]

        except Exception as e:
            self.log.warning("LLM selection error: %s, using top score", e)
            return top_candidates[0]

    def _build_selection_prompt(
        self,
        candidates: List[ChunkCandidate],
        spec: SpecItem,
        hint: ExtractionHint
    ) -> str:
        """선택 프롬프트 생성"""

        # 후보 목록 구성
        candidate_list = []
        for idx, cand in enumerate(candidates):
            preview = cand.text[:200] + "..." if len(cand.text) > 200 else cand.text
            candidate_list.append(
                f"[{idx}] (Section: {cand.section_num}, Source: {cand.source})\n{preview}\n"
            )

        candidates_text = "\n".join(candidate_list)

        # 힌트 정보
        hint_text = ""
        if hint:
            hint_parts = []
            if hint.historical_values:
                hint_parts.append(f"과거값 예시: {', '.join(hint.historical_values[:2])}")
            if hint.pos_umgv_desc:
                hint_parts.append(f"다른이름: {hint.pos_umgv_desc}")
            if hint.section_num:
                hint_parts.append(f"예상 섹션: {hint.section_num}")

            if hint_parts:
                hint_text = "\n힌트: " + ", ".join(hint_parts)

        prompt = f"""당신은 POS 문서에서 사양값을 추출하는 전문가입니다.
아래의 후보 chunk 중에서 "{spec.spec_name}" 사양값을 추출하기에 가장 적합한 chunk를 선택하세요.

**추출 대상:**
- 사양명: {spec.spec_name}
- 장비: {spec.equipment or '미지정'}
- 예상단위: {spec.expected_unit or '미지정'}{hint_text}

**후보 Chunks:**
{candidates_text}

**작업:**
위 후보 중 "{spec.spec_name}" 값을 추출하기에 가장 적합한 chunk의 번호를 선택하세요.
주의: Section 2 (TECHNICAL PARTICULARS)의 chunk가 일반적으로 가장 정확합니다.

**출력 형식:**
정확히 다음 형식으로만 응답하세요:
SELECTED: [번호]
CONFIDENCE: [0.0-1.0]
REASON: [선택 이유]

예:
SELECTED: 2
CONFIDENCE: 0.9
REASON: Section 2의 테이블에서 사양명과 장비가 모두 일치함"""

        return prompt

    def _parse_selection_response(
        self,
        response: str,
        num_candidates: int
    ) -> Optional[int]:
        """LLM 응답 파싱"""
        # "SELECTED: 2" 패턴 찾기
        match = re.search(r'SELECTED:\s*(\d+)', response, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < num_candidates:
                return idx

        return None
```

---

## Stage 4: ChunkExpander

### 목적
- 선택된 chunk가 너무 짧으면 주변 컨텍스트 추가
- 테이블 구조 보존
- 최대 크기 제한 준수

### 클래스 설계

```python
class ChunkExpander:
    """
    짧은 chunk를 주변 컨텍스트로 확장

    확장 전략:
    1. 길이 확인: < 100 chars이면 확장 필요
    2. 테이블 chunk: 전체 테이블 포함
    3. 텍스트 chunk: ±500 chars 추가
    4. 최대 크기: 5000 chars 제한
    """

    def __init__(
        self,
        section_parser: HTMLSectionParser,
        chunk_parser: HTMLChunkParser
    ):
        self.section_parser = section_parser
        self.chunk_parser = chunk_parser

    def expand_if_needed(
        self,
        chunk: str,
        candidate: ChunkCandidate,
        max_size: int = 5000
    ) -> str:
        """
        필요시 chunk 확장

        Args:
            chunk: 원본 chunk 텍스트
            candidate: Chunk 후보 정보 (start_pos, end_pos 등)
            max_size: 최대 크기

        Returns:
            확장된 chunk (또는 원본)
        """
        # 이미 충분히 긴 경우
        if len(chunk) >= 100:
            return chunk

        self.log.debug(
            "Expanding short chunk (%d chars) from %s",
            len(chunk), candidate.source
        )

        # 테이블 chunk인 경우: 전체 테이블 포함
        if candidate.is_table:
            expanded = self._expand_table_chunk(candidate)
        else:
            # 텍스트 chunk: 주변 컨텍스트 추가
            expanded = self._expand_text_chunk(candidate)

        # 크기 제한
        if len(expanded) > max_size:
            expanded = expanded[:max_size] + "...[truncated]"

        self.log.debug("Expanded to %d chars", len(expanded))
        return expanded

    def _expand_table_chunk(
        self,
        candidate: ChunkCandidate
    ) -> str:
        """테이블 chunk 확장: 전체 테이블 포함"""
        # 해당 섹션의 테이블들 가져오기
        section = self.section_parser.get_section_by_number(candidate.section_num)
        if not section:
            return candidate.text

        # 테이블 찾기 및 전체 컨텐츠 반환
        # 구현 예정
        return candidate.text

    def _expand_text_chunk(
        self,
        candidate: ChunkCandidate
    ) -> str:
        """텍스트 chunk 확장: 주변 ±500 chars 추가"""
        # 원본 HTML에서 start_pos, end_pos 활용
        # 구현 예정
        return candidate.text
```

---

## Integration: 기존 Extractor와 통합

### RuleBasedExtractor 통합

```python
class RuleBasedExtractor:
    # 기존 코드...

    def __init__(self, ...):
        # 기존 초기화...

        # 새로운 컴포넌트 추가
        self.section_parser = None  # Lazy loading
        self.candidate_generator = None
        self.quality_scorer = None
        self.chunk_expander = None

    def extract(self, parser: HTMLChunkParser, spec: SpecItem, hint: ExtractionHint = None):
        """
        개선된 추출 로직

        1. HTMLSectionParser로 섹션 파싱 (lazy)
        2. ChunkCandidateGenerator로 후보 생성
        3. ChunkQualityScorer로 점수 부여
        4. 최고 점수 chunk 선택
        5. ChunkExpander로 확장 (필요시)
        6. 기존 value 추출 로직 실행
        """

        # Lazy initialization
        if not self.section_parser:
            self._init_new_components(parser)

        # 1. 후보 생성
        candidates = self.candidate_generator.generate_candidates(spec, hint)

        if not candidates:
            # 후보 없으면 기존 방식 fallback
            return self._extract_legacy(parser, spec, hint)

        # 2. 품질 점수 계산
        for candidate in candidates:
            candidate.quality_score = self.quality_scorer.score_candidate(
                candidate, spec, hint
            )

        # 3. 점수순 정렬
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        # 4. 최고 점수 chunk 선택
        best_candidate = candidates[0]

        # 5. 필요시 확장
        expanded_chunk = self.chunk_expander.expand_if_needed(
            best_candidate.text, best_candidate
        )

        # 6. 값 추출 (기존 로직 재사용)
        # 구현 예정: expanded_chunk에서 값 추출

        return None  # 임시
```

### LLMFallbackExtractor 통합

```python
class LLMFallbackExtractor:
    # 기존 코드...

    def __init__(self, ...):
        # 기존 초기화...

        # 새로운 컴포넌트 추가
        self.section_parser = None
        self.candidate_generator = None
        self.quality_scorer = None
        self.llm_chunk_selector = None  # LLM 기반 선택
        self.chunk_expander = None

    def extract(self, parser: HTMLChunkParser, spec: SpecItem, hint: ExtractionHint = None):
        """
        개선된 LLM Fallback 추출

        1. HTMLSectionParser로 섹션 파싱
        2. ChunkCandidateGenerator로 후보 생성
        3. ChunkQualityScorer로 점수 부여
        4. LLMChunkSelector로 최적 chunk 선택 (multi-stage)
        5. ChunkExpander로 확장
        6. LLM으로 값 추출
        """

        # Lazy initialization
        if not self.section_parser:
            self._init_new_components(parser)

        # 1. 후보 생성
        candidates = self.candidate_generator.generate_candidates(spec, hint)

        if not candidates:
            # 후보 없으면 기존 방식
            return self._extract_legacy(parser, spec, hint)

        # 2. 품질 점수 계산
        for candidate in candidates:
            candidate.quality_score = self.quality_scorer.score_candidate(
                candidate, spec, hint
            )

        # 3. 점수순 정렬
        candidates.sort(key=lambda c: c.quality_score, reverse=True)

        # 4. LLM으로 최적 chunk 선택 (Top 5 중에서)
        best_candidate = self.llm_chunk_selector.select_best_chunk(
            candidates, spec, hint, top_k=5
        )

        if not best_candidate:
            best_candidate = candidates[0]

        # 5. 필요시 확장
        expanded_chunk = self.chunk_expander.expand_if_needed(
            best_candidate.text, best_candidate
        )

        # 6. LLM으로 값 추출 (기존 로직)
        prompt = self._build_prompt(spec, expanded_chunk, hint)
        response = self._call_ollama(prompt)

        if response:
            return self._parse_llm_response(response, spec, expanded_chunk)

        return None
```

---

## Batch Processing Integration

### extract_full 메서드 개선

```python
class POSExtractorV52:
    # 기존 코드...

    def extract_full(self, ...):
        """
        Full 모드 추출 (배치 처리 통합)

        개선:
        1. POS 문서별로 사양 그룹화
        2. 그룹당 extract_batch 호출 (15개씩)
        3. 전체 처리 시간 대폭 단축
        """

        # POS 문서별로 사양 그룹화
        pos_groups = self._group_specs_by_pos(all_specs)

        all_results = []

        for pos_file, specs_in_pos in pos_groups.items():
            logger.info(f"Processing {pos_file}: {len(specs_in_pos)} specs")

            # HTML 파싱
            parser = HTMLChunkParser(file_path=pos_file)

            # 배치 단위로 분할 (15개씩)
            batch_size = 15
            for i in range(0, len(specs_in_pos), batch_size):
                batch_specs = specs_in_pos[i:i+batch_size]

                # Rule 기반 시도
                rule_results = []
                for spec in batch_specs:
                    result = self.rule_extractor.extract(parser, spec)
                    rule_results.append(result)

                # 실패한 것들만 LLM Batch 처리
                failed_indices = [
                    i for i, r in enumerate(rule_results)
                    if not r or not r.value
                ]

                if failed_indices:
                    failed_specs = [batch_specs[i] for i in failed_indices]
                    llm_results = self.llm_extractor.extract_batch(
                        parser, failed_specs
                    )

                    # 결과 병합
                    for idx, llm_result in zip(failed_indices, llm_results):
                        rule_results[idx] = llm_result

                all_results.extend(rule_results)

        return all_results
```

---

## Performance Targets

### 목표 성능
- **정확도**: 85-90% (현재 ~70%)
- **속도**: 5-10 sec/spec (현재 5.2 sec/spec 유지)
- **배치 처리**: POS당 처리 시간 50% 단축
  - 예: 18 specs → 2 LLM calls (15+3) vs. 기존 18 calls
  - 약 70% LLM call 감소

### 예상 개선 효과
1. **정확도 향상**: 잘못된 chunk 선택 방지 → Rule 성공률 12% → 30%
2. **LLM Fallback 감소**: 74% → 50%
3. **실패율 감소**: 14% → 5%
4. **배치 효과**: 대량 추출 시 2-3배 속도 향상

---

## Implementation Plan

### Phase 1: Core Components (1-2시간)
1. HTMLSectionParser 구현
2. ChunkCandidate 데이터클래스
3. ChunkCandidateGenerator 구현
4. ChunkQualityScorer 구현

### Phase 2: Advanced Components (1시간)
5. LLMChunkSelector 구현
6. ChunkExpander 구현

### Phase 3: Integration (1-2시간)
7. RuleBasedExtractor 통합
8. LLMFallbackExtractor 통합
9. Batch processing 통합

### Phase 4: Testing (30분-1시간)
10. 실제 POS 데이터로 테스트
11. 성능 비교 및 디버깅
12. Commit & Push

---

## Notes

- 모든 컴포넌트는 lazy loading으로 초기화 (Light 모드 호환)
- 기존 코드 호환성 유지 (fallback 제공)
- 로깅 충분히 추가 (디버깅 용이)
- Confidence score 계산 정교화
- 실제 데이터 테스트 필수

---

**설계 완료. 이제 구현 단계로 진행합니다.**
