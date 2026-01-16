# Dynamic Learning Architecture for POS Extraction

## Overview

Human-in-the-loop 강화학습을 위한 동적 지식 베이스 시스템

```
┌─────────────────────────────────────────────────────────────────┐
│ Static Knowledge Base (정적 지식)                                 │
│ - glossary.xlsx: 용어집 (umgv_desc ↔ pos_umgv_desc)             │
│ - spec_db.xlsx: 사양값DB (과거 추출 결과)                         │
└────────────┬────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Pattern Learners (패턴 학습기) - 초기 학습 + 증분 학습           │
│ ├── SynonymLearner: umgv_desc 동의어 학습                        │
│ ├── UnitVariantLearner: umgv_uom 변형 학습                       │
│ ├── AbbreviationLearner: 약어 패턴 학습                          │
│ └── MatAttrContextLearner: mat_attr 맥락 패턴 학습               │
└────────────┬────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Dynamic Knowledge Stores (SQLite 기반 동적 저장소)                │
│ ├── synonyms.db                                                  │
│ │   └── Table: synonyms (standard, variant, source, confidence) │
│ ├── unit_variants.db                                             │
│ │   └── Table: variants (standard, variant, source, confidence) │
│ ├── abbreviations.db                                             │
│ │   └── Table: abbrevs (abbrev, full_form, source, confidence)  │
│ └── mat_attr_patterns.db                                         │
│     └── Table: patterns (mat_attr, pattern, source, confidence) │
└────────────┬────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Enhanced Extractors (동적 지식 활용)                              │
│ - 실시간 DB 조회 (캐시 활용)                                      │
│ - Fuzzy matching with learned synonyms                          │
│ - Unit normalization with learned variants                      │
│ - Context-aware disambiguation                                  │
└────────────┬────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Extraction Results + User Feedback                              │
│ - extraction_results.json: 추출 결과                             │
│ - user_feedback.json: 사용자 검증/수정 사항                       │
└────────────┬────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Feedback Loop & Auto-Update (자동 업데이트)                      │
│ 1. 피드백 분석: 정답/오답 패턴 추출                               │
│ 2. 용어집 업데이트: 새로운 pos_* 항목 추가                        │
│ 3. Knowledge Store 재학습: 증분 학습                             │
│ 4. 성능 모니터링: 정확도 추적                                     │
└─────────────────────────────────────────────────────────────────┘
             ↓ (Loop back to Pattern Learners)
```

## Data Flow

### 1. Initial Learning (초기 학습)

```bash
# 용어집과 사양값DB에서 패턴 학습
python -m knowledge_base.learners --mode=initial \
    --glossary=data/glossary.xlsx \
    --spec_db=data/spec_db.xlsx \
    --pos_documents=data/pos_html/
```

**학습 내용:**
- Synonyms: umgv_desc ↔ pos_umgv_desc 매핑 (용어집)
- Unit Variants: umgv_uom ↔ pos_umgv_uom 매핑 (용어집)
- Abbreviations: POS 문서 텍스트에서 약어 패턴 추출
- Mat Attr Patterns: 테이블 위 mat_attr_desc 패턴 추출

### 2. Extraction with Dynamic Knowledge (동적 지식 활용 추출)

```python
# v53_extractor.py에서 자동으로 동적 지식 로드
extractor = EnhancedLLMFallbackExtractor(
    use_dynamic_knowledge=True,
    knowledge_base_path="knowledge_base/data/"
)

result = extractor.extract(parser, spec, hint)
```

### 3. Incremental Learning (증분 학습)

```bash
# 새로운 추출 결과에서 학습
python -m knowledge_base.learners --mode=incremental \
    --extraction_results=results/extraction_results_20260116.json
```

### 4. User Feedback Integration (피드백 통합)

```bash
# 사용자 피드백 반영 → 용어집 업데이트 → 재학습
python -m knowledge_base.updaters --feedback=user_feedback.json \
    --update_glossary=True \
    --retrain=True
```

## Database Schemas

### synonyms.db

```sql
CREATE TABLE synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_term TEXT NOT NULL,       -- 표준 용어 (umgv_desc)
    variant_term TEXT NOT NULL,        -- 변형 용어 (pos_umgv_desc)
    source TEXT,                        -- 출처: 'glossary', 'pos_doc', 'feedback'
    confidence REAL DEFAULT 1.0,       -- 신뢰도 (0-1.0)
    frequency INTEGER DEFAULT 1,       -- 발견 빈도
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(standard_term, variant_term)
);

CREATE INDEX idx_variant ON synonyms(variant_term);
CREATE INDEX idx_standard ON synonyms(standard_term);
```

### unit_variants.db

```sql
CREATE TABLE variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    standard_unit TEXT NOT NULL,       -- 표준 단위 (°C)
    variant_unit TEXT NOT NULL,        -- 변형 단위 (OC, oc, degC)
    source TEXT,
    confidence REAL DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(standard_unit, variant_unit)
);

CREATE INDEX idx_variant_unit ON variants(variant_unit);
```

### abbreviations.db

```sql
CREATE TABLE abbrevs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    abbreviation TEXT NOT NULL,        -- 약어 (M/E)
    full_form TEXT NOT NULL,           -- 전체 형태 (Main Engine)
    context TEXT,                      -- 발견된 맥락
    source TEXT,
    confidence REAL DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(abbreviation, full_form)
);

CREATE INDEX idx_abbrev ON abbrevs(abbreviation);
```

### mat_attr_patterns.db

```sql
CREATE TABLE patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mat_attr_desc TEXT NOT NULL,       -- mat_attr_desc
    pattern_text TEXT NOT NULL,        -- 발견된 패턴 (예: "Main Engine Particulars")
    pattern_type TEXT,                 -- 패턴 유형: 'header', 'section_title', etc.
    distance_to_table INTEGER,         -- 테이블까지의 거리 (문자 수)
    source TEXT,
    confidence REAL DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mat_attr ON patterns(mat_attr_desc);
```

### extraction_history.db (추출 이력)

```sql
CREATE TABLE extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hull TEXT NOT NULL,
    umgv_desc TEXT NOT NULL,
    mat_attr_desc TEXT,
    extracted_value TEXT,
    extracted_unit TEXT,
    chunk_source TEXT,                 -- 어떤 chunk에서 추출했는지
    confidence REAL,
    extraction_method TEXT,            -- 'rule', 'llm', 'batch'
    user_feedback TEXT,                -- 'correct', 'incorrect', 'corrected'
    corrected_value TEXT,              -- 사용자가 수정한 값
    corrected_unit TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_spec ON extractions(hull, umgv_desc);
```

## Learning Strategies

### 1. Synonym Learning

**Sources:**
1. **Glossary (용어집)**: umgv_desc ↔ pos_umgv_desc 직접 매핑
2. **POS Documents**: 키워드 co-occurrence 분석
3. **Extraction History**: 성공적으로 매칭된 케이스
4. **User Feedback**: 사용자가 인정한 동의어

**Algorithm:**
```
For each glossary row:
    IF pos_umgv_desc != umgv_desc AND pos_umgv_desc is not empty:
        Add (umgv_desc, pos_umgv_desc, source='glossary', confidence=1.0)

For each POS document:
    Extract potential synonyms using NLP patterns:
        - "also known as", "AKA", "or"
        - Parenthetical expressions: "Main Engine (M/E)"
        - Table headers vs. section titles

For each successful extraction:
    IF matched_term != umgv_desc:
        Increment frequency of (umgv_desc, matched_term)
        Update confidence based on success rate
```

### 2. Unit Variant Learning

**Sources:**
1. **Glossary**: umgv_uom ↔ pos_umgv_uom
2. **POS Documents**: 정규 표현식으로 단위 패턴 추출
3. **Extraction History**: 추출된 단위들

**Algorithm:**
```
For each glossary row:
    IF pos_umgv_uom != umgv_uom AND pos_umgv_uom is not empty:
        normalized = normalize_unit(umgv_uom)
        Add (normalized, pos_umgv_uom, source='glossary', confidence=1.0)

For each POS document:
    Extract units after numbers:
        - Pattern: \d+\.?\d*\s*([A-Za-z°/]+)
        - Common variants: OC, oc, degC for °C
        - kw, KW, kilowatt for kW

Build normalization rules:
    Group similar variants → standard form
    Confidence = frequency / total_occurrences
```

### 3. Abbreviation Learning

**Sources:**
1. **POS Documents**: 약어 패턴 추출
2. **Glossary**: pos_umgv_desc에서 약어 감지

**Algorithm:**
```
Pattern 1: "Full Form (ABBREV)"
    Example: "Main Engine (M/E)" → M/E = Main Engine

Pattern 2: Slash-separated uppercase
    Example: "M/E", "G/E" → 주변 문맥에서 full form 추론

Pattern 3: All caps in headers
    Example: "MCR" in table header → "Maximum Continuous Rating"

For each pattern match:
    Add (abbreviation, full_form, context, confidence)
    Confidence based on:
        - Clarity of pattern (Pattern 1 = 1.0, Pattern 3 = 0.6)
        - Frequency of occurrence
```

### 4. Mat Attr Context Learning

**Sources:**
1. **POS Documents**: 테이블 위 텍스트 분석
2. **Glossary**: mat_attr_desc 매핑

**Algorithm:**
```
For each table in POS documents:
    Extract preceding text (up to 500 chars):
        - Section headers
        - Subsection titles
        - Descriptive paragraphs

    Identify mat_attr candidates:
        - Pattern: "[A-Z][a-z]+ (Particulars|Specifications|Details)"
        - Pattern: "\d+\.\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
        - Pattern: "^[A-Z ]{10,}$" (all caps headers)

    For each candidate:
        Calculate distance to table (chars)
        Store (mat_attr, pattern, distance, confidence)

    Confidence factors:
        - Distance < 100: +0.3
        - Distance < 300: +0.2
        - Pattern clarity: +0.5
        - Frequency: +log(freq)/10
```

## Feedback Loop

### User Feedback Format (user_feedback.json)

```json
{
  "feedback": [
    {
      "extraction_id": "12345",
      "hull": "2377",
      "umgv_desc": "OUTPUT",
      "mat_attr_desc": "MAIN ENGINE",
      "original_value": "150",
      "original_unit": "kW",
      "feedback_type": "correct",  // or "incorrect", "corrected"
      "corrected_value": null,
      "corrected_unit": null,
      "user_comment": "Correct extraction",
      "timestamp": "2026-01-16T10:30:00Z"
    },
    {
      "extraction_id": "12346",
      "hull": "2377",
      "umgv_desc": "MINIMUM OPERATING TEMPERATURE",
      "mat_attr_desc": "REFRIGERATION SYSTEM",
      "original_value": "-15~60",
      "original_unit": "OC",
      "feedback_type": "corrected",
      "corrected_value": "-15",
      "corrected_unit": "°C",
      "user_comment": "Should extract min value from range, fix unit",
      "timestamp": "2026-01-16T10:31:00Z"
    }
  ]
}
```

### Auto-Update Process

```python
def process_feedback(feedback_json):
    """
    1. Analyze feedback patterns
    2. Update glossary
    3. Update knowledge stores
    4. Retrain if needed
    """

    for item in feedback['feedback']:
        if item['feedback_type'] == 'correct':
            # Boost confidence of used patterns
            boost_confidence(item)

        elif item['feedback_type'] == 'corrected':
            # Learn from correction
            if item['corrected_value'] != item['original_value']:
                learn_value_extraction_pattern(item)

            if item['corrected_unit'] != item['original_unit']:
                # Add unit variant
                add_unit_variant(
                    standard=item['corrected_unit'],
                    variant=item['original_unit'],
                    source='user_feedback',
                    confidence=1.0
                )

        elif item['feedback_type'] == 'incorrect':
            # Reduce confidence of used patterns
            penalize_patterns(item)
```

### Glossary Auto-Update

```python
def update_glossary_from_feedback(feedback_json, glossary_path):
    """
    사용자 피드백을 기반으로 용어집 업데이트
    """

    new_entries = []

    for item in feedback['feedback']:
        if item['feedback_type'] == 'corrected':
            # Check if correction reveals new synonym
            if should_add_to_glossary(item):
                new_entry = {
                    'hull': item['hull'],
                    'umgv_desc': item['umgv_desc'],
                    'pos_umgv_desc': extract_matched_term(item),
                    'umgv_uom': item['corrected_unit'],
                    'pos_umgv_uom': item['original_unit'],
                    'mat_attr_desc': item['mat_attr_desc'],
                    'source': 'user_feedback',
                    'confidence': 1.0
                }
                new_entries.append(new_entry)

    # Append to glossary
    append_to_glossary(glossary_path, new_entries)

    # Trigger re-learning
    retrain_knowledge_base()
```

## Performance Monitoring

### Metrics to Track

```python
metrics = {
    'extraction_accuracy': {
        'total': 30000,
        'correct': 25500,
        'rate': 0.85
    },
    'rule_based_success_rate': 0.32,  # Target: 30% (vs 12% baseline)
    'llm_fallback_rate': 0.48,         # Target: 50% (vs 74% baseline)
    'failure_rate': 0.05,              # Target: 5% (vs 14% baseline)

    'knowledge_base_stats': {
        'synonym_count': 15000,
        'unit_variant_count': 500,
        'abbreviation_count': 200,
        'mat_attr_pattern_count': 300
    },

    'learning_progress': {
        'initial_accuracy': 0.70,
        'current_accuracy': 0.85,
        'improvement': 0.15,
        'feedback_cycles': 5
    }
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (P0)
- [ ] Create SQLite schemas
- [ ] Implement DynamicKnowledgeStore base class
- [ ] Implement basic learners (Synonym, Unit)

### Phase 2: Pattern Learning (P1)
- [ ] SynonymLearner from glossary
- [ ] UnitVariantLearner from glossary + POS docs
- [ ] AbbreviationLearner from POS docs
- [ ] MatAttrContextLearner from POS docs

### Phase 3: Integration (P1)
- [ ] Integrate dynamic stores into extractors
- [ ] Add caching layer for performance
- [ ] Test with 100 sample documents

### Phase 4: Feedback Loop (P2)
- [ ] Implement feedback analyzer
- [ ] Implement glossary auto-updater
- [ ] Implement incremental learning
- [ ] Create monitoring dashboard

### Phase 5: Scale Testing (P2)
- [ ] Test with 10,000 documents
- [ ] Performance optimization
- [ ] Production deployment

## File Structure

```
POSLLM/
├── v53_extractor.py (기존, 통합 예정)
├── knowledge_base/
│   ├── __init__.py
│   ├── learners.py          # Pattern learning logic
│   ├── stores.py            # Dynamic knowledge stores
│   ├── updaters.py          # Feedback processing
│   ├── schemas.sql          # DB schemas
│   └── data/
│       ├── synonyms.db
│       ├── unit_variants.db
│       ├── abbreviations.db
│       ├── mat_attr_patterns.db
│       └── extraction_history.db
├── scripts/
│   ├── initial_learning.py  # 초기 학습 스크립트
│   ├── incremental_learning.py
│   └── process_feedback.py
└── tests/
    └── test_dynamic_learning.py
```

## Usage Examples

### 1. Initial Setup

```bash
# 1. 초기 학습
python scripts/initial_learning.py \
    --glossary data/glossary.xlsx \
    --spec_db data/spec_db.xlsx \
    --pos_docs data/pos_html/ \
    --output knowledge_base/data/

# 2. 추출 실행 (자동으로 dynamic knowledge 사용)
python v53_extractor.py --input data/pos_html/hull_2377.html

# 3. 피드백 처리
python scripts/process_feedback.py \
    --feedback user_feedback.json \
    --update_glossary \
    --retrain
```

### 2. Programmatic Usage

```python
from knowledge_base.stores import DynamicSynonymStore, DynamicUnitStore

# 동의어 조회
synonym_store = DynamicSynonymStore()
variants = synonym_store.get_variants("OUTPUT")
# → ["POWER OUTPUT", "RATED OUTPUT", "M/E OUTPUT", ...]

# 단위 변형 조회
unit_store = DynamicUnitStore()
variants = unit_store.get_variants("°C")
# → ["OC", "oc", "degC", "deg C", "celsius", ...]

# 표준화
standard = unit_store.normalize("OC")
# → "°C"
```

---

**Next Steps**: Implement the core components starting with stores.py and learners.py
