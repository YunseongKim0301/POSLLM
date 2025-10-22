# -*- coding: utf-8 -*-
"""
POS Spec 추출/검증 파이프라인 — 합의(Consensus) + 테이블 인지 + 값-백링크 + LLM 판사 모드 + spaCy IE
- Jupyter / VSCode ipynb 친화: __main__에서 자동 실행 안 함, build_config_for_notebook 제공
- 라벨 느슨 매칭: RapidFuzz token_set_ratio / partial_ratio
- 테이블 인지: pandas.read_html + BeautifulSoup (lxml 필요)
- 값-우선 백링크: 입력값으로 문서에서 역추적하여 라벨 근접 탐색
- 단위 변환 수용: bar↔MPa, mm↔inch 등 물리량 차원 인식과 오차 허용
- LLM 판사(entailment) 모드: 후보 라벨-스펙 의미 일치 여부를 이항 판단으로만 조회(가점/감점)
- 합의 엔진 확장: 방법 다중 일치 보너스, 호환(범위/±/스케일) 보너스, 테이블 근접 보너스, LLM-판사 보너스, spaCy IE 보너스
- 파일 선택 개선: 호선/자재속성그룹명/파일명 토큰 유사도로 스코어링
- spaCy IE: 문장/토큰 단위로 라벨 근처에서 숫자+단위 추출(모델 부재 시 blank('xx') + EntityRuler로 폴백)
"""

from __future__ import annotations
import os, re, sys, json, logging, pickle, argparse, math, unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# ========== 필수 외부 의존 ========== #
# pandas
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas가 필요합니다. `pip install pandas`로 설치하세요.") from e

# tqdm (없어도 동작하도록 대체)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(total or 0)

# BeautifulSoup + lxml (테이블 파싱용)
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception as e:
    raise RuntimeError("BeautifulSoup4가 필요합니다. `pip install beautifulsoup4`로 설치하세요.") from e

try:
    import lxml  # noqa: F401
    _HAVE_LXML = True
except Exception:
    _HAVE_LXML = False

# RapidFuzz (느슨한/의미 유사도)
try:
    from rapidfuzz import fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

# Ollama Python (선택)
try:
    import ollama  # type: ignore
    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False

# spaCy (선택)
try:
    import spacy  # type: ignore
    from spacy.matcher import PhraseMatcher, Matcher
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    PhraseMatcher = None
    Matcher = None
    Language = None
    Doc = None
    Span = None
    SPACY_AVAILABLE = False


# =====================
# Jupyter/Notebook 감지
# =====================

def is_notebook() -> bool:
    """VSCode/Notebook 환경인지 감지"""
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


# =====================
# 설정 & 데이터 구조
# =====================

@dataclass
class Config:
    # 경로
    base_folder: str = ""
    spec_table_path: str = ""
    output_path: str = ""

    # 학습/캐시 경로
    pattern_db_path: str = "learned_patterns.pkl"
    alias_db_path: str = "alias_db.pkl"

    # 처리 설정
    skip_word_section1: bool = True
    remove_strikethrough: bool = True

    # 유사도/문턱
    fuzzy_match_threshold: float = 0.65        # RapidFuzz token_set/partial 기반
    fuzzy_partial_threshold: float = 0.70
    exact_window: int = 200                    # 스펙명 주변 윈도우
    table_header_min_sim: float = 0.60
    table_value_max_scan_cells: int = 8
    backlink_window_chars: int = 200

    # LLM 설정
    enable_llm: bool = True
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"  # 실제 설치된 태그로 맞추세요
    llm_first: bool = True
    llm_accept_threshold: float = 0.80
    llm_require_substring: bool = True
    verify_window_chars: int = 200
    max_value_len: int = 48
    llm_judge_enable: bool = True
    llm_judge_bonus: float = 0.15

    # 합의/보너스
    consensus_group_bonus: float = 0.15
    consensus_group_bonus_max: float = 0.45
    consensus_compat_support: float = 0.12
    table_tight_bonus: float = 0.20

    # 인코딩
    default_encoding: str = "utf-8"
    fallback_encodings: List[str] = field(default_factory=lambda: ["cp949", "euc-kr", "latin-1"])

    # 런타임
    enable_multiprocessing: bool = True
    num_workers: int = 4
    log_path: Optional[str] = None
    xlsx_path: Optional[str] = None
    result_pkl_path: Optional[str] = None

    # 파일 선택 개선 가중치
    file_match_ship_weight: float = 0.4
    file_match_group_weight: float = 0.6
    file_min_accept_score: float = 0.50

    # ===== spaCy IE 옵션 =====
    enable_spacy_ie: bool = True
    spacy_model_name: Optional[str] = None      # None이면 'en_core_web_sm' 시도→실패시 blank('xx')
    spacy_section_min_sim: float = 0.55         # 어떤 섹션을 IE 대상으로 할지
    spacy_topk_sections: int = 3                # IE 수행 섹션 수
    spacy_window_tokens: int = 40               # 라벨 매치 주변 토큰 창
    spacy_max_chars_per_section: int = 8000     # 섹션 길이 제한(성능)


@dataclass
class DocumentSection:
    number: str
    title: str
    content: str


@dataclass
class SpecExtraction:
    작업일자: str = ""
    작업자: str = "시스템"
    file_name: str = ""
    자재번호: str = ""
    POS: str = ""
    관리Spec명: str = ""
    Spec값: str = ""        # POS에서 추출한 실제 값
    단위: str = ""          # POS에서 추출한 실제 단위
    PMG_CODE: str = ""
    PMG_이름: str = ""
    UMG_CODE: str = ""
    UMG_이름: str = ""
    자재속성그룹: str = ""
    자재속성그룹명: str = ""
    자재속성그룹_SPEC: str = ""
    재료비CG: str = ""
    대상_여부: str = "Y"
    extraction_method: str = "not_found"
    confidence: float = 0.0
    section_path: str = ""     # 섹션 번호
    section_title: str = ""    # 섹션 제목
    evidence: str = ""         # 근거 스니펫
    label_context: str = ""    # 후보 라벨


# =====================
# 로깅
# =====================

def setup_logging(log_path: Optional[str]) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as e:
            root.error(f"로그 파일 핸들러 설정 실패: {e}")


# =====================
# 유틸
# =====================

def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    return s.replace('\uFEFF', '').strip()


def _norm_space_lower(s: str) -> str:
    s = _norm(s).lower()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def try_read_text(path: str, encodings: List[str]) -> str:
    last_err = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc, errors='strict') as f:
                txt = f.read()
            return txt
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"텍스트를 읽을 수 없습니다: {path}; 마지막 에러: {last_err}")


def normalize_number(s: str) -> str:
    s = _norm(s)
    if not s:
        return s
    return s.replace(',', '')


# ========== 단위/변환 ==========
_UNIT_DIM = {
    'mm': 'length', 'cm': 'length', 'm': 'length', 'inch': 'length', 'in': 'length',
    'bar': 'pressure', 'mpa': 'pressure', 'kpa': 'pressure', 'psi': 'pressure',
    '%': 'ratio', 'wt%': 'ratio',
    '°c': 'temperature', 'c': 'temperature', '℃': 'temperature',
    'kg': 'mass', 'g': 'mass',
    'v': 'voltage', 'kv': 'voltage',
    'a': 'current',
    'hz': 'freq',
}

def map_unit(u: str) -> str:
    u0 = _norm(u).replace('％', '%').replace('℃', '°C').replace('°c', '°C').strip()
    if not u0:
        return ''
    u1 = u0.lower()
    t = {
        'mm': 'mm', 'ｍｍ': 'mm', '㎜': 'mm',
        'cm': 'cm',
        'm': 'm',
        'inch': 'inch', 'in': 'inch',
        'bar': 'bar',
        'mpa': 'mpa',
        'kpa': 'kpa',
        'psi': 'psi',
        '%': '%', 'wt%': '%',
        '°c': '°C', 'c': '°C',
        'v': 'V', 'kv': 'kV',
        'a': 'A',
        'hz': 'Hz',
        'kg': 'kg', 'g': 'g'
    }
    return t.get(u1, u0)

# 단위 변환 함수 테이블
def _conv_bar_to_mpa(x): return x / 10.0
def _conv_mpa_to_bar(x): return x * 10.0
def _conv_inch_to_mm(x): return x * 25.4
def _conv_mm_to_inch(x): return x / 25.4
def _conv_kpa_to_mpa(x): return x / 1000.0
def _conv_mpa_to_kpa(x): return x * 1000.0
def _conv_kv_to_v(x): return x * 1000.0
def _conv_v_to_kv(x): return x / 1000.0
def _conv_c_to_k(x): return x + 273.15
def _conv_k_to_c(x): return x - 273.15

_UNIT_CONV = {
    ('bar', 'mpa'): _conv_bar_to_mpa,
    ('mpa', 'bar'): _conv_mpa_to_bar,
    ('inch', 'mm'): _conv_inch_to_mm,
    ('mm', 'inch'): _conv_mm_to_inch,
    ('kpa', 'mpa'): _conv_kpa_to_mpa,
    ('mpa', 'kpa'): _conv_mpa_to_kpa,
    ('kV', 'V'): _conv_kv_to_v,
    ('V', 'kV'): _conv_v_to_kv,
    # 온도 변환은 특수 처리 필요 — 일반 비교에서 제외
}

def _float_or_none(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def parse_value_form(s: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
    """
    '12', '12.5', '12~15', '12±0.5' 등을 해석하여 (type, (a,b))를 반환
    type: 'num' | 'range' | 'pm' | 'text'
    """
    s0 = _norm(s)
    if not s0:
        return 'text', (None, None)
    s1 = normalize_number(s0)
    # range
    m = re.match(r'^\s*(\d[\d\.]*)\s*[~–-]\s*(\d[\d\.]*)\s*$', s1)
    if m:
        a, b = _float_or_none(m.group(1)), _float_or_none(m.group(2))
        return 'range', (a, b)
    # plus/minus
    m = re.match(r'^\s*(\d[\d\.]*)\s*±\s*(\d[\d\.]*)\s*$', s1)
    if m:
        v, dv = _float_or_none(normalize_number(m.group(1))), _float_or_none(normalize_number(m.group(2)))
        if v is not None and dv is not None:
            return 'pm', (v - dv, v + dv)
    # pure number
    m = re.match(r'^\s*(\d[\d\.]*)\s*$', s1)
    if m:
        z = _float_or_none(m.group(1))
        return 'num', (z, z)
    return 'text', (None, None)

def same_dimension(u1: str, u2: str) -> bool:
    a = map_unit(u1); b = map_unit(u2)
    if not a or not b:
        return False
    d1 = _UNIT_DIM.get(a.lower())
    d2 = _UNIT_DIM.get(b.lower())
    return d1 and d2 and d1 == d2

def unit_convert(value: float, u_from: str, u_to: str) -> Optional[float]:
    uf = map_unit(u_from); ut = map_unit(u_to)
    fn = _UNIT_CONV.get((uf, ut))
    if fn is None:
        return None
    try:
        return fn(value)
    except Exception:
        return None

def value_unit_compatible(val_a: str, unit_a: str, val_b: str, unit_b: str, tol: float = 0.05) -> bool:
    """
    값/단위 호환성:
    - 동일 단위: 범위/± 내부, 또는 상대오차 <= tol
    - 변환 가능 단위: 변환 후 비교
    """
    type_a, (a_lo, a_hi) = parse_value_form(val_a)
    type_b, (b_lo, b_hi) = parse_value_form(val_b)
    ua, ub = map_unit(unit_a), map_unit(unit_b)
    if a_lo is None and b_lo is None:
        return False

    def in_interval(z: float, lo: Optional[float], hi: Optional[float]) -> bool:
        if z is None or lo is None or hi is None:
            return False
        return lo <= z <= hi

    if ua == ub:
        if a_lo is not None and a_hi is not None and b_lo is not None and b_hi is not None:
            return not (a_hi < b_lo or b_hi < a_lo)
        if a_lo == a_hi and b_lo is not None and b_hi is not None:
            return in_interval(a_lo, b_lo, b_hi)
        if b_lo == b_hi and a_lo is not None and a_hi is not None:
            return in_interval(b_lo, a_lo, a_hi)
        if a_lo == a_hi and b_lo == b_hi and a_lo is not None and b_lo is not None:
            denom = max(1.0, abs(b_lo))
            return abs(a_lo - b_lo) <= tol * denom
        return False

    if same_dimension(ua, ub):
        def center(lo: Optional[float], hi: Optional[float]) -> Optional[float]:
            if lo is None or hi is None:
                return None
            return 0.5 * (lo + hi)
        a_val = a_lo if a_lo == a_hi else center(a_lo, a_hi)
        b_val = b_lo if b_lo == b_hi else center(b_lo, b_hi)
        if a_val is None or b_val is None:
            return False
        conv = unit_convert(a_val, ua, ub)
        if conv is None:
            return False
        denom = max(1.0, abs(b_val))
        return abs(conv - b_val) <= tol * denom

    return False


# =====================
# 라벨 정규화 / 유사도
# =====================

_ALIAS_BASE = {
    # 약어/동의어(한/영 혼용) — 필요시 확장
    'od': 'outside diameter',
    'id': 'inside diameter',
    'rpm': 'r/min',
    'max': 'maximum',
    'min': 'minimum',
    'nom': 'nominal',
    'wp': 'working pressure',
    'dp': 'design pressure',
    'temp': 'temperature',
    'press': 'pressure',
    'dia': 'diameter',
}

def normalize_label_tokens(s: str) -> str:
    s0 = _norm_space_lower(s)
    s0 = unicodedata.normalize("NFKC", s0)
    s0 = re.sub(r'[\(\)\[\]\{\}<>\|:;,/\\\-\+_~`"“”’\'·•]', ' ', s0)
    s0 = re.sub(r'\s+', ' ', s0).strip()
    toks = s0.split()
    toks2 = []
    for t in toks:
        if t in _ALIAS_BASE:
            toks2.extend(_ALIAS_BASE[t].split())
        else:
            toks2.append(t)
    s1 = ' '.join(toks2)
    s1 = re.sub(r'\b(spec|value|range|note|remarks?|비고|설명|요구사항|요건)\b', ' ', s1)
    s1 = re.sub(r'\s+', ' ', s1).strip()
    return s1

def label_similarity(a: str, b: str) -> float:
    """RapidFuzz 기반 라벨 유사도 [0,1]"""
    a1 = normalize_label_tokens(a)
    b1 = normalize_label_tokens(b)
    if not a1 or not b1:
        return 0.0
    if not _HAVE_RAPIDFUZZ:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a1, b1).ratio()
    s1 = fuzz.token_set_ratio(a1, b1) / 100.0
    s2 = fuzz.partial_ratio(a1, b1) / 100.0
    return max(s1, s2)


# =====================
# TXT 로더
# =====================

class TxtTableLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_spec_table(self, file_path: str) -> pd.DataFrame:
        encodings = [self.config.default_encoding] + self.config.fallback_encodings
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, sep='\t', dtype=str, engine='python', encoding=enc)
                df.columns = [c.replace('\uFEFF', '').strip() for c in df.columns]
                self.logger.info(f"파일 로드 성공: {len(df)}행, 인코딩: {enc}")
                if 'POS' not in df.columns:
                    self.logger.warning("경고: 'POS' 컬럼이 없습니다. 전체를 하나의 그룹으로 처리합니다.")
                return df
            except Exception as e:
                last_err = e
                continue
        raise ValueError(f"파일을 읽을 수 없습니다: {file_path}; 마지막 에러: {last_err}")


# =====================
# 파일 찾기 (개선판)
# =====================

class FileFinder:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _score_name(self, fname: str, ship_no: str, group_name: str) -> float:
        base = os.path.basename(fname)
        base_l = normalize_label_tokens(base)
        ship_l = normalize_label_tokens(ship_no)
        group_l = normalize_label_tokens(group_name)
        s_ship = label_similarity(base_l, ship_l) if ship_l else 0.0
        s_group = label_similarity(base_l, group_l) if group_l else 0.0
        score = self.cfg.file_match_ship_weight * s_ship + self.cfg.file_match_group_weight * s_group
        return score

    def find_html_file(self, pos_number: str, ship_no: str, group_name: str, base_folder: str, hint_file: str = "") -> Optional[str]:
        base_path = Path(base_folder)
        clean_pos = str(pos_number).replace('YS-', '').replace('POS-', '')
        digits = re.sub(r'[^0-9]', '', clean_pos)

        candidates = [f"YS-POS-{clean_pos}", f"POS-{clean_pos}", clean_pos, str(pos_number)]
        paths = []
        for pat in candidates:
            folder = base_path / pat
            if folder.exists():
                if hint_file:
                    hf = folder / hint_file.replace('.DOC', '.html').replace('.doc', '.html')
                    if hf.exists():
                        return str(hf)
                paths.extend([str(p) for p in folder.glob("*.html")])

        if not paths:
            try:
                for p in base_path.rglob("*.html"):
                    if clean_pos in p.name or clean_pos in p.parent.name or (digits and digits in p.name):
                        paths.append(str(p))
            except Exception:
                pass

        if not paths:
            return None

        best, best_score = None, -1.0
        for p in paths:
            sc = self._score_name(p, ship_no, group_name)
            if sc > best_score:
                best_score, best = sc, p

        if best is None or best_score < self.cfg.file_min_accept_score:
            self.logger.warning(f"파일 매칭 스코어가 낮음({best_score:.2f}) — 우선 {best} 선택")
            return best
        return best


# =====================
# HTML 파서
# =====================

class HierarchicalHTMLParser:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _strip_tags_preserve_tables(self, html: str) -> str:
        if self.cfg.remove_strikethrough:
            html = re.sub(r'<(s|strike)>.*?</\1>', ' ', html, flags=re.IGNORECASE | re.DOTALL)
            html = re.sub(r'<[^>]*style=["\']?[^>]*line-through[^>]*>', ' ', html, flags=re.IGNORECASE)
        text = re.sub(r'<table.*?>.*?</table>', ' [TABLE] ', html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = (text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>'))
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _remove_wordsection1(self, html: str) -> str:
        if not self.cfg.skip_word_section1:
            return html
        html = re.sub(r'<!--\s*\[if gte mso.*?endif\]-->', ' ', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'WordSection1.*?(?=WordSection\d|$)', ' ', html, flags=re.IGNORECASE | re.DOTALL)
        return html

    def parse_html_file(self, html_path: str) -> Tuple[List[DocumentSection], Dict[str, Any]]:
        try:
            raw = try_read_text(html_path, [self.cfg.default_encoding] + self.cfg.fallback_encodings)
        except Exception as e:
            self.logger.error(f"HTML 읽기 실패: {html_path} / {e}")
            return [], {"raw_html": "", "soup": None, "html_path": html_path}

        raw2 = self._remove_wordsection1(raw)
        text = self._strip_tags_preserve_tables(raw2)
        lines = re.split(r'(?<=\.)\s+|\n+', text)
        sections: List[DocumentSection] = []
        curr_num, curr_title, curr_buf = None, None, []
        header_re = re.compile(r'^(?P<num>\d+(?:\.\d+)*)(?:\.|\))?\s+(?P<title>.+)$')
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            m = header_re.match(s)
            if m:
                if curr_num is not None:
                    sections.append(DocumentSection(curr_num, curr_title or "", ' '.join(curr_buf).strip()))
                curr_num = m.group('num')
                curr_title = m.group('title')
                curr_buf = []
            else:
                if curr_num is None:
                    curr_num, curr_title = '0', 'PREFACE'
                curr_buf.append(s)
        if curr_num is not None:
            sections.append(DocumentSection(curr_num, curr_title or "", ' '.join(curr_buf).strip()))

        soup = BeautifulSoup(raw2, 'lxml' if _HAVE_LXML else 'html.parser')
        return sections, {"raw_html": raw2, "soup": soup, "html_path": html_path, "text_len": len(text)}


# =====================
# 테이블 매처
# =====================

def extract_value_unit_generic(text: str) -> Tuple[str, str]:
    """
    일반 텍스트에서 '숫자(~범위/±포함)+단위'를 탐지하여 (value, unit)을 반환. 실패 시 ('','')
    """
    s0 = _norm(text)
    # 숫자 부분
    m = re.search(r'(\d[\d.,]*\s*(?:[~–-]\s*\d[\d.,]*)?(?:\s*±\s*\d[\d.,]*)?)', s0)
    val = (m.group(1) if m else '').strip()
    unit = ''
    if m:
        end = m.end(1)
        tail = s0[end:end+20]
        m2 = re.match(r'\s*([A-Za-z°%μ/\\\-]+)', tail)
        if m2:
            unit = m2.group(1)
    return val, unit

class TableMatcher:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _parse_tables_pd(self, html: str) -> List[pd.DataFrame]:
        try:
            dfs = pd.read_html(html, flavor='lxml' if _HAVE_LXML else 'bs4')
            return dfs
        except Exception as e:
            self.logger.warning(f"pandas.read_html 실패: {e}")
            return []

    def _extract_value_and_unit_from_cell(self, s: str) -> Tuple[str, str]:
        return extract_value_unit_generic(s)

    def _score_table_label(self, label: str, spec_name: str) -> float:
        return label_similarity(label, spec_name)

    def _iter_table_candidates(self, df: pd.DataFrame, spec_name: str) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        header_labels = []
        if isinstance(df.columns, pd.MultiIndex):
            header_labels.extend([' '.join([_norm(x) for x in tup]) for tup in df.columns])
        else:
            header_labels.extend([_norm(x) for x in df.columns])
        first_col_labels = [_norm(x) for x in df.iloc[:,0].values] if df.shape[1] >= 2 else []
        labels = [(lab, ('col', j)) for j, lab in enumerate(header_labels)]
        labels += [(lab, ('row', i)) for i, lab in enumerate(first_col_labels)]
        sim_labels = []
        for lab, pos in labels:
            s = self._score_table_label(lab, spec_name)
            if s >= self.cfg.table_header_min_sim:
                sim_labels.append((s, lab, pos))
        for sim, lab, pos in sorted(sim_labels, key=lambda x: x[0], reverse=True):
            if pos[0] == 'col':
                j = pos[1]
                for i in range(min(df.shape[0], self.cfg.table_value_max_scan_cells)):
                    cell = str(df.iloc[i, j])
                    val, unit = self._extract_value_and_unit_from_cell(cell)
                    if val:
                        cands.append({
                            "found": True, "value": val, "unit": unit,
                            "label_context": lab, "table_pos": f"col:{j},row:{i}",
                            "confidence": max(0.75, sim),
                            "method": "table",
                            "evidence": cell[:160],
                            "table_tight": True
                        })
            else:
                i = pos[1]
                row_vals = df.iloc[i, :min(df.shape[1], self.cfg.table_value_max_scan_cells)]
                for j in range(len(row_vals)):
                    cell = str(row_vals.iloc[j])
                    val, unit = self._extract_value_and_unit_from_cell(cell)
                    if val:
                        cands.append({
                            "found": True, "value": val, "unit": unit,
                            "label_context": lab, "table_pos": f"row:{i},col:{j}",
                            "confidence": max(0.75, sim),
                            "method": "table",
                            "evidence": cell[:160],
                            "table_tight": True
                        })

        return cands

    def table_match(self, html: str, spec_name: str) -> List[Dict[str, Any]]:
        dfs = self._parse_tables_pd(html)
        out: List[Dict[str, Any]] = []
        for df in dfs:
            try:
                out.extend(self._iter_table_candidates(df, spec_name))
            except Exception:
                continue
        return out


# =====================
# spaCy 정보추출기
# =====================

class SpacyExtractor:
    """
    - 섹션 중 spec_name과 유사도가 높은 상위 K개에서만 spaCy 처리(성능/효율)
    - PhraseMatcher로 spec_name(및 단순 변형) 스팬을 잡고, 주변 토큰 창에서 숫자+단위 추출
    - spaCy 미설치/모델 부재 시 자동 폴백(blank('xx') 토크나이저)
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._nlp = None  # type: Optional[Language]
        self._ensure_pipeline()

    def _ensure_pipeline(self):
        if not (self.cfg.enable_spacy_ie and SPACY_AVAILABLE):
            self._nlp = None
            return
        nlp = None
        try:
            if self.cfg.spacy_model_name:
                nlp = spacy.load(self.cfg.spacy_model_name, disable=[])
            else:
                # 기본 영어 소형 모델 시도
                try:
                    nlp = spacy.load("en_core_web_sm", disable=[])
                except Exception:
                    nlp = spacy.blank("xx")  # 다국어 토크나이저 폴백
        except Exception as e:
            self.logger.warning(f"spaCy 모델 로드 실패: {e}")
            nlp = spacy.blank("xx") if SPACY_AVAILABLE else None
        self._nlp = nlp

    def _build_phrase_matcher(self, nlp: Language, spec_name: str) -> PhraseMatcher:
        pm = PhraseMatcher(nlp.vocab, attr="LOWER")
        # 기본 표현 + 약어 확장 표현
        variants = set()
        base = normalize_label_tokens(spec_name)
        variants.add(base)
        # 간단한 변형(하이픈/스페이스)
        dash = base.replace(' ', '-')
        variants.add(dash)
        # max/min/nominal 토큰 포함 가능성
        if 'maximum' not in base:
            variants.add('maximum ' + base)
        if 'minimum' not in base:
            variants.add('minimum ' + base)
        if 'nominal' not in base:
            variants.add('nominal ' + base)

        phrases = [nlp.make_doc(v) for v in variants if v]
        if phrases:
            pm.add("SPECNAME", phrases)
        return pm

    def _sentences(self, doc: Doc):
        # sentencizer가 없을 수 있으므로 간단 폴백
        if doc.has_annotation("SENT_START"):
            for s in doc.sents:
                yield s
        else:
            # 마침표 기준 단순 분할
            text = doc.text
            for seg in re.split(r'(?<=[\.\n])\s+', text):
                if seg.strip():
                    yield self._nlp.make_doc(seg)

    def spacy_match(self, sections: List[DocumentSection], spec_name: str) -> List[Dict[str, Any]]:
        if not self.cfg.enable_spacy_ie or self._nlp is None:
            return []
        # 섹션 선별
        sims = []
        for s in sections:
            probe = (s.title + " " + s.content[:400])
            sim = label_similarity(probe, spec_name)
            sims.append((sim, s))
        sims.sort(key=lambda x: x[0], reverse=True)
        chosen = [s for sim, s in sims if sim >= self.cfg.spacy_section_min_sim][: self.cfg.spacy_topk_sections]
        if not chosen:
            return []

        nlp = self._nlp
        pm = self._build_phrase_matcher(nlp, spec_name)
        out: List[Dict[str, Any]] = []
        for s in chosen:
            text = s.content[: self.cfg.spacy_max_chars_per_section]
            doc = nlp(text)
            matches = pm(doc)
            # 매치가 없으면 유사도 상위 문장들에서 라벨 근사로 시도
            spans = [doc[start:end] for mid, start, end in matches]
            if not spans:
                # 유사도 높은 문장 몇 개만 스캔
                sent_scores = []
                for sent in self._sentences(doc):
                    sent_scores.append((label_similarity(sent.text, spec_name), sent))
                sent_scores.sort(key=lambda x: x[0], reverse=True)
                spans = [sent_scores[i][1] for i in range(min(3, len(sent_scores))) if sent_scores[i][0] >= 0.60]

            for sp in spans:
                # 주변 토큰 창
                start = max(0, sp.start - self.cfg.spacy_window_tokens)
                end = min(len(doc), sp.end + self.cfg.spacy_window_tokens)
                window = doc[start:end].text
                # 숫자+단위 추출
                val, unit = extract_value_unit_generic(window)
                if not val:
                    continue
                sim = label_similarity(sp.text, spec_name)
                out.append({
                    "found": True,
                    "value": val,
                    "unit": unit,
                    "section_path": s.number,
                    "section_title": s.title,
                    "confidence": max(0.72, sim),
                    "method": "spacy",
                    "evidence": window[:160],
                    "label_context": sp.text
                })
        return out


# =====================
# 엔티티/패턴/LLM 매처
# =====================

class HierarchicalMatcher:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.table = TableMatcher(config)
        self.spacyie = SpacyExtractor(config)

    # ---------- 공통

    def _traverse_sections(self, sections: List[DocumentSection]):
        for s in sections:
            yield s

    def _gen_patterns(self, spec_name: str) -> List[Tuple[str, str]]:
        name = re.escape(spec_name)
        return [
            (rf"{name}\s*[:：]\s*([0-9][0-9.,]*(?:\s*[~–-]\s*[0-9][0-9.,]*)?(?:\s*±\s*[0-9][0-9.,]*)?)\s*([a-zA-Z%/°℃℉\-^0-9]+)?", 'colon'),
            (rf"{name}\s*=\s*([0-9][0-9.,]*(?:\s*[~–-]\s*[0-9][0-9.,]*)?(?:\s*±\s*[0-9][0-9.,]*)?)\s*([a-zA-Z%/°℃℉\-^0-9]+)?", 'equal'),
            (rf"{name}[^\n]{{0,40}}?([0-9][0-9.,]*(?:\s*[~–-]\s*[0-9][0-9.,]*)?(?:\s*±\s*[0-9][0-9.,]*)?)\s*([a-zA-Z%/°℃℉\-^0-9]+)?", 'near'),
        ]

    def _apply_patterns_to_text(self, text: str, spec_name: str) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        patterns = self._gen_patterns(spec_name)
        for pat, ptype in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                value = (m.group(1) or '').strip()
                unit = (m.group(2) or '').strip()
                span = m.span(1)
                return value, unit, span
        return None

    # ---------- 매칭 방법

    def exact_match(self, sections: List[DocumentSection], spec_name: str) -> Optional[Dict]:
        nm = _norm(spec_name)
        for sec in self._traverse_sections(sections):
            loc = sec.content.lower().find(nm.lower()) if nm else -1
            if loc != -1:
                ap = self._apply_patterns_to_text(sec.content, nm)
                if ap:
                    val, unit, span = ap
                    idx = span[0]
                    snippet = sec.content[max(0, idx-60): min(len(sec.content), idx+60)]
                    return {"found": True, "value": val, "unit": unit, "section_path": sec.number, "section_title": sec.title,
                            "confidence": 0.80, "method": "exact", "evidence": snippet, "label_context": nm}
                snippet = sec.content[max(0, loc-60): min(len(sec.content), loc+len(nm)+60)]
                return {"found": True, "value": "", "unit": "", "section_path": sec.number, "section_title": sec.title,
                        "confidence": 0.60, "method": "exact", "evidence": snippet, "label_context": nm}
        return None

    def fuzzy_match(self, sections: List[DocumentSection], spec_name: str) -> Optional[Dict]:
        best, best_score = None, -1.0
        nm = _norm(spec_name)
        for sec in self._traverse_sections(sections):
            idx = sec.content.lower().find(nm.lower())
            if idx != -1:
                lo = max(0, idx - self.cfg.exact_window)
                hi = min(len(sec.content), idx + len(nm) + self.cfg.exact_window)
                window = sec.content[lo:hi]
            else:
                window = sec.content[:800]
            score = label_similarity(window, nm)
            if score > best_score:
                best_score = score
                best = (sec, window)
        if best and best_score >= self.cfg.fuzzy_match_threshold:
            sec, window = best
            ap = self._apply_patterns_to_text(window, nm)
            if ap:
                val, unit, span = ap
                snippet = window[max(0, span[0]-60): min(len(window), span[0]+60)]
                return {"found": True, "value": val, "unit": unit, "section_path": sec.number, "section_title": sec.title,
                        "confidence": max(0.70, best_score), "method": "fuzzy", "evidence": snippet, "label_context": nm}
            return {"found": True, "value": "", "unit": "", "section_path": sec.number, "section_title": sec.title,
                    "confidence": best_score, "method": "fuzzy", "evidence": window[:160], "label_context": nm}
        return None

    def pattern_match(self, sections: List[DocumentSection], spec_name: str) -> Optional[Dict]:
        for sec in self._traverse_sections(sections):
            ap = self._apply_patterns_to_text(sec.content, spec_name)
            if ap:
                val, unit, span = ap
                idx = span[0]
                snippet = sec.content[max(0, idx-60): min(len(sec.content), idx+60)]
                return {"found": True, "value": val, "unit": unit,
                        "section_path": sec.number, "section_title": sec.title,
                        "confidence": 0.90, "method": "pattern", "evidence": snippet, "label_context": spec_name}
        return None

    def table_match(self, raw_html: str, spec_name: str) -> List[Dict[str, Any]]:
        return self.table.table_match(raw_html, spec_name)

    def spacy_ie_match(self, sections: List[DocumentSection], spec_name: str) -> List[Dict[str, Any]]:
        return self.spacyie.spacy_match(sections, spec_name)

    def value_backlink_match(self, raw_text: str, spec_name: str, input_value: str) -> Optional[Dict]:
        z = _norm(input_value)
        if not z:
            return None
        z_norm = normalize_number(z)
        variants = set([z, z_norm])
        m_pm = re.match(r'^\s*(\d[\d.,]*)\s*±\s*(\d[\d.,]*)', z)
        if m_pm:
            v = _float_or_none(normalize_number(m_pm.group(1)))
            dv = _float_or_none(normalize_number(m_pm.group(2)))
            center = (v if (v is not None and dv is not None) else None)
            if center is not None:
                variants.add(str(center))
        patt = r'|'.join([re.escape(x) for x in variants if x])
        hits = list(re.finditer(patt, raw_text))
        if not hits:
            return None
        best, best_score = None, -1.0
        for h in hits:
            lo = max(0, h.start() - self.cfg.backlink_window_chars)
            hi = min(len(raw_text), h.end() + self.cfg.backlink_window_chars)
            window = raw_text[lo:hi]
            mlabel = re.search(r'([A-Za-z가-힣0-9\s\-/]+?)[:：]\s*' + re.escape(h.group(0)), window)
            label = mlabel.group(1) if mlabel else window[:80]
            sim = label_similarity(label, spec_name)
            if sim > best_score:
                best_score = sim
                best = (window, label, h)
        if best and best_score >= self.cfg.fuzzy_partial_threshold:
            window, label, h = best
            ap = self._apply_patterns_to_text(window, spec_name)
            val, unit = '', ''
            if ap:
                val, unit, _ = ap
            else:
                munit = re.search(r'([A-Za-z°%μ/\\\-]+)', window)
                unit = munit.group(1) if munit else ''
                val = _norm(h.group(0))
            return {"found": True, "value": val, "unit": unit, "section_path": "", "section_title": "",
                    "confidence": max(0.70, best_score), "method": "backlink", "evidence": window[:160],
                    "label_context": label}
        return None

    # ---------- LLM 관련

    def _llm_chat(self, prompt: str) -> Optional[str]:
        if not self.cfg.enable_llm or not OLLAMA_AVAILABLE or not ollama:
            return None
        try:
            resp = ollama.chat(model=self.cfg.llm_model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
            text = resp.get('message', {}).get('content', '') if isinstance(resp, dict) else ''
            return text
        except Exception as e:
            logging.getLogger(__name__).warning(f"LLM 호출 실패: {e}")
            msg = str(e)
            if "model" in msg and "not found" in msg:
                try:
                    self.cfg.enable_llm = False
                    logging.getLogger(__name__).warning("LLM 비활성화(모델 미설치 감지). 규칙/테이블/백링크/spaCy로 계속.")
                except Exception:
                    pass
            return None

    def llm_match_extractor(self, sections: List[DocumentSection], spec_name: str) -> Optional[Dict]:
        if not self.cfg.enable_llm:
            return None
        context = []
        total = 0
        for s in self._traverse_sections(sections):
            frag = f"[{s.number}] {s.title} :: {s.content[:1200]}"
            context.append(frag)
            total += len(frag)
            if total > 3000:
                break
        prompt = (
            "You are a strict information extractor for engineering specs.\n"
            "Given a SPEC NAME and document excerpts, return numeric-like value and unit if present.\n"
            "Reply strictly as JSON with keys: value, unit, section, section_title, evidence.\n"
            "Return empty strings for absent fields. Do not guess.\n"
            f"SPEC: {spec_name}\n" + '\n'.join(context)
        )
        text = self._llm_chat(prompt)
        if not text:
            return None
        m_val = re.search(r'"value"\s*:\s*"(.*?)"', text)
        m_unit = re.search(r'"unit"\s*:\s*"(.*?)"', text)
        m_sect = re.search(r'"section"\s*:\s*"(.*?)"', text)
        m_title= re.search(r'"section_title"\s*:\s*"(.*?)"', text)
        m_evd  = re.search(r'"evidence"\s*:\s*"(.*?)"', text)
        if m_val:
            return {
                "found": True,
                "value": (m_val.group(1) or "").strip(),
                "unit": (m_unit.group(1) or "").strip(),
                "section_path": (m_sect.group(1) or "").strip(),
                "section_title": (m_title.group(1) or "").strip(),
                "confidence": 0.75,
                "method": "llm",
                "evidence": (m_evd.group(1) or "").strip(),
                "label_context": spec_name
            }
        return None

    def llm_entailment_judge(self, spec_name: str, label_context: str, evidence: str) -> Optional[bool]:
        if not (self.cfg.enable_llm and self.cfg.llm_judge_enable):
            return None
        prompt = (
            "You are a careful judge for terminology equivalence in engineering.\n"
            "Question: Is the LABEL semantically equivalent to the SPEC NAME in this context? Answer strictly in JSON.\n"
            "JSON keys: decision (yes/no), reason (short).\n"
            f"SPEC NAME: {spec_name}\n"
            f"LABEL: {label_context}\n"
            f"CONTEXT: {evidence[:500]}\n"
            "Return only JSON."
        )
        text = self._llm_chat(prompt)
        if not text:
            return None
        m = re.search(r'"decision"\s*:\s*"(.*?)"', text, re.IGNORECASE)
        if not m:
            return None
        dec = m.group(1).strip().lower()
        if dec in ('yes', 'y', 'true'):
            return True
        if dec in ('no', 'n', 'false'):
            return False
        return None

    def verify_llm_extractor(self, sections: List[DocumentSection], spec_name: str, r: Dict) -> Tuple[bool, str]:
        if not r:
            return False, "empty"
        val = (r.get("value") or "").strip()
        unit = (r.get("unit") or "").strip()
        if not val or len(val) > self.cfg.max_value_len:
            return False, "empty_or_long"
        sec = None
        spath = (r.get("section_path") or "").strip()
        stitle = (r.get("section_title") or "").strip().lower()
        for s in sections:
            if spath and s.number == spath:
                sec = s; break
        if sec is None and stitle:
            for s in sections:
                if stitle in s.title.lower():
                    sec = s; break
        if sec is None:
            sec = sections[0] if sections else None
            if sec is None:
                return False, "no_sections"
        text = sec.content
        hit = text.lower().find(spec_name.lower())
        if hit != -1:
            w = self.cfg.verify_window_chars
            lo = max(0, hit - w); hi = min(len(text), hit + len(spec_name) + w)
            window = text[lo:hi]
        else:
            window = text
        val_ok = (val in window) if self.cfg.llm_require_substring else True
        unit_ok = (unit in window) if (self.cfg.llm_require_substring and unit) else True
        if not val_ok:
            return False, "value_not_in_text_window"
        if not unit_ok:
            return False, "unit_not_in_text_window"
        return True, "ok"

    # ---------- 합의 선택

    def _canon_value(self, s: str) -> str:
        t, (lo, hi) = parse_value_form(s)
        if t == 'num' and lo is not None:
            return f"num:{lo}"
        if t == 'range' and lo is not None and hi is not None:
            return f"range:{lo}:{hi}"
        if t == 'pm' and lo is not None and hi is not None:
            return f"pm:{lo}:{hi}"
        return f"text:{_norm_space_lower(s)}"

    def choose_by_consensus(
        self,
        sections: List[DocumentSection],
        spec_name: str,
        raw_html: str,
        raw_text: str,
        input_value: str = "",
        use_llm_extractor: bool = False
    ) -> Optional[Dict]:

        candidates: List[Dict[str, Any]] = []

        def add_candidate(result: Optional[Dict], base_weight: float):
            if not result or not result.get("found"):
                return
            val = _norm(result.get("value", ""))
            if not val:
                return
            conf = float(result.get("confidence", 0.0))
            score = base_weight * conf
            r = dict(result)
            r["_score"] = score
            r["_base"] = base_weight
            candidates.append(r)

        # 기본 가중치
        W = {
            "llm_verified": 0.90,
            "llm_unverified": 0.10,
            "pattern": 0.90,
            "exact": 0.85,
            "fuzzy": 0.75,
            "table": 0.92,
            "backlink": 0.80,
            "spacy": 0.88,   # NEW: spaCy IE
        }

        # --- 후보 생성 ---
        if self.cfg.llm_first and use_llm_extractor:
            r_llm = self.llm_match_extractor(sections, spec_name)
            if r_llm and r_llm.get("found"):
                ok, why = self.verify_llm_extractor(sections, spec_name, r_llm)
                r_llm["verified"] = ok
                r_llm["verify_reason"] = why
                add_candidate(r_llm, W["llm_verified"] if ok else W["llm_unverified"])

        # 규칙/패턴/퍼지
        add_candidate(self.pattern_match(sections, spec_name), W["pattern"])
        add_candidate(self.exact_match(sections, spec_name),   W["exact"])
        add_candidate(self.fuzzy_match(sections, spec_name),   W["fuzzy"])

        # 테이블
        for t in self.table_match(raw_html, spec_name):
            add_candidate(t, W["table"])

        # spaCy IE
        for t in self.spacy_ie_match(sections, spec_name):
            add_candidate(t, W["spacy"])

        # 값-백링크
        if input_value:
            add_candidate(self.value_backlink_match(raw_text, spec_name, input_value), W["backlink"])

        # LLM-last
        if (not self.cfg.llm_first) and use_llm_extractor:
            r_llm = self.llm_match_extractor(sections, spec_name)
            if r_llm and r_llm.get("found"):
                ok, why = self.verify_llm_extractor(sections, spec_name, r_llm)
                r_llm["verified"] = ok
                r_llm["verify_reason"] = why
                add_candidate(r_llm, W["llm_verified"] if ok else W["llm_unverified"])

        if not candidates:
            return None

        def canon_pair(val: str, unit: str) -> Tuple[str, str]:
            return (self._canon_value(val), map_unit(unit))

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for c in candidates:
            key = canon_pair(_norm(c.get("value","")), _norm(c.get("unit","")))
            groups.setdefault(key, []).append(c)

        best_key, best_total, best_rep = None, -1.0, None
        bonus_per = self.cfg.consensus_group_bonus
        bonus_max = self.cfg.consensus_group_bonus_max
        compat_bonus = self.cfg.consensus_compat_support
        table_tight_bonus = self.cfg.table_tight_bonus
        judge_bonus = self.cfg.llm_judge_bonus

        for key, items in groups.items():
            base_sum = sum(float(it.get("_score", 0.0)) for it in items)
            methods = set(it.get("method") for it in items)
            group_bonus = min(bonus_per * max(0, len(methods) - 1), bonus_max)
            rep_val = items[0].get("value","")
            rep_unit= items[0].get("unit","")
            compat_count = 0
            for other_key, others in groups.items():
                if other_key == key:
                    continue
                if value_unit_compatible(rep_val, rep_unit, others[0].get("value",""), others[0].get("unit","")):
                    compat_count += 1
            support_bonus = compat_bonus * compat_count
            tight = any(it.get("method") == "table" and it.get("table_tight") for it in items)
            table_bonus = table_tight_bonus if tight else 0.0
            rep = max(items, key=lambda x: float(x.get("_score", 0.0)))
            judge = None
            if self.cfg.llm_judge_enable:
                judge = self.llm_entailment_judge(spec_name, rep.get("label_context",""), rep.get("evidence",""))
            judge_adj = 0.0
            if judge is True:
                judge_adj += judge_bonus
            elif judge is False:
                judge_adj -= judge_bonus
            total = base_sum + group_bonus + support_bonus + table_bonus + judge_adj
            if total > best_total:
                best_total = total
                best_rep = rep
                best_key = key

        if not best_rep:
            return None

        accept = True
        if best_rep.get("method") == "llm":
            if not best_rep.get("verified") or float(best_rep.get("confidence", 0.0)) < self.cfg.llm_accept_threshold:
                accept = False

        if not accept:
            rest = []
            for key, items in groups.items():
                if key == best_key:
                    continue
                base_sum = sum(float(it.get("_score", 0.0)) for it in items)
                methods = set(it.get("method") for it in items)
                group_bonus = min(bonus_per * max(0, len(methods) - 1), bonus_max)
                rep_val = items[0].get("value","")
                rep_unit= items[0].get("unit","")
                compat_count = 0
                for other_key, others in groups.items():
                    if other_key == key:
                        continue
                    if value_unit_compatible(rep_val, rep_unit, others[0].get("value",""), others[0].get("unit","")):
                        compat_count += 1
                support_bonus = compat_bonus * compat_count
                tight = any(it.get("method") == "table" and it.get("table_tight") for it in items)
                table_bonus = table_tight_bonus if tight else 0.0
                rep2 = max(items, key=lambda x: float(x.get("_score", 0.0)))
                judge = None
                if self.cfg.llm_judge_enable:
                    judge = self.llm_entailment_judge(spec_name, rep2.get("label_context",""), rep2.get("evidence",""))
                judge_adj = 0.0
                if judge is True: judge_adj += judge_bonus
                elif judge is False: judge_adj -= judge_bonus
                total = base_sum + group_bonus + support_bonus + table_bonus + judge_adj
                rest.append((rep2, total))
            if not rest:
                return None
            best_rep, best_total = max(rest, key=lambda x: x[1])

        return {
            "found": True,
            "value": _norm(best_rep.get("value","")),
            "unit": _norm(best_rep.get("unit","")),
            "section_path": _norm(best_rep.get("section_path","")),
            "section_title": _norm(best_rep.get("section_title","")),
            "confidence": float(best_rep.get("confidence", 0.0)),
            "method": _norm(best_rep.get("method","")),
            "evidence": _norm(best_rep.get("evidence","")),
            "label_context": _norm(best_rep.get("label_context","")),
            "_score": float(best_rep.get("_score", 0.0)),
        }


# =====================
# 패턴/별칭 학습(간략)
# =====================

class AliasDB:
    def __init__(self, path: str):
        self.path = path
        self.db: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self):
        if self.path and os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    self.db = pickle.load(f)
            except Exception:
                self.db = {}

    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self.db, f)
        except Exception:
            pass

    def record(self, spec_name: str, label_context: str):
        s = normalize_label_tokens(spec_name)
        l = normalize_label_tokens(label_context)
        if not s or not l:
            return
        d = self.db.setdefault(s, {})
        d[l] = d.get(l, 0) + 1

    def best_aliases(self, spec_name: str, topk: int = 5) -> List[str]:
        s = normalize_label_tokens(spec_name)
        d = self.db.get(s, {})
        return [k for k, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk]]


# =====================
# 결과 저장/검증
# =====================

def unit_matches(input_unit: str, extracted_unit: str) -> Tuple[bool, str]:
    a = map_unit(input_unit); b = map_unit(extracted_unit)
    if not a and not b:
        return True, "둘다공백"
    if a == b:
        return True, "동일단위"
    if same_dimension(a, b):
        return True, "차원동일_환산가능"
    return False, "단위상이"

def value_matches(input_val: str, extracted_val: str, input_unit: str, extracted_unit: str) -> Tuple[str, str]:
    ai = _norm(input_val); bi = _norm(extracted_val)
    if not ai and not bi:
        return ("미검증", "둘다공백")
    if ai == bi and ai != "":
        return ("일치", "동일문자열")
    ok = value_unit_compatible(ai, input_unit, bi, extracted_unit)
    if ok:
        return ("부분일치_호환", "범위/±/환산호환")
    if ai and bi and ai != bi:
        return ("불일치_값상이", "문자열상이")
    return ("미검증", "비교불가")

def classify_verdict(row: pd.Series) -> str:
    method = _norm(row.get('extraction_method', ''))
    if method in {'not_found_file', 'not_found_file_retry'}:
        return 'POS없음'
    if method in {'parse_failed'}:
        return '파싱실패'
    if method in {'exception', 'exception_retry'}:
        return '예외'

    in_val = _norm(row.get('입력_Spec값', ''))
    in_unit= _norm(row.get('입력_단위', ''))
    ex_val = _norm(row.get('POS_실제_Spec값', ''))
    ex_unit= _norm(row.get('POS_실제_단위', ''))

    if not in_val and not in_unit:
        if ex_val or ex_unit:
            return '입력값없음_자동추출성공'
        else:
            return '입력값없음_미발견'

    val_res, _ = value_matches(in_val, ex_val, in_unit, ex_unit)
    uok, uwhy = unit_matches(in_unit, ex_unit)

    if method in {'not_found', 'skip'} and not ex_val:
        return '미검증'

    if val_res == '일치' and uok and uwhy == '동일단위':
        return '일치'
    if val_res.startswith('부분일치'):
        if uwhy in {'동일단위', '차원동일_환산가능'}:
            return '부분일치'
    if not uok and uwhy == '단위상이':
        return '불일치_단위상이'
    if val_res.startswith('불일치'):
        return '불일치_값상이'
    return '미검증'

def augment_with_validation(base_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.copy()
    fin = final_df.copy()
    key_cols = ['POS', '자재번호', '관리Spec명']
    for df in (base, fin):
        for kc in key_cols:
            if kc not in df.columns:
                df[kc] = ''
            df[kc] = df[kc].astype(str)
        df['__key__'] = df[key_cols].agg('|'.join, axis=1)

    in_val_col = 'Spec값' if 'Spec값' in base.columns else '입력_Spec값'
    in_unit_col = '단위' if '단위' in base.columns else '입력_단위'
    base = base.rename(columns={in_val_col: '입력_Spec값', in_unit_col: '입력_단위'})

    if 'Spec값' in fin.columns:
        fin = fin.rename(columns={'Spec값': 'POS_실제_Spec값'})
    else:
        fin['POS_실제_Spec값'] = ''
    if '단위' in fin.columns:
        fin = fin.rename(columns={'단위': 'POS_실제_단위'})
    else:
        fin['POS_실제_단위'] = ''

    if 'section_path' in fin.columns:
        fin = fin.rename(columns={'section_path': 'POS_섹션경로'})
    else:
        fin['POS_섹션경로'] = ''
    if 'section_title' in fin.columns:
        fin = fin.rename(columns={'section_title': 'POS_섹션제목'})
    else:
        fin['POS_섹션제목'] = ''
    if 'evidence' not in fin.columns:
        fin['evidence'] = ''

    base_slim = base[['__key__', '입력_Spec값', '입력_단위']]
    joined = pd.merge(fin, base_slim, on='__key__', how='left')

    verdict, vtype, reason, sug_val, sug_unit = [], [], [], [], []
    for _, r in joined.iterrows():
        in_val = _norm(r.get('입력_Spec값',''))
        in_uni = _norm(r.get('입력_단위',''))
        ex_val = _norm(r.get('POS_실제_Spec값',''))
        ex_uni = _norm(r.get('POS_실제_단위',''))
        vres, why = value_matches(in_val, ex_val, in_uni, ex_uni)
        uok, uwhy = unit_matches(in_uni, ex_uni)
        typ = classify_verdict(r)
        vtype.append(typ)
        if typ == '일치':
            verdict.append('일치')
        elif typ.startswith('부분일치'):
            verdict.append('부분일치')
        elif typ in {'POS없음','파싱실패','예외','입력값없음_미발견','미검증'}:
            verdict.append('미검증')
        else:
            verdict.append('불일치')
        reason.append(f"{why}/{uwhy}")
        if (not in_val and not in_uni) or verdict[-1] in {'불일치','부분일치'}:
            sug_val.append(ex_val); sug_unit.append(ex_uni)
        else:
            sug_val.append(''); sug_unit.append('')
    joined['검증_결과'] = verdict
    joined['검증결과_유형'] = vtype
    joined['검증_사유'] = reason
    joined['보정_Spec값'] = sug_val
    joined['보정_단위'] = sug_unit

    pref = [
        '작업일자','작업자','file_name','POS','자재번호','관리Spec명',
        '자재속성그룹','자재속성그룹명','PMG_CODE','PMG_이름','UMG_CODE','UMG_이름',
        '입력_Spec값','입력_단위',
        'POS_실제_Spec값','POS_실제_단위','POS_섹션경로','POS_섹션제목',
        '보정_Spec값','보정_단위',
        '검증_결과','검증결과_유형','검증_사유',
        'extraction_method','confidence','evidence','label_context',
        '자재속성그룹_SPEC','재료비CG','대상_여부',
    ]
    others = [c for c in joined.columns if c not in pref and c != '__key__']
    cols = [c for c in pref if c in joined.columns] + others
    return joined[[c for c in cols if c in joined.columns]]


def save_outputs(df: pd.DataFrame, txt_path: str, xlsx_path: str, pkl_path: str):
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('\t'.join(df.columns) + '\n')
        for _, row in df.iterrows():
            vals = ['' if (pd.isna(v) or v is None) else str(v) for v in row.values]
            f.write('\t'.join(vals) + '\n')
    tsv16 = re.sub(r'\.txt$', '.excel.tsv', txt_path)
    with open(tsv16, 'w', encoding='utf-16le', newline='') as f:
        f.write('\t'.join(df.columns) + '\n')
        for _, row in df.iterrows():
            vals = ['' if (pd.isna(v) or v is None) else str(v) for v in row.values]
            f.write('\t'.join(vals) + '\n')
    try:
        with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='results')
    except Exception as e:
        logging.getLogger(__name__).error(f"엑셀 저장 실패: {e}")
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        logging.getLogger(__name__).error(f"피클 저장 실패: {e}")


# =====================
# 프로세서
# =====================

class POSSpecProcessor:
    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.txt_loader = TxtTableLoader(config)
        self.parser = HierarchicalHTMLParser(config)
        self.matcher = HierarchicalMatcher(config)
        self.finder = FileFinder(config)
        self.alias_db = AliasDB(config.alias_db_path)

    def mk_not_found_row(self, row: pd.Series, pos_number: str, reason: str = 'not_found_file', file_name: str = '') -> SpecExtraction:
        return SpecExtraction(
            작업일자=datetime.now().strftime('%Y-%m-%d'),
            file_name=file_name or _norm(row.get('file_name', '')),
            자재번호=_norm(row.get('자재번호', '')),
            POS=_norm(pos_number),
            관리Spec명=_norm(row.get('관리Spec명', '')),
            자재속성그룹=_norm(row.get('자재속성그룹', '')),
            자재속성그룹명=_norm(row.get('자재속성그룹명', '')),
            PMG_CODE=_norm(row.get('PMG CODE', row.get('PMG_CODE', ''))),
            PMG_이름=_norm(row.get('PMG 이름', row.get('PMG_이름', ''))),
            UMG_CODE=_norm(row.get('UMG CODE', row.get('UMG_CODE', ''))),
            UMG_이름=_norm(row.get('UMG 이름', row.get('UMG_이름', ''))),
            자재속성그룹_SPEC=_norm(row.get('자재속성그룹 SPEC', row.get('자재속성그룹_SPEC', ''))),
            재료비CG=_norm(row.get('재료비CG', '')),
            대상_여부=_norm(row.get('대상 여부', row.get('대상_여부', 'Y'))),
            extraction_method=reason,
            confidence=0.0,
            section_path='',
            section_title='',
        )

    def process_spec_table(self, spec_table_path: str, base_folder: str, output_path: str) -> pd.DataFrame:
        self.logger.info("Spec 테이블 로드 중...")
        spec_df = self.txt_loader.load_spec_table(spec_table_path)
        results: List[Dict[str, Any]] = []
        if 'POS' in spec_df.columns:
            grouped = spec_df.groupby('POS')
        else:
            grouped = [("", spec_df)]
        with tqdm(total=getattr(grouped, 'ngroups', 1), desc="POS 문서 처리") as pbar:
            for pos, group in grouped:
                try:
                    first_row = group.iloc[0] if not group.empty else None
                    ship_no = _norm(first_row.get('호선', first_row.get('선종', ''))) if first_row is not None else ''
                    group_name = _norm(first_row.get('자재속성그룹명','')) if first_row is not None else ''
                    hint_file = _norm(first_row.get('file_name','')) if first_row is not None else ''
                    html_path = self.finder.find_html_file(str(pos), ship_no, group_name, base_folder, hint_file)
                    if not html_path:
                        for _, row in group.iterrows():
                            results.append(asdict(self.mk_not_found_row(row, str(pos), reason='not_found_file')))
                        pbar.update(1)
                        continue
                    sections, meta = self.parser.parse_html_file(html_path)
                    raw_html = meta.get("raw_html","")
                    raw_text = re.sub(r'<[^>]+>', ' ', raw_html)
                    if not sections:
                        for _, row in group.iterrows():
                            results.append(asdict(self.mk_not_found_row(row, str(pos), reason='parse_failed', file_name=os.path.basename(html_path))))
                        pbar.update(1)
                        continue
                    for _, row in group.iterrows():
                        spec_name = _norm(row.get('관리Spec명', ''))
                        input_val = _norm(row.get('Spec값', row.get('입력_Spec값', '')))
                        input_unit = _norm(row.get('단위', row.get('입력_단위', '')))
                        if not spec_name:
                            results.append(asdict(self.mk_not_found_row(row, str(pos), reason='skip', file_name=os.path.basename(html_path))))
                            continue
                        match = self.matcher.choose_by_consensus(
                            sections, spec_name, raw_html, raw_text, input_value=input_val, use_llm_extractor=False
                        )
                        if match and match.get('found'):
                            ex = SpecExtraction(
                                작업일자=datetime.now().strftime('%Y-%m-%d'), 작업자='시스템',
                                file_name=os.path.basename(html_path),
                                자재번호=_norm(row.get('자재번호','')),
                                POS=str(pos), 관리Spec명=spec_name,
                                Spec값=_norm(match.get('value','')), 단위=_norm(match.get('unit','')),
                                PMG_CODE=_norm(row.get('PMG CODE', row.get('PMG_CODE',''))), PMG_이름=_norm(row.get('PMG 이름', row.get('PMG_이름',''))),
                                UMG_CODE=_norm(row.get('UMG CODE', row.get('UMG_CODE',''))), UMG_이름=_norm(row.get('UMG 이름', row.get('UMG_이름',''))),
                                자재속성그룹=_norm(row.get('자재속성그룹','')), 자재속성그룹명=_norm(row.get('자재속성그룹명','')),
                                자재속성그룹_SPEC=_norm(row.get('자재속성그룹 SPEC', row.get('자재속성그룹_SPEC',''))), 재료비CG=_norm(row.get('재료비CG','')),
                                대상_여부=_norm(row.get('대상 여부', row.get('대상_여부','Y'))),
                                extraction_method=_norm(match.get('method','unknown')),
                                confidence=float(match.get('confidence',0.0)),
                                section_path=_norm(match.get('section_path','')),
                                section_title=_norm(match.get('section_title','')),
                                evidence=_norm(match.get('evidence','')),
                                label_context=_norm(match.get('label_context','')),
                            )
                            results.append(asdict(ex))
                        else:
                            results.append(asdict(self.mk_not_found_row(row, str(pos), reason='not_found', file_name=os.path.basename(html_path))))
                    pbar.update(1)
                except Exception as e:
                    self.logger.error(f"POS {pos} 처리 오류: {e}")
                    for _, row in group.iterrows():
                        results.append(asdict(self.mk_not_found_row(row, str(pos), reason='exception')))
                    pbar.update(1)

        df = pd.DataFrame(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            f.write('\t'.join(df.columns) + '\n')
            for _, row in df.iterrows():
                vals = ['' if (pd.isna(v) or v is None) else str(v) for v in row.values]
                f.write('\t'.join(vals) + '\n')
        return df


# =====================
# 병렬 워커 (LLM 비활성)
# =====================

def _cfg_from_dict(d: Dict[str, Any]) -> Config:
    fields = set(Config.__dataclass_fields__.keys())
    safe = {k: v for k, v in d.items() if k in fields}
    c = Config(**safe)
    for k, v in d.items():
        if k not in fields:
            setattr(c, k, v)
    return c

def _worker_process_pos_group(config_like: Dict[str, Any], base_folder: str, pos_number: str, rows: List[Dict[str, Any]]):
    cfg = _cfg_from_dict(config_like) if isinstance(config_like, dict) else config_like
    cfg.enable_llm = False           # 워커에서 LLM 비활성(안정/성능)
    cfg.enable_spacy_ie = False      # 워커에서는 spaCy도 비활성(메모리/성능)
    processor = POSSpecProcessor(cfg)
    if not rows:
        return []
    first_row = rows[0]
    ship_no = _norm(first_row.get('호선', first_row.get('선종', '')))
    group_name = _norm(first_row.get('자재속성그룹명',''))
    hint_file = _norm(first_row.get('file_name',''))
    html_path = processor.finder.find_html_file(str(pos_number), ship_no, group_name, base_folder, hint_file)
    results: List[Dict[str, Any]] = []
    if not html_path:
        for r in rows:
            results.append(asdict(processor.mk_not_found_row(pd.Series(r), str(pos_number), reason='not_found_file')))
        return results
    sections, meta = processor.parser.parse_html_file(html_path)
    raw_html = meta.get("raw_html","")
    raw_text = re.sub(r'<[^>]+>', ' ', raw_html)
    if not sections:
        for r in rows:
            results.append(asdict(processor.mk_not_found_row(pd.Series(r), str(pos_number), reason='parse_failed', file_name=os.path.basename(html_path))))
        return results
    for r in rows:
        spec_name = _norm(r.get('관리Spec명',''))
        input_val = _norm(r.get('Spec값', r.get('입력_Spec값','')))
        if not spec_name:
            results.append(asdict(processor.mk_not_found_row(pd.Series(r), str(pos_number), reason='skip', file_name=os.path.basename(html_path))))
            continue
        match = processor.matcher.choose_by_consensus(sections, spec_name, raw_html, raw_text, input_value=input_val, use_llm_extractor=False)
        if match and match.get('found'):
            results.append(asdict(SpecExtraction(
                작업일자=datetime.now().strftime('%Y-%m-%d'), 작업자='시스템',
                file_name=os.path.basename(html_path),
                자재번호=_norm(r.get('자재번호','')),
                POS=str(pos_number), 관리Spec명=spec_name,
                Spec값=_norm(match.get('value','')), 단위=_norm(match.get('unit','')),
                PMG_CODE=_norm(r.get('PMG CODE', r.get('PMG_CODE',''))), PMG_이름=_norm(r.get('PMG 이름', r.get('PMG_이름',''))),
                UMG_CODE=_norm(r.get('UMG CODE', r.get('UMG_CODE',''))), UMG_이름=_norm(r.get('UMG 이름', r.get('UMG_이름',''))),
                자재속성그룹=_norm(r.get('자재속성그룹','')), 자재속성그룹명=_norm(r.get('자재속성그룹명','')),
                자재속성그룹_SPEC=_norm(r.get('자재속성그룹 SPEC', r.get('자재속성그룹_SPEC',''))), 재료비CG=_norm(r.get('재료비CG','')),
                대상_여부=_norm(r.get('대상 여부', r.get('대상_여부','Y'))),
                extraction_method=_norm(match.get('method','unknown')),
                confidence=float(match.get('confidence',0.0)),
                section_path=_norm(match.get('section_path','')),
                section_title=_norm(match.get('section_title','')),
                evidence=_norm(match.get('evidence','')),
                label_context=_norm(match.get('label_context','')),
            )))
        else:
            results.append(asdict(processor.mk_not_found_row(pd.Series(r), str(pos_number), reason='not_found', file_name=os.path.basename(html_path))))
    return results


# =====================
# 재시도/병합
# =====================

def process_with_retry(base_df: pd.DataFrame, base_folder: str, config: Config, first_pass_df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    fail_methods = {'not_found', 'not_found_file', 'parse_failed', 'exception'}
    key_cols = ['POS', '자재번호', '관리Spec명']

    def add_key(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for kc in key_cols:
            if kc not in df.columns:
                df[kc] = ''
            df[kc] = df[kc].astype(str)
        df['__key__'] = df[key_cols].agg('|'.join, axis=1)
        return df

    base_k = add_key(base_df)
    if first_pass_df is None or first_pass_df.empty:
        retry_df = base_k.copy()
    else:
        fp_k = add_key(first_pass_df)
        fp_fail = fp_k[(fp_k['extraction_method'].isin(fail_methods)) | (fp_k['Spec값'].astype(str).str.len() == 0)]
        fail_keys = set(fp_fail['__key__'].tolist())
        retry_df = base_k[base_k['__key__'].isin(fail_keys)].copy()

    if retry_df.empty:
        return first_pass_df

    logger.info(f"재시도 대상 행: {len(retry_df)}")
    results: List[Dict[str, Any]] = []
    grouped = retry_df.groupby('POS') if 'POS' in retry_df.columns else [("", retry_df)]
    with tqdm(total=getattr(grouped, 'ngroups', 1), desc='재시도') as pbar:
        for pos, grp in grouped:
            try:
                first_row = grp.iloc[0]
                ship_no = _norm(first_row.get('호선', first_row.get('선종', '')))
                group_name = _norm(first_row.get('자재속성그룹명',''))
                hint_file = _norm(first_row.get('file_name',''))
                html_path = FileFinder(config).find_html_file(str(pos), ship_no, group_name, base_folder, hint_file)
                if not html_path:
                    for _, row in grp.iterrows():
                        results.append(asdict(POSSpecProcessor(config).mk_not_found_row(row, str(pos), reason='not_found_file_retry')))
                    pbar.update(1); continue
                sections, meta = HierarchicalHTMLParser(config).parse_html_file(html_path)
                raw_html = meta.get("raw_html","")
                raw_text = re.sub(r'<[^>]+>', ' ', raw_html)
                matcher = HierarchicalMatcher(config)
                for _, row in grp.iterrows():
                    spec_name = _norm(row.get('관리Spec명',''))
                    if not spec_name:
                        results.append(asdict(POSSpecProcessor(config).mk_not_found_row(row, str(pos), reason='skip')))
                        continue
                    input_val = _norm(row.get('Spec값', row.get('입력_Spec값','')))
                    match = matcher.choose_by_consensus(sections, spec_name, raw_html, raw_text, input_value=input_val, use_llm_extractor=False)
                    if match and match.get('found'):
                        results.append(asdict(SpecExtraction(
                            작업일자=datetime.now().strftime('%Y-%m-%d'), 작업자='시스템', file_name=os.path.basename(html_path),
                            자재번호=_norm(row.get('자재번호','')), POS=str(pos), 관리Spec명=spec_name,
                            Spec값=_norm(match.get('value','')), 단위=_norm(match.get('unit','')),
                            PMG_CODE=_norm(row.get('PMG CODE', row.get('PMG_CODE',''))), PMG_이름=_norm(row.get('PMG 이름', row.get('PMG_이름',''))),
                            UMG_CODE=_norm(row.get('UMG CODE', row.get('UMG_CODE',''))), UMG_이름=_norm(row.get('UMG 이름', row.get('UMG_이름',''))),
                            자재속성그룹=_norm(row.get('자재속성그룹','')),
                            자재속성그룹명=_norm(row.get('자재속성그룹명','')),
                            자재속성그룹_SPEC=_norm(row.get('자재속성그룹 SPEC', row.get('자재속성그룹_SPEC',''))), 재료비CG=_norm(row.get('재료비CG','')),
                            대상_여부=_norm(row.get('대상 여부', row.get('대상_여부','Y'))),
                            extraction_method=_norm(match.get('method','unknown')),
                            confidence=float(match.get('confidence',0.0)),
                            section_path=_norm(match.get('section_path','')),
                            section_title=_norm(match.get('section_title','')),
                            evidence=_norm(match.get('evidence','')),
                            label_context=_norm(match.get('label_context','')),
                        )))
                    else:
                        results.append(asdict(POSSpecProcessor(config).mk_not_found_row(row, str(pos), reason='not_found')))
                pbar.update(1)
            except Exception:
                for _, row in grp.iterrows():
                    results.append(asdict(POSSpecProcessor(config).mk_not_found_row(row, str(pos), reason='exception_retry')))
                pbar.update(1)

    rp = pd.DataFrame(results)
    if first_pass_df is None or first_pass_df.empty:
        return rp

    bad = {'not_found', 'not_found_file', 'parse_failed', 'exception',
           'not_found_file_retry', 'exception_retry', 'skip'}
    m = rp['extraction_method'].astype(str)
    have_val = rp['Spec값'].astype(str).str.len() > 0
    rp_good = rp[(~m.isin(bad)) & have_val].copy()

    fp = first_pass_df.copy()
    for kc in key_cols:
        if kc not in fp.columns:
            fp[kc] = ''
        if kc not in rp_good.columns:
            rp_good[kc] = ''
    fp['__key__'] = fp[key_cols].agg('|'.join, axis=1)
    rp_good['__key__'] = rp_good[key_cols].agg('|'.join, axis=1)

    rp_best = (rp_good.sort_values(['__key__','confidence'], ascending=[True, False])
                     .drop_duplicates(['__key__'], keep='first'))
    fp = fp.set_index('__key__')
    rp_best = rp_best.set_index('__key__')

    inter = rp_best.index.intersection(fp.index)
    fp.loc[inter, :] = rp_best.loc[inter, fp.columns.intersection(rp_best.columns)]
    return fp.reset_index()


# =====================
# 메인 파이프라인
# =====================

def run_pipeline(config: Config) -> pd.DataFrame:
    setup_logging(config.log_path)
    logging.getLogger(__name__).info(
        f"OLLAMA_AVAILABLE={OLLAMA_AVAILABLE}, SPACY_AVAILABLE={SPACY_AVAILABLE}, "
        f"multiprocessing={config.enable_multiprocessing}, workers={config.num_workers}, "
        f"enable_llm={config.enable_llm}, llm_first={config.llm_first}, model={config.llm_model}, "
        f"enable_spacy_ie={config.enable_spacy_ie}, spacy_model={config.spacy_model_name}"
    )

    print("\nPOS Spec 추출/검증 — 합의+테이블+백링크+LLM-판사+spaCy IE")
    print("=" * 78)
    print("- TXT 입력(엑셀 복붙), 섹션 인지, 취소선 제거, WordSection1 스킵")
    print("- 라벨 느슨 매칭(RapidFuzz), 테이블 구조 인지, 값-백링크")
    print("- 단위 변환 수용(bar↔MPa, mm↔inch 등)")
    print("- LLM 판사 모드(선택), spaCy IE(PhraseMatcher+문장 창)")
    print("=" * 78)

    loader = TxtTableLoader(config)
    base_df = loader.load_spec_table(config.spec_table_path)

    # 1차 처리
    if config.enable_multiprocessing:
        grouped = base_df.groupby('POS') if 'POS' in base_df.columns else [("", base_df)]
        results: List[Dict[str, Any]] = []

        max_inflight = max(4, config.num_workers * 4)

        def wait_some(inflight, results, pbar):
            done = set()
            for fut in as_completed(inflight, timeout=None):
                try:
                    chunk = fut.result()
                    if chunk:
                        results.extend(chunk)
                except Exception as e:
                    logging.getLogger(__name__).error(f"병렬 작업 오류: {e}")
                done.add(fut)
                pbar.update(1)
                break
            return done, (inflight - done)

        def run_with_executor(executor_cls):
            inflight = set()
            total_jobs = getattr(grouped, 'ngroups', 1)
            with executor_cls(max_workers=config.num_workers) as ex:
                it = (base_df.groupby('POS') if 'POS' in base_df.columns else [("", base_df)])
                with tqdm(total=total_jobs, desc='병렬 처리') as pbar:
                    for pos, grp in it:
                        while len(inflight) >= max_inflight:
                            done, inflight = wait_some(inflight, results, pbar)
                        fut = ex.submit(_worker_process_pos_group, asdict(config), config.base_folder, str(pos), grp.to_dict('records'))
                        inflight.add(fut)
                    while inflight:
                        done, inflight = wait_some(inflight, results, pbar)

        try:
            run_with_executor(ProcessPoolExecutor)
        except BrokenProcessPool:
            logging.getLogger(__name__).warning("ProcessPool 깨짐 → ThreadPool로 폴백")
            results.clear()
            run_with_executor(ThreadPoolExecutor)

        first_pass_df = pd.DataFrame(results)
    else:
        processor = POSSpecProcessor(config)
        first_pass_df = processor.process_spec_table(config.spec_table_path, config.base_folder, config.output_path)

    # 2차 재시도
    final_df = process_with_retry(base_df, config.base_folder, config, first_pass_df)

    # 3) 검증/보정 컬럼 확장
    augmented_df = augment_with_validation(base_df, final_df)

    # 저장
    xlsx_path = config.xlsx_path or re.sub(r'\.txt$', '.xlsx', config.output_path)
    pkl_path = config.result_pkl_path or (config.output_path + '.pkl')
    save_outputs(augmented_df, config.output_path, xlsx_path, pkl_path)

    print(f"\n행 개수 (입력/1차/최종): {len(base_df)} / {len(first_pass_df)} / {len(augmented_df)}")
    print(f"결과 저장: {config.output_path} (및 .excel.tsv, .xlsx, .pkl)")
    if not augmented_df.empty:
        cnt = augmented_df['extraction_method'].value_counts(dropna=False)
        print("\n추출 방법별 분포:")
        for method, c in cnt.items():
            print(f"  {method}: {c} ({c/len(augmented_df)*100:.1f}%)")
        if '검증결과_유형' in augmented_df.columns:
            vcnt = augmented_df['검증결과_유형'].value_counts(dropna=False)
            print("\n검증결과 유형 분포:")
            for k, c in vcnt.items():
                print(f"  {k}: {c} ({c/len(augmented_df)*100:.1f}%)")
    return augmented_df


# =====================
# CLI/Notebook 빌더
# =====================

def build_config_from_args() -> Config:
    parser = argparse.ArgumentParser(description="POS Spec 검증/보정 파이프라인", add_help=True)
    parser.add_argument("--base-folder", type=str, default=r"C:\Users\20257071\Desktop\code\phase3_all_pos_html_merged\merged")
    parser.add_argument("--spec-table", type=str, default=r"C:\Users\20257071\Desktop\code\major data\spec추출.txt")
    parser.add_argument("--output", type=str, default=r"C:\Users\20257071\Desktop\code\result\output_hierarchical.txt")
    parser.add_argument("--pattern-db", type=str, default=r"C:\Users\20257071\Desktop\code\result\learned_patterns.pkl")
    parser.add_argument("--alias-db", type=str, default=r"C:\Users\20257071\Desktop\code\result\alias_db.pkl")
    parser.add_argument("--log-path", type=str, default=r"C:\Users\20257071\Desktop\code\result\run.log")
    parser.add_argument("--xlsx-path", type=str, default=r"C:\Users\20257071\Desktop\code\result\output_hierarchical.xlsx")
    parser.add_argument("--pkl-path", type=str, default=r"C:\Users\20257071\Desktop\code\result\results.pkl")
    parser.add_argument("--no-mp", action="store_true", help="멀티프로세싱 비활성화")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--llm-first", action="store_true")
    parser.add_argument("--no-llm-first", action="store_true")
    parser.add_argument("--enable-spacy", action="store_true")
    parser.add_argument("--disable-spacy", action="store_true")
    parser.add_argument("--spacy-model", type=str, default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.getLogger(__name__).warning(f"알 수 없는 인자 무시: {unknown}")

    cfg = Config(
        base_folder=args.base_folder,
        spec_table_path=args.spec_table,
        output_path=args.output,
        pattern_db_path=args.pattern_db,
        alias_db_path=args.alias_db,
        log_path=args.log_path,
        xlsx_path=args.xlsx_path,
        result_pkl_path=args.pkl_path,
        enable_multiprocessing=(not args.no_mp),
        num_workers=args.workers,
        enable_spacy_ie=not args.disable_spacy,
        spacy_model_name=args.spacy_model,
    )
    if args.enable_llm:
        cfg.enable_llm = True
    if args.disable_llm:
        cfg.enable_llm = False
    if args.llm_model:
        cfg.llm_model = args.llm_model
    if args.llm_first:
        cfg.llm_first = True
    if args.no_llm_first:
        cfg.llm_first = False
    if args.enable_spacy:
        cfg.enable_spacy_ie = True
    return cfg


def build_config_for_notebook(
    base_folder: str,
    spec_table: str,
    output: str,
    *,
    pattern_db: Optional[str] = None,
    alias_db: Optional[str] = None,
    log_path: Optional[str] = None,
    xlsx_path: Optional[str] = None,
    pkl_path: Optional[str] = None,
    llm_model: Optional[str] = None,
    enable_llm: bool = True,
    llm_first: bool = True,
    workers: int = 1,
    use_multiprocessing: bool = False,
    enable_spacy_ie: bool = True,
    spacy_model_name: Optional[str] = None,
) -> Config:
    cfg = Config(
        base_folder=base_folder,
        spec_table_path=spec_table,
        output_path=output,
        pattern_db_path=pattern_db or (Path(output).with_suffix('').as_posix() + "_patterns.pkl"),
        alias_db_path=alias_db or (Path(output).with_suffix('').as_posix() + "_alias.pkl"),
        log_path=log_path or (Path(output).with_suffix('').as_posix() + ".log"),
        xlsx_path=xlsx_path or (Path(output).with_suffix('.xlsx').as_posix()),
        result_pkl_path=pkl_path or (output + ".pkl"),
        enable_multiprocessing=use_multiprocessing,
        num_workers=workers,
        enable_llm=enable_llm,
        llm_first=llm_first,
        enable_spacy_ie=enable_spacy_ie,
        spacy_model_name=spacy_model_name,
    )
    if llm_model:
        cfg.llm_model = llm_model
    return cfg


# =====================
# __main__ (노트북 안전)
# =====================

if __name__ == "__main__":
    if is_notebook():
	cfg = build_config_for_notebook(
    		base_folder=r"C:\Users\20257071\Desktop\code\phase3_all_pos_html_merged\merged",
    		spec_table=r"C:\Users\20257071\Desktop\code\major data\spec추출.txt",
    		output=r"C:\Users\20257071\Desktop\code\major data\spaCyresult\output_hierarchical.txt",
    		llm_model="llama3.1:8b-instruct-q4_K_M",  # 설치된 llama3 model
    		enable_llm=True, llm_first=True,
   		workers=1, use_multiprocessing=False,
    		enable_spacy_ie=True, spacy_model_name=None,  # 'en_core_web_sm' 사용 가능 시 지정
	)
	res_df = run_pipeline(cfg)
    else:
        cfg = build_config_from_args()
        _ = run_pipeline(cfg)
