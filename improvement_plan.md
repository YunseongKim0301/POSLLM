# Improvement Plan for v61 and v70 Extractors

## Current Status
- v61 Accuracy: **99.09%** (869/877)
- v70 Accuracy: **99.09%** (869/877)
- **Already exceeds 90% target**, but aiming for 100%

## Failure Analysis

### Failures Identified (8 total)
1. **2606-POS-0057101_001_02_A4(27).html** (6 failures):
   - "Sub-cooler": HTML present but not extracted (value too long: 324 chars > 200 limit)
   - "MainComponent for each unit": NOT in HTML (ground truth error)
   - "Maincomponent": NOT in HTML (ground truth error)
   - "Accessoriesfor AHU": Needs investigation
   - "Starter": Needs investigation
   - "Air supplydevices": Needs investigation

2. **2606-POS-0094102_000_02_A4_CONSOLE.html** (2 failures):
   - "Note": HTML present but extraction complex
   - "Gooseneck light": Actually extracted as "Goose neck light" (normalization issue)

## Root Causes

### 1. **Value Length Limit (200 chars)**
- **Location**: `_extract_vertical_kv_table()` line 4538
- **Impact**: Long but valid values are filtered out
- **Example**: "Sub-cooler" with 324 char description containing "100%"

### 2. **Key Length Limit (150 chars)**
- **Location**: `_extract_vertical_kv_table()` line 4525
- **Impact**: Potentially valid long keys filtered out

### 3. **Ground Truth Quality**
- Some specs in ground truth don't exist in HTML
- Auto-generation script created false positives

## Universal Improvement Strategy

### Improvement 1: Intelligent Long Value Handling
**Principle**: Don't discard long values; extract meaningful info from them

**Approach**:
1. For values > 200 chars, scan for number+unit patterns
2. Extract first significant number+unit found
3. Keep confidence score lower for extracted values
4. Maximum value length: 1000 chars (safety limit)

**Code Location**: `_extract_vertical_kv_table()`

**Benefits**:
- Catches cases like "Sub-cooler" (100% in long text)
- Universal: works for any long descriptive text with embedded values
- No pattern-specific logic

### Improvement 2: Enhanced Key Normalization
**Principle**: Flexible matching while avoiding false positives

**Approach**:
1. Normalize spacing: "Gooseneck" ↔ "Goose neck"
2. CamelCase handling: "MainComponent" → "Main Component"
3. Already implemented in v61, ensure it's used consistently

**Code Location**: `aggressive_normalize()` and `fuzzy_match_keys()`

**Benefits**:
- Solves "Gooseneck light" mismatch
- Universal: handles various spacing conventions
- No document-specific patterns

### Improvement 3: Multi-column Value Extraction
**Principle**: When first value is too long or complex, check adjacent cells

**Approach**:
1. If Cell[i+1] is very long (>500 chars), also try Cell[i+2], Cell[i+3]
2. Extract number+unit from any of these cells
3. Prefer shorter, cleaner values over long ones

**Code Location**: `_extract_vertical_kv_table()`

**Benefits**:
- Handles complex table structures
- Universal: works for multi-column tables
- No specific chunk patterns

### Improvement 4: Value Extraction from Long Text
**Principle**: Extract structured data even from prose

**Approach**:
1. Add utility function: `extract_value_from_long_text(text)`
2. Pattern matching for:
   - "X%" → X with unit "%"
   - "X units" → X with unit "units"
   - "X bar/psi/°C/etc" → X with detected unit
3. Return first match found

**Benefits**:
- Universal value extraction
- Works on any descriptive text
- No document-specific logic

## Implementation Priority

1. ✅ **HIGH**: Long value handling (Improvement 1)
   - Immediate impact on "Sub-cooler" type cases
   - Simple to implement
   - Low risk

2. ✅ **MEDIUM**: Multi-column extraction (Improvement 3)
   - Broader coverage
   - Moderate complexity

3. ✅ **LOW**: Enhanced normalization (Improvement 2)
   - Already mostly covered
   - Fine-tuning needed

## Expected Results

After implementing these improvements:
- **Sub-cooler**: Will be extracted (long value handling)
- **Gooseneck light**: Better matching (normalization)
- **Note**: Potentially extracted (multi-column)
- **Overall accuracy**: Target **99.5%+** (approaching 100%)

## Non-Changes (Important)

What we will **NOT** do:
- ❌ Pattern-specific extraction for certain documents
- ❌ Hardcoded rules for specific spec names
- ❌ Document structure assumptions
- ❌ Chunk-specific string patterns

All improvements are **universal and generalizable**.
