# Testing Instructions for v61 and v70 Extractors

## ⚠️ Important Notice

The previous "99.32% accuracy" test only verified HTMLChunkParser's key-value extraction.
The **full pipeline including LLM has NOT been tested** due to environment limitations.

## What Needs Testing

### 1. NOT_FOUND Handling

**Question**: Does the system properly mark specs as NOT_FOUND when they don't exist in POS documents, while correctly identifying "different expressions" via LLM?

**Test Method**:
```bash
# 1. Create test template with specs not in POS
python test_full_pipeline.py

# 2. Run extraction (requires Ollama + PostgreSQL)
python v61_extractor.py

# 3. Check results
grep "EXTRACTION_FAILED" extraction_result_*.json | wc -l
```

**Expected**:
- Specs truly missing from POS → `EXTRACTION_FAILED`
- Specs with different expressions → Found via LLM
- LLM correctly distinguishes between the two cases

### 2. Performance Test

**Goal**: 300,000 specs in 2-3 days

**Test Method**:
```bash
# Test on 100 specs first
time python v61_extractor.py  # Measure time

# Calculate
# Average time per spec = Total time / 100
# Estimated 300K time = Average × 300,000
```

**Target**:
- v61: < 3 days
- v70: < 2.5 days (with optimizations)

**Key Metrics**:
- Time per spec
- LLM call count
- Cache hit rate

### 3. v61 vs v70 Comparison

**Test Method**:
```bash
# Run both on same data
python v61_extractor.py > v61.log 2>&1
python v70_extractor.py > v70.log 2>&1

# Compare results
python compare_extraction_results.py extraction_result_v61.json extraction_result_v70.json
```

**Compare**:
1. Accuracy (found/not_found ratio)
2. Performance (time per spec)
3. LLM call count
4. Confidence scores
5. Extraction methods used

## Prerequisites

### 1. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull model (choose one)
ollama pull gemma3:27b     # For v61/v70 default
# or
ollama pull qwen2.5:32b    # Alternative

# Verify
ollama list
```

### 2. PostgreSQL Setup

Required tables:
- `pos_dict`: Glossary (umgv_desc ↔ pos_umgv_desc)
- `umgv_fin`: Spec value database
- `ext_tmpl`: Template data

```bash
# Test connection
psql -h <host> -U <user> -d <database> -c '\dt'

# Check data
psql -h <host> -U <user> -d <database> -c 'SELECT COUNT(*) FROM pos_dict;'
```

### 3. Python Environment

```bash
pip install pandas beautifulsoup4 psycopg2-binary requests torch sentence-transformers FlagEmbedding
```

## Running Tests

### Quick Start

```bash
# 1. Environment check
python test_full_pipeline.py

# 2. Small test (10 specs)
# Edit v61_extractor.py or v70_extractor.py
# Set: EXTRACTION_MODE = "light"
# Run
python v61_extractor.py

# 3. Check results
cat extraction_result_*.json | jq '.stats'
```

### Full Test

```bash
# 1. Prepare data
# - Ensure POS HTML files in /workspace/server/uploaded_files/
# - Ensure template data in PostgreSQL

# 2. Run v61
python v61_extractor.py

# 3. Run v70
python v70_extractor.py

# 4. Compare
python compare_extraction_results.py \
    extraction_result_v61_*.json \
    extraction_result_v70_*.json
```

## Understanding Results

### Output JSON Structure

```json
{
  "stats": {
    "total": 1000,
    "found": 850,
    "not_found": 100,
    "extraction_failed": 50
  },
  "total_time": 2500.0,
  "results": [
    {
      "umgv_code": "...",
      "umgv_desc": "Output",
      "pos_umgv_value": "500",
      "pos_umgv_uom": "kW",
      "_method": "rule+llm_validated",
      "_confidence": 0.95,
      "_evidence": "..."
    }
  ]
}
```

### Method Types

1. `rule`: Rule-based extraction only
2. `rule+llm_validated`: Rule + LLM validation
3. `rule+llm_corrected`: Rule extracted, LLM corrected
4. `llm_fallback`: LLM fallback after rule failed
5. `EXTRACTION_FAILED`: All methods failed
6. `FILE_NOT_FOUND`: HTML file not found

### Analyzing NOT_FOUND

```bash
# Count by method
cat extraction_result.json | jq '.results[] | ._method' | sort | uniq -c

# Find EXTRACTION_FAILED cases
cat extraction_result.json | jq '.results[] | select(._method == "EXTRACTION_FAILED") | {code: .umgv_code, name: .umgv_desc}'

# Check if they're truly missing or need LLM improvement
# - Manually verify in POS HTML
# - If present with different expression → LLM needs improvement
# - If truly absent → Correct behavior
```

## Performance Analysis

### Calculate Metrics

```python
import json

with open('extraction_result.json') as f:
    data = json.load(f)

total_time = data['total_time']
total_specs = data['stats']['total']

avg_time = total_time / total_specs
print(f"Average time/spec: {avg_time:.2f}s")

# Estimate 300K
time_300k_hours = (avg_time * 300000) / 3600
time_300k_days = time_300k_hours / 24
print(f"Estimated 300K time: {time_300k_days:.1f} days")

# Check against target
if time_300k_days <= 3:
    print("✓ Target met!")
else:
    print(f"✗ Need {time_300k_days - 3:.1f} days improvement")
```

### Bottleneck Identification

1. **LLM calls**: Most expensive operation
   - Reduce with better rule-based extraction
   - Skip validation when confidence is high (v70 optimization)

2. **PostgreSQL queries**: Second most expensive
   - Use connection pooling
   - Optimize queries with indexes

3. **HTML parsing**: Usually fast
   - Cache parsed results when processing multiple specs from same file

## Troubleshooting

### Ollama Not Working

```bash
# Check if running
ps aux | grep ollama

# Check port
curl http://127.0.0.1:11434/api/tags

# Restart
killall ollama
ollama serve
```

### PostgreSQL Connection Failed

```bash
# Check connectivity
psql -h <host> -U <user> -d <database>

# Check credentials in code
# v61_extractor.py: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

### Low Accuracy

Possible causes:
1. LLM model not suitable → Try different model
2. Template specs don't match POS → Verify template quality
3. HTML parsing issues → Check HTMLChunkParser improvements
4. LLM prompt needs tuning → Adjust prompts in LLMFallbackExtractor

### Slow Performance

Optimization strategies:
1. Increase parallelization (`num_workers`)
2. Reduce LLM calls (better rule-based extraction)
3. Enable `skip_llm_validation_on_rule_success` (v70)
4. Use faster model (but may reduce accuracy)
5. Optimize PostgreSQL queries

## Next Steps

After completing these tests:

1. **Analyze results**
   - Accuracy vs target
   - Performance vs target
   - Error patterns

2. **Identify improvements**
   - LLM prompt tuning
   - Rule-based extraction enhancement
   - Performance optimization

3. **Iterate**
   - Implement improvements
   - Re-test
   - Repeat until targets met

## Questions?

Refer to:
- `HONEST_EVALUATION_REPORT.md`: What was actually tested
- `IMPROVEMENT_SUMMARY.md`: HTMLChunkParser improvements
- `v61_extractor.py` docstring: Full pipeline documentation
