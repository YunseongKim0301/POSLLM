#!/usr/bin/env python3
"""
Test the specific improvements made to v61_extractor.py
- Spec prefix removal ("SWL 6 tonnes" -> "6 tonnes")
- Hierarchical key parsing ("1_Capacity_Air volume(m3/h)#")
- KV Direct Matching optimization
- Fuzzy matching enhancements
"""

import sys
from pathlib import Path

# Import the HTMLChunkParser
from v61_extractor import HTMLChunkParser

def test_spec_prefix_removal():
    """Test that spec prefixes are removed from values"""
    print("\n" + "="*80)
    print("TEST 1: Spec Prefix Removal")
    print("="*80)

    parser = HTMLChunkParser()

    test_cases = [
        ("SWL 6 tonnes", ("6", "ton")),
        ("MCR 1000 kW", ("1000", "kW")),
        ("CAPACITY 250 m3/h", ("250", "m3/h")),
        ("6 tonnes", ("6", "ton")),  # No prefix
    ]

    passed = 0
    failed = 0

    for input_val, expected in test_cases:
        result = parser._parse_value_unit(input_val)
        # Check if value matches (unit might be partial match)
        if result[0] == expected[0]:
            print(f"✅ PASS: '{input_val}' -> value='{result[0]}', unit='{result[1]}'")
            passed += 1
        else:
            print(f"❌ FAIL: '{input_val}' -> expected value='{expected[0]}', got '{result[0]}'")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_hierarchical_key_parsing():
    """Test hierarchical key parsing"""
    print("\n" + "="*80)
    print("TEST 2: Hierarchical Key Parsing")
    print("="*80)

    parser = HTMLChunkParser()

    test_cases = [
        ("1_Capacity_Air volume(m3/h)#", {
            'index': '1',
            'spec': 'CAPACITY',
            'equipment': 'Air volume',
            'unit': 'm3/h'
        }),
        ("Motor Power_Main Engine", {
            'index': '',
            'spec': 'MOTOR POWER',
            'equipment': 'Main Engine',
            'unit': ''
        }),
        ("CAPACITY(m³/h)", {
            'index': '',
            'spec': 'CAPACITY',
            'equipment': '',
            'unit': 'm³/h'
        }),
    ]

    passed = 0
    failed = 0

    for key, expected in test_cases:
        result = parser._parse_hierarchical_key(key)
        if result == expected:
            print(f"✅ PASS: '{key}'")
            print(f"   → {result}")
            passed += 1
        else:
            print(f"❌ FAIL: '{key}'")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_fuzzy_matching():
    """Test fuzzy matching score computation"""
    print("\n" + "="*80)
    print("TEST 3: Fuzzy Matching")
    print("="*80)

    parser = HTMLChunkParser()

    test_cases = [
        # (key, spec_name, equipment, expected_score_range)
        ("MOTOR POWER", "MOTOR POWER", "", (0.65, 0.75)),  # Spec match only (40% + neutral bonuses)
        ("Motor Power_Main Engine", "MOTOR POWER", "Main Engine", (0.80, 0.90)),  # Spec + equipment match
        ("Motor_Power", "MOTOR POWER", "", (0.45, 0.55)),  # Token overlap with separator
        ("CAPACITY", "FLOW RATE", "", (0.0, 0.35)),  # No match
    ]

    passed = 0
    failed = 0

    for key, spec, equip, (min_score, max_score) in test_cases:
        score = parser._compute_fuzzy_match_score(key, spec, equip)
        if min_score <= score <= max_score:
            print(f"✅ PASS: key='{key}', spec='{spec}' -> score={score:.2f}")
            passed += 1
        else:
            print(f"❌ FAIL: key='{key}', spec='{spec}' -> score={score:.2f} (expected {min_score}-{max_score})")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_kv_index():
    """Test KV Direct Matching index"""
    print("\n" + "="*80)
    print("TEST 4: KV Direct Matching Index")
    print("="*80)

    # Create a simple HTML table
    html = """
    <table>
        <tr><td>CAPACITY</td><td>100 m3/h</td></tr>
        <tr><td>MOTOR POWER</td><td>50 kW</td></tr>
        <tr><td>1_Voltage_Main(V)#</td><td>440 V</td></tr>
    </table>
    """

    parser = HTMLChunkParser(html_content=html)

    print(f"Extracted {len(parser.kv_pairs)} KV pairs")
    print(f"Index contains {len(parser.kv_index)} entries")

    # Test index lookup
    if "CAPACITY" in parser.kv_index:
        print("✅ PASS: 'CAPACITY' found in index")
        kv = parser.kv_index["CAPACITY"]
        print(f"   Value: {kv['value']}")
    else:
        print("❌ FAIL: 'CAPACITY' not found in index")
        return False

    if "VOLTAGE" in parser.kv_index:
        print("✅ PASS: 'VOLTAGE' found in index (from hierarchical key)")
        kv = parser.kv_index["VOLTAGE"]
        print(f"   Value: {kv['value']}")
    else:
        print("❌ FAIL: 'VOLTAGE' not found in index")
        return False

    print(f"\nResults: All tests passed")
    return True


def test_value_verification():
    """Test value verification in document"""
    print("\n" + "="*80)
    print("TEST 5: Value Verification in Document")
    print("="*80)

    html = """
    <html>
    <body>
        <p>The capacity is 1,000 m3/h</p>
        <p>Motor power: 50 kW</p>
        <p>Temperature: 3.5 °C</p>
    </body>
    </html>
    """

    parser = HTMLChunkParser(html_content=html)

    test_cases = [
        ("1000", "m3/h", True),  # Should find "1,000"
        ("50", "kW", True),      # Should find "50"
        ("3.5", "°C", True),     # Should find "3.5"
        ("9999", "kW", False),   # Should not find
    ]

    passed = 0
    failed = 0

    for value, unit, expected in test_cases:
        result = parser.verify_value_in_document(value, unit)
        if result == expected:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"{status}: verify_value_in_document('{value}', '{unit}') = {result} (expected {expected})")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Testing v61_extractor.py Improvements")
    print("="*80)

    all_passed = True

    all_passed &= test_spec_prefix_removal()
    all_passed &= test_hierarchical_key_parsing()
    all_passed &= test_fuzzy_matching()
    all_passed &= test_kv_index()
    all_passed &= test_value_verification()

    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
