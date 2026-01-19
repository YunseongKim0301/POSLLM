#!/usr/bin/env python3
"""
Check what was actually extracted for specific failing cases
"""

import sys
sys.path.insert(0, '/home/user/POSLLM')

from v61_extractor import HTMLChunkParser
import re

def normalize(text):
    """Normalize for comparison"""
    return re.sub(r'\s+', '', text.upper())

def check_extractions(html_file, search_keys):
    """Check what was extracted for specific keys"""
    parser = HTMLChunkParser(file_path=html_file)

    print(f"\n{'='*80}")
    print(f"File: {html_file}")
    print(f"Total pairs extracted: {len(parser.kv_pairs)}")
    print(f"{'='*80}")

    for search_key in search_keys:
        print(f"\nSearching for: '{search_key}'")
        print("-"*80)

        search_norm = normalize(search_key)

        # Find matches
        matches = []
        for kv in parser.kv_pairs:
            key_norm = normalize(kv['key'])

            # Exact match
            if search_norm == key_norm:
                matches.append(('EXACT', kv))
            # Substring match
            elif search_norm in key_norm or key_norm in search_norm:
                matches.append(('SUBSTRING', kv))
            # Fuzzy (word-based)
            else:
                search_words = set(re.findall(r'[A-Z0-9]{2,}', search_norm))
                key_words = set(re.findall(r'[A-Z0-9]{2,}', key_norm))
                if search_words and key_words:
                    common = search_words & key_words
                    if len(common) / max(len(search_words), len(key_words)) > 0.5:
                        matches.append(('FUZZY', kv))

        if matches:
            print(f"Found {len(matches)} match(es):")
            for match_type, kv in matches[:5]:  # Show top 5
                print(f"\n  [{match_type}]")
                print(f"  Key: {kv['key'][:80]}")
                print(f"  Val: {kv['value'][:80]}")
        else:
            print("NO MATCHES")

            # Show similar keys
            similar = []
            for kv in parser.kv_pairs:
                key_norm = normalize(kv['key'])
                # Check for partial word matches
                for word in re.findall(r'[A-Z]{3,}', search_norm):
                    if word in key_norm:
                        similar.append(kv)
                        break

            if similar:
                print(f"\nPossibly similar ({len(similar)} found):")
                for kv in similar[:5]:
                    print(f"  Key: {kv['key'][:80]}")
                    print(f"  Val: {kv['value'][:80]}")


if __name__ == "__main__":
    # Check specific failures
    check_extractions(
        '2606-POS-0057101_001_02_A4(27).html',
        ['Sub-cooler', 'MainComponent for each unit', 'Maincomponent']
    )

    check_extractions(
        '2606-POS-0094102_000_02_A4_CONSOLE.html',
        ['Note', 'Gooseneck light', 'Goose neck light']
    )
