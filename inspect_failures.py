#!/usr/bin/env python3
"""
Inspect specific failure cases in HTML files
"""

from bs4 import BeautifulSoup
import json
import re

def inspect_html_for_spec(html_path, spec_name, expected_value):
    """Inspect HTML to find how the spec appears"""
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    print(f"\nSearching for '{spec_name}' with value '{expected_value}' in {html_path}")
    print("="*80)

    # Find all tables
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")

    # Search for the spec_name in the HTML
    spec_norm = re.sub(r'\s+', '', spec_name.upper())

    for table_idx, table in enumerate(tables):
        rows = table.find_all('tr')

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])

            # Check each cell for the spec name
            for cell_idx, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                cell_norm = re.sub(r'\s+', '', cell_text.upper())

                if spec_norm in cell_norm or cell_norm in spec_norm:
                    print(f"\nFOUND in Table {table_idx}, Row {row_idx}, Cell {cell_idx}:")
                    print(f"Cell text: {cell_text[:200]}")

                    # Get all cells in this row
                    print(f"\nAll cells in this row ({len(cells)} cells):")
                    for i, c in enumerate(cells):
                        print(f"  Cell {i}: {c.get_text(strip=True)[:100]}")

                    return True

    print("NOT FOUND in any table")
    return False


def main():
    # Load ground truth
    with open('ground_truth_auto.json', 'r') as f:
        gt = json.load(f)

    # Failures to inspect
    failures = [
        ('2606-POS-0057101_001_02_A4(27).html', 'MainComponent for each unit', '1'),
        ('2606-POS-0057101_001_02_A4(27).html', 'Sub-cooler', '100'),
        ('2606-POS-0094102_000_02_A4_CONSOLE.html', 'Note', '1'),
        ('2606-POS-0094102_000_02_A4_CONSOLE.html', 'Gooseneck light', '3'),
    ]

    for html_file, spec_name, expected_value in failures:
        inspect_html_for_spec(html_file, spec_name, expected_value)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
