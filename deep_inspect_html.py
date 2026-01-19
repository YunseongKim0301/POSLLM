#!/usr/bin/env python3
"""
Deep inspection of HTML to understand failure cases
"""

from bs4 import BeautifulSoup
import json

def inspect_html_raw(html_file):
    """Look at raw HTML structure for failing specs"""
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Get ground truth
    with open('ground_truth_auto.json', 'r') as f:
        gt = json.load(f)

    failing_specs = []
    if html_file in gt:
        # We know from previous analysis which ones failed
        if '0057101' in html_file:
            failing_specs = ['MainComponent for each unit', 'Sub-cooler', 'Maincomponent',
                           'Accessoriesfor AHU', 'Starter', 'Air supplydevices']
        elif '0094102' in html_file:
            failing_specs = ['Note', 'Gooseneck light']

    print(f"\n{'='*80}")
    print(f"Inspecting: {html_file}")
    print(f"{'='*80}")

    # Find raw text containing these specs
    for spec in failing_specs[:3]:  # Just first 3 to keep output manageable
        print(f"\nSearching for: '{spec}'")
        print("-"*80)

        # Get ground truth info
        gt_items = [item for item in gt[html_file] if item['spec_name'] == spec]
        if gt_items:
            gt_item = gt_items[0]
            print(f"GT Value: {gt_item['value']} {gt_item['unit']}")
            print(f"GT Raw (first 200 chars): {gt_item['raw_text'][:200]}")

        # Search in HTML
        found_in_html = False
        for table in soup.find_all('table'):
            table_text = table.get_text()

            # Simple substring search
            if spec.lower().replace(' ', '') in table_text.lower().replace(' ', ''):
                found_in_html = True
                print("\nFOUND in table!")

                # Get the specific rows
                rows = table.find_all('tr')
                for row_idx, row in enumerate(rows):
                    row_text = row.get_text()
                    if spec.lower().replace(' ', '') in row_text.lower().replace(' ', ''):
                        cells = row.find_all(['td', 'th'])
                        print(f"\n  Row {row_idx} ({len(cells)} cells):")
                        for i, cell in enumerate(cells):
                            cell_text = cell.get_text(strip=True)
                            if len(cell_text) > 100:
                                print(f"    Cell {i} (LONG, {len(cell_text)} chars): {cell_text[:100]}...")
                            else:
                                print(f"    Cell {i}: {cell_text}")
                        break
                break

        if not found_in_html:
            print("NOT FOUND in HTML with simple search")
            print("This might be a ground truth error!")


if __name__ == "__main__":
    files = [
        '2606-POS-0057101_001_02_A4(27).html',
        '2606-POS-0094102_000_02_A4_CONSOLE.html'
    ]

    for f in files:
        inspect_html_raw(f)
