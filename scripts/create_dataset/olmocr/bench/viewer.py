#!/usr/bin/env python3
import json
import sys
import os
import re
import argparse
import requests

from collections import defaultdict
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from olmocr.data.renderpdf import render_pdf_to_base64png


def parse_rules_file(file_path):
    """Parse the rules file and organize rules by PDF."""
    pdf_rules = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                rule = json.loads(line)
                # Add checked field if it doesn't exist
                if 'checked' not in rule:
                    rule['checked'] = None
                    
                if 'pdf' in rule:
                    pdf_rules[rule['pdf']].append(rule)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")
    
    return pdf_rules


def get_rule_html(rule, rule_index):
    """Generate HTML representation for a rule with interactive elements."""
    rule_type = rule.get('type', 'unknown')
    rule_id = f"rule-{rule_index}"
    
    # Determine status button class based on 'checked' value
    checked_status = rule.get('checked')
    thumbs_up_class = "active" if checked_status == "verified" else ""
    thumbs_down_class = "active" if checked_status == "rejected" else ""
    
    # Create thumbs up/down buttons
    status_button = f"""
        <div class="status-control">
            <button class="status-button thumbs-up {thumbs_up_class}" 
                    data-rule-id="{rule_id}" 
                    data-action="verified"
                    onclick="toggleStatus(this)"></button>
            <button class="status-button thumbs-down {thumbs_down_class}" 
                    data-rule-id="{rule_id}" 
                    data-action="rejected"
                    onclick="toggleStatus(this)"></button>
        </div>
    """
    
    # Create HTML based on rule type
    if rule_type == 'present':
        return f"""
        <tr class="rule-row present-rule" data-rule-id="{rule_id}" data-rule-index="{rule_index}">
            <td>{status_button}</td>
            <td><span class="rule-type present">PRESENT</span></td>
            <td>
                <div class="editable-text" 
                     contenteditable="true" 
                     data-rule-id="{rule_id}" 
                     data-field="text"
                     onblur="updateRuleText(this)">{rule.get('text', '')}</div>
            </td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    elif rule_type == 'absent':
        return f"""
        <tr class="rule-row absent-rule" data-rule-id="{rule_id}" data-rule-index="{rule_index}">
            <td>{status_button}</td>
            <td><span class="rule-type absent">ABSENT</span></td>
            <td>
                <div class="editable-text" 
                     contenteditable="true" 
                     data-rule-id="{rule_id}" 
                     data-field="text"
                     onblur="updateRuleText(this)">{rule.get('text', '')}</div>
            </td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    elif rule_type == 'order':
        return f"""
        <tr class="rule-row order-rule" data-rule-id="{rule_id}" data-rule-index="{rule_index}">
            <td>{status_button}</td>
            <td><span class="rule-type order">ORDER</span></td>
            <td>
                <p><strong>Before:</strong> 
                    <span class="editable-text" 
                          contenteditable="true" 
                          data-rule-id="{rule_id}" 
                          data-field="before"
                          onblur="updateRuleText(this)">{rule.get('before', '')}</span>
                </p>
                <p><strong>After:</strong> 
                    <span class="editable-text" 
                          contenteditable="true" 
                          data-rule-id="{rule_id}" 
                          data-field="after"
                          onblur="updateRuleText(this)">{rule.get('after', '')}</span>
                </p>
            </td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    else:
        return f"""
        <tr class="rule-row unknown-rule" data-rule-id="{rule_id}" data-rule-index="{rule_index}">
            <td>{status_button}</td>
            <td><span class="rule-type unknown">UNKNOWN</span></td>
            <td>Unknown rule type: {rule_type}</td>
            <td></td>
        </tr>
        """


def generate_html(pdf_rules, rules_file_path):
    """Generate the HTML page with PDF renderings and interactive rules."""
    # Limit to 10 unique PDFs
    pdf_names = list(pdf_rules.keys())[:10]
    
    # Prepare rules data for JavaScript
    all_rules = []
    for pdf_name in pdf_names:
        all_rules.extend(pdf_rules[pdf_name])
    
    rules_json = json.dumps(all_rules)
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive PDF Rules Visualizer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            
            .container {
                max-width: 1920px;
                margin: 0 auto;
            }
            
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            
            .pdf-container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
                overflow: hidden;
            }
            
            .pdf-header {
                background-color: #4a6fa5;
                color: white;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            
            .pdf-content {
                display: flex;
                flex-direction: row;
                padding: 20px;
            }
            
            @media (max-width: 1200px) {
                .pdf-content {
                    flex-direction: column;
                }
            }
            
            .pdf-image {
                flex: 0 0 50%;
                max-width: 800px;
                text-align: center;
                padding-right: 20px;
            }
            
            .pdf-image img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            
            .rules-container {
                flex: 1;
                overflow: auto;
            }
            
            .rules-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .rules-table th {
                background-color: #4a6fa5;
                color: white;
                padding: 10px;
                text-align: left;
            }
            
            .rules-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }
            
            .rule-type {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            
            .present {
                background-color: #28a745;
            }
            
            .absent {
                background-color: #dc3545;
            }
            
            .order {
                background-color: #fd7e14;
            }
            
            .unknown {
                background-color: #6c757d;
            }
            
            .rule-row:hover {
                background-color: #f8f9fa;
            }
            
            /* New styles for interactive elements */
            .editable-text {
                min-height: 20px;
                padding: 5px;
                border-radius: 4px;
                border: 1px solid transparent;
                transition: border-color 0.2s;
            }
            
            .editable-text:hover {
                border-color: #ccc;
                background-color: #f8f9fa;
            }
            
            .editable-text:focus {
                outline: none;
                border-color: #4a6fa5;
                background-color: #fff;
            }
            
            .status-control {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
            }
            
            .status-button {
                width: 36px;
                height: 36px;
                border-radius: 4px;
                border: 1px solid #ccc;
                background-color: #f8f9fa;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .status-button:hover {
                border-color: #999;
                background-color: #e9ecef;
            }
            
            .thumbs-up:before {
                content: "👍";
                font-size: 18px;
                opacity: 0.5;
            }
            
            .thumbs-down:before {
                content: "👎";
                font-size: 18px;
                opacity: 0.5;
            }
            
            .thumbs-up.active {
                background-color: #28a745;
                border-color: #28a745;
            }
            
            .thumbs-up.active:before {
                opacity: 1;
                color: white;
            }
            
            .thumbs-down.active {
                background-color: #dc3545;
                border-color: #dc3545;
            }
            
            .thumbs-down.active:before {
                opacity: 1;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive PDF Rules Visualizer</h1>
    """
    
    # Global rule index for unique IDs
    rule_index = 0
    
    for pdf_name in pdf_names:
        rules = pdf_rules[pdf_name]
        
        # Render the PDF (first page only) from the /pdfs folder
        try:
            pdf_path = os.path.join(os.path.dirname(rules_file_path), 'pdfs', pdf_name)
            base64_img = render_pdf_to_base64png(pdf_path, 0)
            img_html = f'<img src="data:image/png;base64,{base64_img}" alt="{pdf_name}">'
        except Exception as e:
            img_html = f'<div class="error">Error rendering PDF: {str(e)}</div>'
        
        html += f"""
        <div class="pdf-container">
            <div class="pdf-header">{pdf_name}</div>
            <div class="pdf-content">
                <div class="pdf-image">
                    {img_html}
                </div>
                <div class="rules-container">
                    <table class="rules-table">
                        <thead>
                            <tr>
                                <th>Status</th>
                                <th>Type</th>
                                <th>Content</th>
                                <th>Parameters</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for rule in rules:
            html += get_rule_html(rule, rule_index)
            rule_index += 1
        
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
    
    # Add JavaScript to manage interactivity and datastore integration
    html += f"""
        </div>
        
        <script>
            // Store all rules data (initially injected from the JSON file)
            let rulesData = {rules_json};
            
            // Function to toggle status button
            function toggleStatus(button) {{
                const ruleRow = button.closest('.rule-row');
                const ruleIndex = parseInt(ruleRow.dataset.ruleIndex);
                const action = button.dataset.action;
                const currentState = rulesData[ruleIndex].checked;
                const newState = (currentState === action) ? null : action;
                rulesData[ruleIndex].checked = newState;
                
                // Update UI for status buttons
                const buttons = ruleRow.querySelectorAll('.status-button');
                buttons.forEach(btn => {{
                    if (btn.dataset.action === newState) {{
                        btn.classList.add('active');
                    }} else {{
                        btn.classList.remove('active');
                    }}
                }});
                
                // Upload updated data to datastore
                uploadRulesData();
                outputJSON();
            }}
            
            // Function to update rule text
            function updateRuleText(element) {{
                const ruleRow = element.closest('.rule-row');
                const ruleIndex = parseInt(ruleRow.dataset.ruleIndex);
                const field = element.dataset.field;
                const newText = element.innerText.trim();
                
                // Update the rules data
                rulesData[ruleIndex][field] = newText;
                
                // Upload updated data to datastore
                uploadRulesData();
                outputJSON();
            }}
            
            // Function to output JSONL to console
            function outputJSON() {{
                console.clear();
                console.log("Updated JSONL:");
                rulesData.forEach(rule => {{
                    console.log(JSON.stringify(rule));
                }});
            }}
            
            // Function to upload rulesData to datastore using putDatastore
            async function uploadRulesData() {{
                try {{
                    await putDatastore(rulesData);
                    console.log("Datastore updated successfully");
                }} catch (error) {{
                    console.error("Failed to update datastore", error);
                }}
            }}
            
            // Function to update UI from rulesData (used after fetching datastore state)
            function updateUIFromRulesData() {{
                document.querySelectorAll('.rule-row').forEach(ruleRow => {{
                    const ruleIndex = parseInt(ruleRow.dataset.ruleIndex);
                    const rule = rulesData[ruleIndex];
                    // Update status buttons
                    const buttons = ruleRow.querySelectorAll('.status-button');
                    buttons.forEach(btn => {{
                        if (btn.dataset.action === rule.checked) {{
                            btn.classList.add('active');
                        }} else {{
                            btn.classList.remove('active');
                        }}
                    }});
                    // Update editable text fields
                    ruleRow.querySelectorAll('.editable-text').forEach(div => {{
                        const field = div.dataset.field;
                        if (rule[field] !== undefined) {{
                            div.innerText = rule[field];
                        }}
                    }});
                }});
            }}
            
            // On page load, fetch data from the datastore and update UI accordingly
            document.addEventListener('DOMContentLoaded', async function() {{
                try {{
                    const datastoreState = await fetchDatastore();
                    if (datastoreState.length) {{
                        rulesData = datastoreState;
                        updateUIFromRulesData();
                        outputJSON();
                    }}
                }} catch (error) {{
                    console.error("Error fetching datastore", error);
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html

def get_page_datastore(html: str):
    """
    Fetch the JSON datastore from the presigned URL.
    Returns a dict. If any error or no content, returns {}.
    """
    match = re.search(r"const presignedGetUrl = \"(.*?)\";", html)
    if not match:
        return None
    presigned_url = match.group(1)

    try:
        # Clean up the presigned URL (sometimes the signature may need re-encoding)
        url_parts = urlsplit(presigned_url)
        query_params = parse_qs(url_parts.query)
        encoded_query = urlencode(query_params, doseq=True)
        cleaned_url = urlunsplit((url_parts.scheme, url_parts.netloc, url_parts.path, encoded_query, url_parts.fragment))

        resp = requests.get(cleaned_url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching datastore from {presigned_url}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate an interactive HTML visualization of PDF rules.')
    parser.add_argument('rules_file', help='Path to the rules file (JSON lines format)')
    parser.add_argument('-o', '--output', help='Output HTML file path', default='interactive_pdf_rules.html')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.rules_file):
        print(f"Error: Rules file not found: {args.rules_file}")
        sys.exit(1)

    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists, attempting to reload it's datastore")
        with open(args.output, "r") as df:
            datastore = get_page_datastore(df.read())

        if datastore is None:
            print(f"Datastore for {args.output} is empty, please run tinyhost and verify your rules and then rerun the script")
            sys.exit(1)

        print(f"Loaded {len(datastore)} entries from datastore, updating {args.rules_file}")

        with open(args.rules_file, 'w') as of:
            for rule in datastore:
                of.write(json.dumps(rule) + "\n")

        return

    pdf_rules = parse_rules_file(args.rules_file)
    html = generate_html(pdf_rules, args.rules_file)
    
    with open(args.output, 'w') as f:
        f.write(html)
    
    print(f"Interactive HTML visualization created: {args.output}")


if __name__ == "__main__":
    main()
