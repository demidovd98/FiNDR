import json
import os
from pathlib import Path
from typing import List, Dict, Optional


# unused currently
def load_guessed_classnames(
    dataset_name: str,
    llm_name: str = 'qwen',
    data_dir: str = './data'
) -> Dict:
    """
    Load guessed class names from JSON file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'bird200', 'dog120')
        llm_name: Name of the LLM used (e.g., 'qwen', 'gemini')
        data_dir: Path to the data directory
    
    Returns:
        Dictionary with guessed class names
    """
    path = f"{data_dir}/guessed_classnames/{dataset_name}/{dataset_name}s_{llm_name}.json"
    with open(path, 'r') as f:
        return json.load(f)


def generate_html_visualization(
    gt_names: List[str],
    guessed_names: List[str],
    output_path: str = './class_names_visualization.html',
    title: str = 'Class Names Visualization',
    candidate_names: Optional[List[str]] = None,
    removed_names: Optional[List[str]] = None
) -> str:
    """
    Generate a nice HTML visualization for ground truth and guessed class names.
    
    Args:
        gt_names: List of ground truth class names
        guessed_names: List of guessed class names
        output_path: Path where to save the HTML file
        title: Title for the visualization
        candidate_names: Optional list of candidate class names after filtering
        removed_names: Optional list of removed class names after filtering
    
    Returns:
        Path to the generated HTML file
    """
    
    # Find common names (case-insensitive)
    gt_set = set(name.lower().strip() for name in gt_names)
    guessed_set = set(name.lower().strip() for name in guessed_names)
    
    common_names = sorted(gt_set.intersection(guessed_set))
    only_in_gt = sorted(gt_set - guessed_set)
    only_in_guessed = sorted(guessed_set - gt_set)
    
    # Create a mapping for display
    gt_names_lower = [name.lower().strip() for name in gt_names]
    guessed_names_lower = [name.lower().strip() for name in guessed_names]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .stat-box {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px 25px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            display: block;
            margin-top: 5px;
        }}
        
        .content {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .section {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .section summary {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            cursor: pointer;
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            list-style: none;
        }}
        
        .section summary:hover {{
            background: #e9ecef;
        }}
        
        .section summary::-webkit-details-marker {{
            display: none;
        }}
        
        .section summary::before {{
            content: "▶";
            margin-right: 10px;
            transition: transform 0.3s;
        }}
        
        .section[open] summary::before {{
            transform: rotate(90deg);
        }}
        
        .section-content {{
            padding: 30px;
        }}
        
        .name-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .name-tag {{
            display: inline-block;
            background: #f0f0f0;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
            word-break: break-word;
            max-width: 100%;
        }}
        
        .name-tag:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            background: #e8eef7;
        }}
        
        .name-tag.in-gt {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }}
        
        .name-tag.in-gt:hover {{
            background: #c3e6cb;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        }}
        
        .name-tag.not-in-gt {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}
        
        .name-tag.not-in-gt:hover {{
            background: #f5c6cb;
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3);
        }}
        
        .analysis {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }}
        
        .analysis-box {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .analysis-box h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #764ba2;
            padding-bottom: 10px;
            position: sticky;
            top: 0;
            background: white;
            z-index: 10;
        }}
        
        .analysis-box ul {{
            list-style: none;
            padding: 0;
        }}
        
        .analysis-box li {{
            padding: 8px 0;
            padding-left: 20px;
            position: relative;
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }}
        
        .analysis-box li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #667eea;
            font-weight: bold;
        }}
        
        .common {{
            border-left: 4px solid #4CAF50;
        }}
        
        .only-gt {{
            border-left: 4px solid #2196F3;
        }}
        
        .only-guessed {{
            border-left: 4px solid #FF9800;
        }}
        
        .more-items {{
            padding: 8px 0 !important;
            padding-left: 20px !important;
            color: #999;
            font-style: italic;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            font-size: 0.9em;
        }}
        
        .scrollable-hint {{
            font-size: 0.8em;
            color: #999;
            margin-top: 10px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="stats">
                <div class="stat-box">
                    <div>Ground Truth</div>
                    <span class="stat-value">{len(gt_names)}</span>
                </div>
                <div class="stat-box">
                    <div>Guessed</div>
                    <span class="stat-value">{len(guessed_names)}</span>
                </div>
                <div class="stat-box">
                    <div>Match</div>
                    <span class="stat-value">{len(common_names)}</span>
                </div>
                <div class="stat-box">
                    <div>Match %</div>
                    <span class="stat-value">{(len(common_names) / len(gt_names) * 100):.1f}%</span>
                </div>
                {f'''
                <div class="stat-box">
                    <div>Candidates</div>
                    <span class="stat-value">{len(candidate_names)}</span>
                </div>
                ''' if candidate_names else ''}
                {f'''
                <div class="stat-box">
                    <div>Removed</div>
                    <span class="stat-value">{len(removed_names)}</span>
                </div>
                ''' if removed_names else ''}
            </div>
        </div>
        
        <div class="content">
            <details class="section">
                <summary>Ground Truth Class Names</summary>
                <div class="section-content">
                    <div class="name-list">
                        {"".join(f'<span class="name-tag">{name}</span>' for name in sorted(gt_names))}
                    </div>
                </div>
            </details>
            
            <details class="section">
                <summary>Guessed Class Names</summary>
                <div class="section-content">
                    <div class="name-list">
                        {"".join(f'<span class="name-tag {("in-gt" if name.lower().strip() in gt_set else "not-in-gt")}">{name}</span>' for name in sorted(guessed_names))}
                    </div>
                </div>
            </details>
            
            {f'''
            <details class="section">
                <summary>Removed Class Names</summary>
                <div class="section-content">
                    <div class="name-list">
                        {"".join(f'<span class="name-tag {("in-gt" if name.lower().strip() in gt_set else "not-in-gt")}">{name}</span>' for name in sorted(removed_names))}
                    </div>
                </div>
            </details>
            ''' if removed_names else ''}
            
            {f'''
            <details class="section">
                <summary>Final Kept Class Names</summary>
                <div class="section-content">
                    <div class="name-list">
                        {"".join(f'<span class="name-tag {("in-gt" if name.lower().strip() in gt_set else "not-in-gt")}">{name}</span>' for name in sorted(candidate_names))}
                    </div>
                </div>
            </details>
            ''' if candidate_names else ''}
        </div>
        
        <div class="footer">
            <p>Generated visualization of ground truth vs. guessed class names</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Visualization saved to: {output_path}")
    return output_path
