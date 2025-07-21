#!/usr/bin/env python3
"""
Create Final Validation Protocol Figure (Simple Version)
Generates text-based flowchart for the validation protocol
"""

def create_text_flowchart():
    """Create a text-based flowchart for the Final Validation Protocol"""
    
    flowchart = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                            FINAL VALIDATION PROTOCOL                            ║
║                      Three-Way Model Comparison Framework                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────┐
│     50 UNSEEN TOPICS        │
│   (No Training Exposure)    │
└─────────────────────────────┘
              │
              ▼
┌─────────────┬─────────────────┬─────────────┐
│   BASELINE  │  DPO-SYNTHETIC  │ DPO-HYBRID  │
│    MODEL    │     MODEL       │    MODEL    │
│             │                 │             │
│ Email Gen.  │   Email Gen.    │ Email Gen.  │
└─────────────┴─────────────────┴─────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL ANALYSIS FRAMEWORK                            │
├─────────────────────────────────┬───────────────────────────────────────────────┤
│     STATISTICAL ANALYSIS        │            EXPERT VALIDATION                 │
│                                 │                                               │
│ • Paired t-tests (all pairs)    │ • Blind evaluation protocol                  │
│ • ANOVA (three-way comparison)  │ • Human professional assessment              │
│ • Effect sizes: d = 0.3-1.0     │ • Correlation analysis: r > 0.80            │
│ • Thresholds: η² > 0.06         │ • Automated-expert agreement                 │
└─────────────────────────────────┴───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┬───────────────────────────────────────────────┐
│        EXPECTED EFFECTS         │            VALIDATION CRITERIA                │
│                                 │                                               │
│ • Baseline vs DPO-Synth:        │ • Effect sizes within predicted ranges       │
│   d = 0.5-0.7 (medium)          │ • Statistical significance (p < 0.05)        │
│ • Baseline vs DPO-Hybrid:       │ • Practical significance (η² > 0.06)         │
│   d = 0.7-1.0 (large)           │ • Expert agreement (r > 0.80)                │
│ • DPO-Synth vs DPO-Hybrid:      │                                               │
│   d = 0.3-0.5 (small-medium)    │                                               │
└─────────────────────────────────┴───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VALIDATION RESULTS                                │
│                                                                                 │
│  PASS: All criteria met | PARTIAL: Some criteria met | FAIL: Criteria not met  │
│                                                                                 │
│           Publication-ready statistical evidence for optimization              │
│                            effectiveness assessment                             │
└─────────────────────────────────────────────────────────────────────────────────┘

METHODOLOGY PHASES:
═══════════════════

INPUT      → 50 unseen validation topics
PROCESSING → Three model variants generate emails  
ANALYSIS   → Statistical framework + Expert validation
VALIDATION → Criteria assessment + Effect size validation
OUTPUT     → Publication-ready evidence + Validation status
"""
    
    return flowchart

def create_latex_tikz_code():
    """Create LaTeX TikZ code for the validation protocol figure"""
    
    tikz_code = r"""
% LaTeX TikZ code for Final Validation Protocol Figure
% Save this as final_validation_protocol.tex and compile with LaTeX

\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning,fit}

\begin{document}
\begin{tikzpicture}[
    node distance=1.5cm,
    box/.style={rectangle, rounded corners, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!10},
    model/.style={rectangle, rounded corners, minimum width=2.5cm, minimum height=1.2cm, text centered, draw=black},
    analysis/.style={rectangle, rounded corners, minimum width=4cm, minimum height=2cm, text centered, draw=black, fill=purple!10},
    arrow/.style={thick,->,>=stealth}
]

% Title
\node at (0,8) {\textbf{\Large Final Validation Protocol}};
\node at (0,7.5) {\textit{Three-Way Model Comparison Framework}};

% Input
\node[box, fill=blue!20] (input) at (0,6) {\textbf{50 Unseen Topics}\\(No training exposure)};

% Models
\node[model, fill=red!20] (baseline) at (-3,4) {\textbf{Baseline}\\Model};
\node[model, fill=green!20] (synthetic) at (0,4) {\textbf{DPO-Synthetic}\\Model};
\node[model, fill=orange!20] (hybrid) at (3,4) {\textbf{DPO-Hybrid}\\Model};

% Analysis sections
\node[analysis] (stats) at (-2,1.5) {\textbf{Statistical Analysis}\\
• Paired t-tests\\
• ANOVA\\
• Effect sizes\\
• η² > 0.06};

\node[analysis, fill=teal!10] (expert) at (2,1.5) {\textbf{Expert Validation}\\
• Blind evaluation\\
• Professional assessment\\
• Correlation r > 0.80\\
• Agreement analysis};

% Output
\node[box, fill=lime!20, minimum width=6cm] (output) at (0,-1) {\textbf{Validation Results}\\
PASS | PARTIAL | FAIL\\
Publication-ready evidence};

% Arrows
\draw[arrow] (input) -- (baseline);
\draw[arrow] (input) -- (synthetic);
\draw[arrow] (input) -- (hybrid);

\draw[arrow] (baseline) -- (stats);
\draw[arrow] (synthetic) -- (stats);
\draw[arrow] (hybrid) -- (stats);

\draw[arrow] (baseline) -- (expert);
\draw[arrow] (synthetic) -- (expert);
\draw[arrow] (hybrid) -- (expert);

\draw[arrow] (stats) -- (output);
\draw[arrow] (expert) -- (output);

\end{tikzpicture}
\end{document}
"""
    
    return tikz_code

def save_figures():
    """Save both text and LaTeX versions of the figure"""
    
    # Save text flowchart
    with open("final_validation_protocol_text.txt", "w") as f:
        f.write(create_text_flowchart())
    
    # Save LaTeX TikZ code
    with open("final_validation_protocol.tex", "w") as f:
        f.write(create_latex_tikz_code())
    
    print("Final Validation Protocol figures created:")
    print("  Text version: final_validation_protocol_text.txt")
    print("  LaTeX TikZ: final_validation_protocol.tex")
    print()
    print("To use the LaTeX version:")
    print("1. Copy the TikZ code into your LaTeX document")
    print("2. Or compile final_validation_protocol.tex directly")
    print("3. Include the resulting PDF in your methodology section")

def main():
    """Create the Final Validation Protocol figures"""
    print("Creating Final Validation Protocol figures...")
    
    # Display the text version
    print("\nFINAL VALIDATION PROTOCOL FLOWCHART:")
    print("="*80)
    print(create_text_flowchart())
    
    # Save files
    save_figures()

if __name__ == "__main__":
    main()