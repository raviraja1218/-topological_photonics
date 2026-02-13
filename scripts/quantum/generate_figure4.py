#!/usr/bin/env python3
"""
Figure 4 Generator - Quantum Limit Visualization
Fixed version with correct log directory path
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os

def setup_matplotlib_style():
    """Set matplotlib style for Nature publications"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7.2, 5.4)
    plt.rcParams['axes.unicode_minus'] = False

def create_panel_a(ax, sensitivity_df):
    """Panel A: Sensitivity vs Measurement Time"""
    # Find closest value to 1.0 for annotations
    closest_idx = (sensitivity_df['time_s'] - 1.0).abs().idxmin()
    t1_data = sensitivity_df.loc[closest_idx]
    
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['SQL_Hz'], 
              'k--', linewidth=1.5, label='Standard Quantum Limit')
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['HL_Hz'], 
              'k:', linewidth=1.5, label='Heisenberg Limit')
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['classical_Hz'], 
              'r-', linewidth=1.5, label='Classical sensor')
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['NV_Hz'], 
              'b-', linewidth=1.5, label='NV center')
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['superconducting_Hz'], 
              'g-', linewidth=1.5, label='Superconducting')
    ax.loglog(sensitivity_df['time_s'], sensitivity_df['your_design_Hz'], 
              'purple', linewidth=2.5, label='Crescent-Moire v1')
    
    # Mark T=1s point
    ax.plot(1.0, t1_data['your_design_Hz'], 'p', color='purple', 
            markersize=8, markeredgecolor='black', markeredgewidth=0.5)
    
    # Add arrow annotation
    ax.annotate(f"{t1_data['your_design_Hz']/t1_data['HL_Hz']:.1f}× from HL", 
                xy=(1.0, t1_data['your_design_Hz']), 
                xytext=(3, t1_data['your_design_Hz']*2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=7, ha='left')
    
    ax.set_xlabel('Measurement Time (s)')
    ax.set_ylabel('Sensitivity (Hz$^{-1/2}$)')
    ax.set_xlim([1e-6, 1e4])
    ax.set_ylim([1e-14, 1e0])
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')
    ax.set_title('a', loc='left', fontweight='bold', fontsize=10)

def create_panel_b(ax, cramer_rao):
    """Panel B: Comparison to State-of-Art"""
    technologies = ['Classical', 'NV Center', 'Superconducting', 'Crescent-Moire']
    values = [
        2.1e-3,
        3.5e-7,
        1.2e-8,
        cramer_rao['cramer_rao_bound_Hz']
    ]
    colors = ['red', 'blue', 'green', 'purple']
    
    bars = ax.bar(technologies, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Sensitivity at 1s (Hz$^{-1/2}$)')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*1.1,
                f'{val:.1e}'.replace('e-0', 'e-'), 
                ha='center', va='bottom', fontsize=6)
    
    # Add improvement annotations
    imp = cramer_rao['improvement_vs_classical']
    ax.annotate(f'{imp:.0f}×', xy=(0.5, values[1]), xytext=(1.5, values[1]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1),
                fontsize=7, ha='center')
    
    ax.set_title('b', loc='left', fontweight='bold', fontsize=10)

def create_panel_c(ax, squeezing_file):
    """Panel C: Squeezed Light Enhancement"""
    squeezing_df = pd.read_csv(squeezing_file)
    
    db_levels = squeezing_df['squeezing_dB'].values
    sensitivities = squeezing_df['sensitivity_Hz'].values
    
    bars = ax.bar([f'{int(db)} dB' for db in db_levels], sensitivities,
                  color=['lightgray', 'silver', 'darkgray'],
                  edgecolor='black', linewidth=0.5)
    
    # Add Heisenberg limit line
    hl_line = 1.6e-12
    ax.axhline(y=hl_line, color='k', linestyle=':', linewidth=1.5, label='Heisenberg Limit')
    
    ax.set_yscale('log')
    ax.set_ylabel('Sensitivity (Hz$^{-1/2}$)')
    ax.set_ylim([1e-13, 1e-8])
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*1.5,
                f'{sens:.1e}'.replace('e-0', 'e-'), 
                ha='center', va='bottom', fontsize=6)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax.set_title('c', loc='left', fontweight='bold', fontsize=10)

def create_panel_d(ax):
    """Panel D: Applications Roadmap"""
    years = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
    
    # Define phases
    ax.axvspan(2026, 2028, alpha=0.2, color='lightblue', label='Laboratory')
    ax.axvspan(2028, 2031, alpha=0.2, color='lightgreen', label='Prototype')
    ax.axvspan(2031, 2035, alpha=0.2, color='lightcoral', label='Commercial')
    
    # Add milestones
    milestones = [
        (2026.5, 'Fabricate Crescent-Moire'),
        (2027.5, 'Validate Q-factor'),
        (2029.0, 'Gravitational wave demo'),
        (2030.5, 'Dark matter sensor'),
        (2032.0, 'Medical imaging'),
        (2034.0, 'Navigation systems')
    ]
    
    for year, text in milestones:
        ax.plot(year, 1, 'ko', markersize=4)
        ax.text(year, 1.2, text, rotation=45, ha='right', va='bottom', fontsize=6)
    
    ax.set_xlim([2026, 2035])
    ax.set_ylim([0, 2])
    ax.set_xlabel('Year')
    ax.set_yticks([])
    ax.set_title('d', loc='left', fontweight='bold', fontsize=10)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=6)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Figure 4')
    parser.add_argument('--sensitivity_data', type=str, required=True)
    parser.add_argument('--squeezing_data', type=str, required=True)
    parser.add_argument('--cramer_rao', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--format', type=str, default='png,eps')
    
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    log_dir = Path('logs/phase4')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'figure_generation.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Generating Figure 4: Quantum Limit Visualization")
    
    # Load data
    sensitivity_df = pd.read_csv(args.sensitivity_data)
    with open(args.cramer_rao, 'r') as f:
        cramer_rao = json.load(f)
    
    # Setup plot
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    
    # Create panels
    create_panel_a(axes[0, 0], sensitivity_df)
    create_panel_b(axes[0, 1], cramer_rao)
    create_panel_c(axes[1, 0], args.squeezing_data)
    create_panel_d(axes[1, 1])
    
    plt.tight_layout()
    
    # Save in requested formats
    formats = args.format.split(',')
    for fmt in formats:
        output_file = f"{args.output}.{fmt}"
        plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        logging.info(f"Saved {output_file}")
    
    logging.info("Figure 4 generation complete")

if __name__ == '__main__':
    main()
