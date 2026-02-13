#!/usr/bin/env python3
"""
Squeezed Light Enhancement Calculator
Computes sensitivity improvement with squeezed states
"""

import numpy as np
import json
import csv
from pathlib import Path
import logging

class SqueezedLightCalculator:
    def __init__(self, cramer_rao_file):
        with open(cramer_rao_file, 'r') as f:
            self.cramer_rao = json.load(f)
        
        self.crb = self.cramer_rao['cramer_rao_bound_Hz']
        self.hl = self.cramer_rao['heisenberg_limit_Hz']
        
    def calculate_enhancement(self, squeezing_levels_db=[0, 10, 20]):
        """Calculate sensitivity enhancement for different squeezing levels"""
        results = []
        
        for db in squeezing_levels_db:
            # Squeezing factor G = 10^(dB/10)
            factor = 10 ** (db / 10)
            sensitivity = self.crb / np.sqrt(factor)
            
            results.append({
                'squeezing_dB': db,
                'squeezing_factor': float(factor),
                'sensitivity_Hz': float(sensitivity),
                'improvement_factor': float(np.sqrt(factor)),
                'distance_to_heisenberg': float(sensitivity / self.hl),
                'notes': self._get_notes(db)
            })
        
        return results
    
    def _get_notes(self, db):
        """Get descriptive notes for squeezing levels"""
        notes = {
            0: "No squeezing (coherent state)",
            10: "Standard lab squeezing",
            20: "State-of-art squeezing"
        }
        return notes.get(db, f"{db} dB squeezing")
    
    def save_results(self, results, output_dir):
        """Save squeezing enhancement results to CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / 'squeezed_light_enhancement.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['squeezing_dB', 'squeezing_factor', 'sensitivity_Hz', 
                           'improvement_factor', 'distance_to_heisenberg', 'notes'])
            
            for r in results:
                writer.writerow([
                    r['squeezing_dB'],
                    f"{r['squeezing_factor']:.0f}",
                    f"{r['sensitivity_Hz']:.2e}",
                    f"{r['improvement_factor']:.2f}",
                    f"{r['distance_to_heisenberg']:.2f}",
                    r['notes']
                ])
        
        return csv_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate squeezed light enhancement')
    parser.add_argument('--cramer_rao', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--squeezing_levels', type=str, default='0,10,20')
    
    args = parser.parse_args()
    
    log_file = Path(args.output_dir).parent.parent / 'logs' / 'phase4' / 'quantum_calculations.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    squeezing_levels = list(map(int, args.squeezing_levels.split(',')))
    
    logging.info("Calculating squeezed light enhancement")
    logging.info(f"Squeezing levels: {squeezing_levels} dB")
    
    calculator = SqueezedLightCalculator(args.cramer_rao)
    results = calculator.calculate_enhancement(squeezing_levels)
    
    csv_file = calculator.save_results(results, args.output_dir)
    logging.info(f"Squeezing enhancement saved to {csv_file}")
    
    for r in results:
        logging.info(f"{r['squeezing_dB']} dB: {r['sensitivity_Hz']:.2e} Hz^{-1/2} "
                    f"({r['improvement_factor']:.1f}× improvement, "
                    f"{r['distance_to_heisenberg']:.2f}× from HL)")
    
    logging.info("Squeezed light calculation complete")

if __name__ == '__main__':
    main()
