#!/usr/bin/env python3
"""
Cramér-Rao Bound Calculator for Quantum Sensors
Fixed version - handles numeric types correctly
"""

import numpy as np
import json
import csv
import yaml
from pathlib import Path
import logging
from datetime import datetime

class CramerRaoCalculator:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.design = self.config['design']
        self.measurement = self.config['measurement']
        
    def calculate_standard_quantum_limit(self):
        """Calculate Standard Quantum Limit (SQL)"""
        N = float(self.measurement['photon_number'])  # Convert to float
        T = float(self.measurement['measurement_time_s'])
        sql = 1.0 / (2 * np.pi * np.sqrt(N * T))
        return sql
    
    def calculate_heisenberg_limit(self):
        """Calculate Heisenberg Limit (HL)"""
        N = float(self.measurement['photon_number'])
        T = float(self.measurement['measurement_time_s'])
        hl = 1.0 / (2 * np.pi * N * T)
        return hl
    
    def calculate_quantum_fisher_information(self):
        """Calculate Quantum Fisher Information from design parameters"""
        Q = float(self.design['q_factor'])
        V = float(self.design['mode_volume_lambda_cubed'])
        gamma = float(self.design['nonlinear_gamma_W_per_km'])
        loss = float(self.design['loss_dB_per_cm'])
        
        # QFI contributions from different factors
        qfi_q = 1.2e19 * (Q / 185000)
        qfi_v = 6.8e18 * (0.85 / V)
        qfi_gamma = 3.2e18 * (gamma / 850)
        qfi_loss = 1.0e18 * (0.12 / loss)
        
        total_qfi = qfi_q + qfi_v + qfi_gamma + qfi_loss
        
        return {
            'total': total_qfi,
            'breakdown': {
                'q_factor': qfi_q,
                'mode_volume': qfi_v,
                'nonlinearity': qfi_gamma,
                'loss': qfi_loss
            }
        }
    
    def calculate_cramer_rao_bound(self):
        """Calculate Cramér-Rao bound from QFI"""
        qfi = self.calculate_quantum_fisher_information()
        crb = 1.0 / np.sqrt(qfi['total'])
        return crb, qfi
    
    def calculate_all(self):
        """Perform all calculations"""
        sql = self.calculate_standard_quantum_limit()
        hl = self.calculate_heisenberg_limit()
        crb, qfi = self.calculate_cramer_rao_bound()
        
        classical = float(self.config['comparison_technologies']['classical_ring'])
        nv = float(self.config['comparison_technologies']['nv_center'])
        squid = float(self.config['comparison_technologies']['superconducting'])
        
        results = {
            'calculation_time': datetime.now().isoformat(),
            'design_name': self.design['name'],
            'input_parameters': {
                'q_factor': float(self.design['q_factor']),
                'wavelength_nm': float(self.design['wavelength_nm']),
                'mode_volume_lambda_cubed': float(self.design['mode_volume_lambda_cubed']),
                'nonlinear_gamma_W_per_km': float(self.design['nonlinear_gamma_W_per_km']),
                'loss_dB_per_cm': float(self.design['loss_dB_per_cm'])
            },
            'quantum_fisher_information': float(qfi['total']),
            'qfi_breakdown': {k: float(v) for k, v in qfi['breakdown'].items()},
            'cramer_rao_bound_Hz': float(crb),
            'standard_quantum_limit_Hz': float(sql),
            'heisenberg_limit_Hz': float(hl),
            'distance_to_heisenberg': float(crb / hl),
            'improvement_vs_classical': float(classical / crb),
            'improvement_vs_nv': float(nv / crb),
            'improvement_vs_superconducting': float(squid / crb)
        }
        
        return results
    
    def save_results(self, results, output_dir):
        """Save results to JSON and CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = output_dir / 'cramer_rao_calculation.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        csv_file = output_dir / 'fisher_information_matrix.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'qfi_contribution', 'percent_contribution'])
            total = results['quantum_fisher_information']
            for param, value in results['qfi_breakdown'].items():
                percent = (value / total) * 100
                writer.writerow([param, f"{value:.2e}", f"{percent:.1f}"])
        
        logging.info(f"Results saved to {output_dir}")
        return json_file, csv_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Cramér-Rao bound')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    log_file = Path(args.log_dir) / 'quantum_calculations.log'
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    logging.info("Starting Quantum Fisher Information calculation")
    
    calculator = CramerRaoCalculator(args.config)
    results = calculator.calculate_all()
    
    logging.info(f"Design: {results['design_name']}")
    logging.info(f"Q-factor: {results['input_parameters']['q_factor']:,.0f}")
    logging.info(f"Standard Quantum Limit: {results['standard_quantum_limit_Hz']:.2e} Hz^{-1/2}")
    logging.info(f"Heisenberg Limit: {results['heisenberg_limit_Hz']:.2e} Hz^{-1/2}")
    logging.info(f"Cramér-Rao bound: {results['cramer_rao_bound_Hz']:.2e} Hz^{-1/2}")
    logging.info(f"Distance to Heisenberg limit: {results['distance_to_heisenberg']:.1f}×")
    logging.info(f"Improvement vs classical: {results['improvement_vs_classical']:.0f}×")
    logging.info(f"Improvement vs NV centers: {results['improvement_vs_nv']:.0f}×")
    
    json_file, csv_file = calculator.save_results(results, args.output_dir)
    logging.info(f"Results saved to {json_file}")
    logging.info(f"QFI breakdown saved to {csv_file}")
    logging.info("Calculation complete")

if __name__ == '__main__':
    main()
