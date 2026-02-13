#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_quick_summary(masterpieces_dir, output_file):
    """Create a quick summary PDF"""
    
    with PdfPages(output_file) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        plt.text(0.5, 0.95, 'Topological Masterpiece Discoveries', 
                fontsize=16, ha='center', weight='bold')
        
        # Find all JSON files
        json_files = [f for f in os.listdir(masterpieces_dir) 
                     if f.endswith('.json') and 'design' in f]
        json_files.sort()
        
        y_pos = 0.85
        for i, json_file in enumerate(json_files[:5]):
            with open(f'{masterpieces_dir}/{json_file}', 'r') as f:
                data = json.load(f)
            
            # Properties are at top level
            text = f"Design {i+1}: {data['design_id']}\n"
            text += f"  Chern: {data['chern']}  "
            text += f"Drift: {data['drift']:.4f} pm/K  "
            text += f"Bandgap: {data['bandgap']:.1f} meV\n"
            
            plt.text(0.1, y_pos, text, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            y_pos -= 0.12
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Individual design pages
        for i, json_file in enumerate(json_files[:5]):
            with open(f'{masterpieces_dir}/{json_file}', 'r') as f:
                data = json.load(f)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Geometry image
            img_file = json_file.replace('.json', '.png')
            img_path = f'{masterpieces_dir}/{img_file}'
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax1.imshow(img, cmap='gray')
                ax1.set_title(f'Design {i+1}: {data["design_id"]} Unit Cell')
                ax1.axis('off')
            
            # Bandstructure
            band_file = json_file.replace('.json', '_bandstructure.png')
            band_path = f'{masterpieces_dir}/{band_file}'
            if os.path.exists(band_path):
                band_img = plt.imread(band_path)
                ax2.imshow(band_img)
                ax2.axis('off')
                ax2.set_title('Band Structure')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Metrics summary page
        fig, ax = plt.subplots(figsize=(10, 6))
        
        designs = []
        for json_file in json_files[:5]:
            with open(f'{masterpieces_dir}/{json_file}', 'r') as f:
                designs.append(json.load(f))
        
        x = range(len(designs))
        chern = [d['chern'] for d in designs]
        drift = [d['drift'] * 100 for d in designs]
        bandgap = [d['bandgap'] / 10 for d in designs]
        
        ax.bar([i-0.2 for i in x], chern, width=0.2, label='Chern', color='green')
        ax.bar(x, drift, width=0.2, label='Drift (x100 pm/K)', color='blue')
        ax.bar([i+0.2 for i in x], bandgap, width=0.2, label='Bandgap/10 (meV)', color='red')
        
        ax.set_xlabel('Design')
        ax.set_ylabel('Scaled Value')
        ax.set_title('Design Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'D{i+1}' for i in x])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        pdf.savefig()
        plt.close()
    
    print(f"Summary PDF saved to {output_file}")

if __name__ == '__main__':
    masterpieces_dir = 'results/inverse_design/topological_masterpieces/'
    output_file = 'results/inverse_design/discovery_summary.pdf'
    create_quick_summary(masterpieces_dir, output_file)
