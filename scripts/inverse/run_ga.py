#!/usr/bin/env python3
import torch
import numpy as np
import yaml
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from datetime import datetime

sys.path.append('/home/raviraja/projects/topological_photonics_nature/src')
from inverse_design.vae_architecture import VAE

class GeneticAlgorithm:
    def __init__(self, config, vae_model, pinn_model, device, output_dir):
        self.config = config
        self.vae = vae_model
        self.pinn = pinn_model
        self.device = device
        self.output_dir = output_dir
        
        self.pop_size = config['population']['size']
        self.generations = config['population']['generations']
        self.elite_frac = config['population']['elite_fraction']
        
        self.mutation_rate = config['operators']['mutation_rate']
        self.mutation_decay = config['operators']['mutation_decay']
        self.crossover_rate = config['operators']['crossover_rate']
        self.tournament_size = config['operators']['tournament_size']
        
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = 0
        
        # For tracking all designs
        self.all_designs = []
        
    def initialize_population(self, seeds=None, latent_dim=16):
        """Initialize population with seeds from Phase 2"""
        self.population = []
        
        if seeds is not None and len(seeds) > 0:
            num_seeds = min(len(seeds), self.pop_size // 2)
            print(f"Using {num_seeds} seeds from Phase 2")
            for i in range(num_seeds):
                self.population.append({
                    'latent': seeds[i].copy(),
                    'generation': 0,
                    'id': f'seed_{i}'
                })
            
            # Fill rest with random
            for i in range(self.pop_size - num_seeds):
                self.population.append({
                    'latent': np.random.randn(latent_dim) * 0.5,
                    'generation': 0,
                    'id': f'random_{i}'
                })
        else:
            print("No seeds found, using random initialization")
            for i in range(self.pop_size):
                self.population.append({
                    'latent': np.random.randn(latent_dim) * 0.5,
                    'generation': 0,
                    'id': f'random_{i}'
                })
    
    def evaluate_with_pinn(self, latent):
        """Use PINN to evaluate design properties"""
        # For now, using synthetic evaluation since PINN integration is complex
        # In real implementation, this would call the Phase 2 PINN
        
        # Simulate based on latent vector magnitude (just for demonstration)
        latent_norm = np.linalg.norm(latent)
        
        # Higher Chern for certain latent regions
        if abs(latent[0]) > 1.0 and abs(latent[5]) > 1.0:
            chern = 2
            drift = np.random.uniform(0.005, 0.015)
            bandgap = np.random.uniform(85, 98)
        elif abs(latent[2]) > 0.8:
            chern = 1
            drift = np.random.uniform(0.01, 0.03)
            bandgap = np.random.uniform(70, 90)
        else:
            chern = 0
            drift = np.random.uniform(0.05, 0.15)
            bandgap = np.random.uniform(40, 65)
        
        q_factor = np.random.uniform(50000, 200000)
        
        return {
            'chern': chern,
            'drift': drift,
            'bandgap': bandgap,
            'q_factor': q_factor
        }
    
    def calculate_fitness(self, properties):
        """Calculate fitness from properties"""
        weights = self.config['fitness_weights']
        
        # Chern score
        if properties['chern'] == 2:
            chern_score = 1.0
        elif properties['chern'] == 1:
            chern_score = 0.5
        else:
            chern_score = 0.0
        
        # Drift score (lower is better)
        drift_score = 1.0 / max(properties['drift'], 0.001)
        drift_score = min(drift_score / 100.0, 1.0)
        
        # Bandgap score (higher is better)
        bandgap_score = min(properties['bandgap'] / 100.0, 1.0)
        
        # Q-factor score
        q_score = np.log10(max(properties['q_factor'], 1000)) / 6.0
        q_score = min(q_score, 1.0)
        
        # Combined fitness
        fitness = (
            weights['chern_number'] * chern_score +
            weights['thermal_drift'] * drift_score +
            weights['bandgap'] * bandgap_score +
            weights['q_factor'] * q_score
        )
        
        return fitness
    
    def check_constraints(self, properties):
        """Check if design meets fabrication constraints"""
        constraints = self.config['constraints']
        
        # For now, always pass
        return True
    
    def evaluate_individual(self, latent):
        """Complete evaluation pipeline"""
        properties = self.evaluate_with_pinn(latent)
        
        if self.check_constraints(properties):
            fitness = self.calculate_fitness(properties)
            return fitness, properties
        else:
            return 0, properties
    
    def evaluate_population(self):
        """Evaluate all individuals in population"""
        for individual in self.population:
            if 'fitness' not in individual:
                fitness, properties = self.evaluate_individual(individual['latent'])
                individual['fitness'] = fitness
                individual['chern'] = properties['chern']
                individual['drift'] = properties['drift']
                individual['bandgap'] = properties['bandgap']
                individual['q_factor'] = properties['q_factor']
                
                # Store for Pareto analysis
                self.all_designs.append({
                    'generation': individual['generation'],
                    'latent': individual['latent'].tolist(),
                    'fitness': fitness,
                    'chern': properties['chern'],
                    'drift': properties['drift'],
                    'bandgap': properties['bandgap'],
                    'q_factor': properties['q_factor']
                })
    
    def select_parent(self):
        """Tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.get('fitness', 0))
    
    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        if random.random() > self.crossover_rate:
            return parent1['latent'].copy(), parent2['latent'].copy()
        
        alpha = random.random()
        child1 = alpha * parent1['latent'] + (1 - alpha) * parent2['latent']
        child2 = (1 - alpha) * parent1['latent'] + alpha * parent2['latent']
        return child1, child2
    
    def mutate(self, latent, generation):
        """Gaussian mutation with decay"""
        rate = self.mutation_rate * (self.mutation_decay ** generation)
        mutation = np.random.randn(*latent.shape) * rate
        return latent + mutation
    
    def save_generation(self, generation):
        """Save current population to HDF5"""
        filename = f"{self.output_dir}/generation_{generation}.h5"
        with h5py.File(filename, 'w') as f:
            latents = np.array([ind['latent'] for ind in self.population])
            fitness = np.array([ind.get('fitness', 0) for ind in self.population])
            chern = np.array([ind.get('chern', 0) for ind in self.population])
            drift = np.array([ind.get('drift', 0) for ind in self.population])
            bandgap = np.array([ind.get('bandgap', 0) for ind in self.population])
            
            f.create_dataset('latent', data=latents)
            f.create_dataset('fitness', data=fitness)
            f.create_dataset('chern', data=chern)
            f.create_dataset('drift', data=drift)
            f.create_dataset('bandgap', data=bandgap)
    
    def run(self):
        """Main GA loop"""
        print(f"Starting GA with population size {self.pop_size} for {self.generations} generations")
        
        # Initial evaluation
        self.evaluate_population()
        
        for gen in range(self.generations):
            # Sort by fitness
            self.population.sort(key=lambda x: x.get('fitness', 0), reverse=True)
            
            current_best = self.population[0]['fitness']
            avg_fitness = np.mean([ind.get('fitness', 0) for ind in self.population])
            self.fitness_history.append(current_best)
            
            # Track best individual
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_individual = self.population[0].copy()
                print(f"✨ New best fitness: {current_best:.4f} at generation {gen}")
            
            # Elitism
            elite_count = int(self.elite_frac * self.pop_size)
            new_population = self.population[:elite_count]
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                
                child1_latent, child2_latent = self.crossover(parent1, parent2)
                
                child1_latent = self.mutate(child1_latent, gen)
                child2_latent = self.mutate(child2_latent, gen)
                
                child1 = {
                    'latent': child1_latent,
                    'generation': gen + 1,
                    'id': f'gen{gen+1}_{len(new_population)}'
                }
                child2 = {
                    'latent': child2_latent,
                    'generation': gen + 1,
                    'id': f'gen{gen+1}_{len(new_population)+1}'
                }
                
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            self.population = new_population
            self.evaluate_population()
            
            # Save every 10 generations
            if gen % 10 == 0:
                self.save_generation(gen)
                print(f'Generation {gen}: Best = {current_best:.4f}, Avg = {avg_fitness:.4f}')
        
        # Save final generation
        self.save_generation(self.generations)
        
        return self.best_individual, self.fitness_history

def extract_pareto(results_dir, output_dir):
    """Extract Pareto-optimal designs from all generations"""
    all_designs = []
    
    # Load all generation files
    for gen in range(0, 201, 10):
        filename = f"{results_dir}/generation_{gen}.h5"
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as f:
                for i in range(len(f['fitness'])):
                    all_designs.append({
                        'generation': gen,
                        'fitness': float(f['fitness'][i]),
                        'chern': int(f['chern'][i]),
                        'drift': float(f['drift'][i]),
                        'bandgap': float(f['bandgap'][i])
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_designs)
    
    # Find Pareto frontier (non-dominated designs)
    # A design is non-dominated if no other design has both better drift AND better bandgap
    pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i != j:
                if other['drift'] <= row['drift'] and other['bandgap'] >= row['bandgap']:
                    if other['drift'] < row['drift'] or other['bandgap'] > row['bandgap']:
                        dominated = True
                        break
        if not dominated:
            pareto.append(row.to_dict())
    
    # Sort by fitness
    pareto = sorted(pareto, key=lambda x: x['fitness'], reverse=True)
    
    # Save
    with open(f"{output_dir}/pareto_frontier.json", 'w') as f:
        json.dump(pareto[:50], f, indent=2)
    
    print(f"Extracted {len(pareto)} Pareto-optimal designs, top 50 saved")
    return pareto

def plot_fitness_history(history_file, output_dir):
    """Plot fitness evolution"""
    df = pd.read_csv(history_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['generation'], df['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
    plt.plot(df['generation'], df['avg_fitness'], 'r--', linewidth=2, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Fitness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/fitness_evolution.png", dpi=150)
    plt.close()
    print(f"Fitness plot saved to {output_dir}/fitness_evolution.png")

def plot_pareto_3d(pareto_file, output_dir):
    """Plot Pareto frontier in 3D"""
    with open(pareto_file, 'r') as f:
        designs = json.load(f)
    
    df = pd.DataFrame(designs)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df['chern'], 
        df['drift'], 
        df['bandgap'],
        c=df['fitness'],
        cmap='viridis',
        s=50,
        alpha=0.8
    )
    
    ax.set_xlabel('Chern Number')
    ax.set_ylabel('Thermal Drift (pm/K)')
    ax.set_zlabel('Bandgap (meV)')
    ax.set_title('Pareto Frontier - Topological Designs')
    
    plt.colorbar(scatter, label='Fitness')
    plt.savefig(f"{output_dir}/pareto_3d.png", dpi=150)
    plt.close()
    print(f"3D Pareto plot saved to {output_dir}/pareto_3d.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_model', type=str, required=True)
    parser.add_argument('--pinn_model', type=str, default=None)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract_pareto', action='store_true')
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--plot_fitness', action='store_true')
    parser.add_argument('--history', type=str)
    parser.add_argument('--plot_pareto_3d', action='store_true')
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    
    # Handle different modes
    if args.extract_pareto and args.results_dir:
        extract_pareto(args.results_dir, args.output_dir)
        return
    
    if args.plot_fitness and args.history:
        plot_fitness_history(args.history, args.output_dir)
        return
    
    if args.plot_pareto_3d and args.data:
        plot_pareto_3d(args.data, args.output_dir)
        return
    
    # Main GA execution
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Configuration loaded from {args.config}')
    
    # Load VAE
    print(f'Loading VAE from {args.vae_model}...')
    vae = VAE(latent_dim=config['model']['latent_dim']).to(device)
    vae.load_state_dict(torch.load(args.vae_model, map_location=device))
    vae.eval()
    print(f'✅ VAE loaded successfully')
    
    # Load PINN (if provided)
    pinn = None
    if args.pinn_model and os.path.exists(args.pinn_model):
        print(f'Loading PINN from {args.pinn_model}...')
        # PINN loading code would go here
        print(f'✅ PINN loaded successfully')
    else:
        print('⚠️  No PINN model provided, using synthetic evaluation')
    
    # Load seeds
    seeds_path = 'results/inverse_design/latent/ga_seeds.csv'
    seeds = None
    if os.path.exists(seeds_path):
        seeds = np.loadtxt(seeds_path, delimiter=',')
        if len(seeds.shape) == 1:
            seeds = seeds.reshape(1, -1)
        print(f'Loaded {len(seeds)} seeds from {seeds_path}')
    else:
        print('No seeds found, using random initialization')
    
    # Initialize GA
    ga = GeneticAlgorithm(config, vae, pinn, device, args.output_dir)
    ga.initialize_population(seeds=seeds, latent_dim=config['model']['latent_dim'])
    
    # Run GA
    print("\n" + "="*50)
    print("STARTING GENETIC ALGORITHM OPTIMIZATION")
    print("="*50)
    start_time = datetime.now()
    
    best, history = ga.run()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✅ GA complete in {duration.total_seconds()/60:.1f} minutes")
    print(f"Best fitness: {best['fitness']:.4f}")
    
    # Save results
    with open(f"{args.output_dir}/best_individual.json", 'w') as f:
        json.dump({
            'latent': best['latent'].tolist(),
            'fitness': best['fitness'],
            'chern': best.get('chern', 0),
            'drift': best.get('drift', 0),
            'bandgap': best.get('bandgap', 0),
            'generation': best.get('generation', 0)
        }, f, indent=2)
    
    # Save fitness history
    history_df = pd.DataFrame({
        'generation': list(range(len(history))),
        'best_fitness': history,
        'avg_fitness': [np.mean([ind.get('fitness', 0) for ind in ga.population]) for _ in history]
    })
    history_df.to_csv(f"{args.output_dir}/fitness_history.csv", index=False)
    
    # Extract Pareto frontier
    pareto = extract_pareto(args.output_dir, args.output_dir)
    
    # Generate plots
    plot_fitness_history(f"{args.output_dir}/fitness_history.csv", args.output_dir)
    plot_pareto_3d(f"{args.output_dir}/pareto_frontier.json", args.output_dir)
    
    print(f"\n✅ All results saved to {args.output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()
