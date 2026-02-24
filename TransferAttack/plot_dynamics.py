import os
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='Plot Rebuttal Results for MDCS Dynamics')
    parser.add_argument('--results_dir', default='./results', type=str, help='Directory containing result subfolders from rebuttal experiments')
    parser.add_argument('--output_dir', default='./figs', type=str, help='Directory to save generated figures')
    return parser.parse_args()

def load_results(results_dir):
    aggregated_data = {}
    
    # Walk through results directory to find all results_dump.json
    for root, dirs, files in os.walk(results_dir):
        if 'results_dump.json' in files:
            json_path = os.path.join(root, 'results_dump.json')
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # data structure: {attack_name: {...}}
                    for attack_name, metrics in data.items():
                        aggregated_data[attack_name] = metrics
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                
    return aggregated_data

def plot_metric(data, metric_key, ylabel, title, save_path, is_bb=False):
    plt.figure(figsize=(10, 6))
    
    # Define styles for different method groups
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (attack_name, metrics) in enumerate(data.items()):
        # X-axis: T
        T = metrics.get('T', [])
        if not T:
            continue
            
        y_values = []
        
        if is_bb:
            # For Black-box, average across all BB models
            bb_dict = metrics.get(metric_key, {})
            if not bb_dict:
                continue
            
            # Combine all bb models
            bb_matrix = []
            for model_name, values in bb_dict.items():
                if values:
                    bb_matrix.append(values)
            
            if bb_matrix:
                y_values = np.mean(bb_matrix, axis=0)
            else:
                continue
        else:
            y_values = metrics.get(metric_key, [])
            
        if len(y_values) == 0:
            continue
            
        # Plotting
        plt.plot(T, y_values, label=attack_name, marker=markers[i % len(markers)], 
                 linestyle=linestyles[i % len(linestyles)], linewidth=2, markersize=6)

    plt.xlabel('Iterations (T)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()

def main():
    args = get_parser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Loading results from {args.results_dir}...")
    data = load_results(args.results_dir)
    
    if not data:
        print("No results found! Make sure you run the experiments first.")
        return
        
    print(f"Found data for attacks: {list(data.keys())}")
    
    # 1. White-box ASR
    plot_metric(data, 'wb_asr', 'Attack Success Rate (%)', 'White-box ASR vs Iterations', 
                os.path.join(args.output_dir, 'wb_asr.png'))
                
    # 2. White-box Loss
    plot_metric(data, 'wb_loss', 'Cross Entropy Loss', 'White-box Loss vs Iterations', 
                os.path.join(args.output_dir, 'wb_loss.png'))
                
    # 3. Black-box ASR (Avg)
    plot_metric(data, 'bb_asr', 'Avg Attack Success Rate (%)', 'Black-box ASR (Avg) vs Iterations', 
                os.path.join(args.output_dir, 'bb_asr_avg.png'), is_bb=True)
                
    # 4. Black-box Loss (Avg)
    plot_metric(data, 'bb_loss', 'Avg Cross Entropy Loss', 'Black-box Loss (Avg) vs Iterations', 
                os.path.join(args.output_dir, 'bb_loss_avg.png'), is_bb=True)
                
    # 5. dt Dynamics
    plot_metric(data, 'dt', 'Averaged $d_t$', 'Averaged $d_t$ vs Iterations', 
                os.path.join(args.output_dir, 'dt_dynamics.png'))
                
    # 6. Momentum Dynamics
    plot_metric(data, 'momentum', 'Averaged $|m_t|$', 'Averaged Momentum $|m_t|$ vs Iterations', 
                os.path.join(args.output_dir, 'momentum_dynamics.png'))

if __name__ == '__main__':
    main()
