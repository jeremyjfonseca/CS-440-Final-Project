import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# Add the project root to sys.path so imports work from here
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from project modules
from data_loader.loader import load_data, get_split
from perceptron.model import Perceptron
from nn_scratch.nn import NeuralNetScratch
from nn_pytorch.model import PyTorchNN

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_experiment(algo_name, X, y, pct, test_pct, runs, hidden_dim1=100, hidden_dim2=50, 
                  lr=0.01, epochs=10, batch_size=32, face_dataset=False, save_model=False):
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    # For logging
    dataset_name = "faces" if face_dataset else "digits"
    print(f"Running {algo_name} on {dataset_name} dataset with {pct}% training data, {runs} runs...")
    
    # Store results for this configuration
    times = []
    accs = []
    last_model = None
    
    for run in range(runs):
        # Split data
        X_train, y_train, X_test, y_test = get_split(X, y, pct=pct, test_pct=test_pct, seed=run)
        
        # Instantiate model based on algorithm
        if algo_name == "perceptron":
            model = Perceptron(lr=lr, epochs=epochs)
        elif algo_name == "scratch":
            model = NeuralNetScratch(
                input_dim=n_features,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                output_dim=n_classes,
                lr=lr,
                epochs=epochs
            )
        else:  # pytorch
            model = PyTorchNN(
                input_dim=n_features,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                output_dim=n_classes,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Train and time
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy(y_test, preds)
        accs.append(acc)
        
        # Print individual run results
        print(f"  Run {run+1}/{runs}: time={elapsed:.3f}s, acc={acc:.3f}")
        
        # Save the last model if requested
        if run == runs - 1 and save_model:
            last_model = model
    
    # Compute statistics
    results = {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'acc_mean': np.mean(accs),
        'acc_std': np.std(accs)
    }
    
    # Add the model to results if requested
    if save_model:
        results['model'] = last_model
    
    # Print summary
    print(f"Summary: time={results['time_mean']:.3f}±{results['time_std']:.3f}s, "
          f"acc={results['acc_mean']:.3f}±{results['acc_std']:.3f}")
    
    return results

def save_trained_models(models_dict):
    # Create directory if it doesn't exist
    models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save each model
    for key, model in models_dict.items():
        model_path = os.path.join(models_dir, f"{key}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_path}")

def main():
    # Configure experiment parameters
    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    test_pct = 20
    algorithms = ["perceptron", "scratch", "pytorch"]
    runs = 5  # Number of runs for each configuration to get stable statistics
    
    # Configure model hyperparameters
    hidden_dim1 = 100
    hidden_dim2 = 50
    lr = 0.01
    epochs = 10
    batch_size = 32
    
    # Load datasets
    (X_faces, y_faces), (X_digits, y_digits) = load_data()
    print(f"Loaded face data: {X_faces.shape}, {y_faces.shape}")
    print(f"Loaded digit data: {X_digits.shape}, {y_digits.shape}")
    
    # Prepare result storage
    results = {
        'digits': defaultdict(list),
        'faces': defaultdict(list)
    }
    
    # For saving trained models (only save 100% training data models)
    trained_models = {}
    
    # Run experiments for digits
    print("\nDIGIT CLASSIFICATION EXPERIMENTS")
    for algo in algorithms:
        for pct in percentages:
            # Determine if we should save this model (only save 100% models)
            save_model = (pct == 100)
            
            res = run_experiment(
                algo_name=algo,
                X=X_digits,
                y=y_digits,
                pct=pct,
                test_pct=test_pct,
                runs=runs,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                face_dataset=False,
                save_model=save_model
            )
            
            # Store results
            results['digits'][algo].append({
                'pct': pct,
                'time_mean': res['time_mean'],
                'time_std': res['time_std'],
                'acc_mean': res['acc_mean'],
                'acc_std': res['acc_std']
            })
            
            # Save model if applicable
            if save_model:
                trained_models[f"digits_{algo}"] = res['model']
    
    # Run experiments for faces
    print("\nFACE CLASSIFICATION EXPERIMENTS")
    for algo in algorithms:
        for pct in percentages:
            # Determine if we should save this model (only save 100% models)
            save_model = (pct == 100)
            
            res = run_experiment(
                algo_name=algo,
                X=X_faces,
                y=y_faces,
                pct=pct,
                test_pct=test_pct,
                runs=runs,
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                face_dataset=True,
                save_model=save_model
            )
            
            # Store results
            results['faces'][algo].append({
                'pct': pct,
                'time_mean': res['time_mean'],
                'time_std': res['time_std'],
                'acc_mean': res['acc_mean'],
                'acc_std': res['acc_std']
            })
            
            # Save model if applicable
            if save_model:
                trained_models[f"faces_{algo}"] = res['model']
    
    # Save results to CSV
    os.makedirs(os.path.join(PROJECT_ROOT, 'report', 'results'), exist_ok=True)
    
    for dataset in ['digits', 'faces']:
        for algo in algorithms:
            df = pd.DataFrame(results[dataset][algo])
            csv_path = os.path.join(PROJECT_ROOT, 'report', 'results', f'{dataset}_{algo}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
    
    # Save trained models
    save_trained_models(trained_models)
    
    # Create and save plots
    create_plots(results)
    
    print("\nExperiments completed...")

def create_plots(results):
    os.makedirs(os.path.join(PROJECT_ROOT, 'report', 'plots'), exist_ok=True)
    algorithms = ["perceptron", "scratch", "pytorch"]
    datasets = ['digits', 'faces']
    
    # Plot accuracy vs training data percentage
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            data = results[dataset][algo]
            x = [d['pct'] for d in data]
            y = [d['acc_mean'] for d in data]
            yerr = [d['acc_std'] for d in data]
            plt.errorbar(x, y, yerr=yerr, label=algo.capitalize(), marker='o', capsize=4)
        
        plt.xlabel('Training Data Percentage (%)')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Training Data Percentage ({dataset.capitalize()})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, 'report', 'plots', f'{dataset}_accuracy.png'))
    
    # Plot training time vs training data percentage
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            data = results[dataset][algo]
            x = [d['pct'] for d in data]
            y = [d['time_mean'] for d in data]
            yerr = [d['time_std'] for d in data]
            plt.errorbar(x, y, yerr=yerr, label=algo.capitalize(), marker='o', capsize=4)
        
        plt.xlabel('Training Data Percentage (%)')
        plt.ylabel('Training Time (s)')
        plt.title(f'Training Time vs Training Data Percentage ({dataset.capitalize()})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, 'report', 'plots', f'{dataset}_time.png'))
    
    # Combined plots for comparison between datasets
    for metric in ['accuracy', 'time']:
        plt.figure(figsize=(12, 8))
        for dataset in datasets:
            for algo in algorithms:
                data = results[dataset][algo]
                x = [d['pct'] for d in data]
                if metric == 'accuracy':
                    y = [d['acc_mean'] for d in data]
                    yerr = [d['acc_std'] for d in data]
                else:
                    y = [d['time_mean'] for d in data]
                    yerr = [d['time_std'] for d in data]
                plt.errorbar(x, y, yerr=yerr, 
                            label=f"{algo.capitalize()} ({dataset.capitalize()})", 
                            marker='o', capsize=3)
        
        plt.xlabel('Training Data Percentage (%)')
        ylabel = 'Accuracy' if metric == 'accuracy' else 'Training Time (s)'
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Training Data Percentage (All Datasets and Algorithms)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_ROOT, 'report', 'plots', f'combined_{metric}.png'))

if __name__ == "__main__":
    main()
