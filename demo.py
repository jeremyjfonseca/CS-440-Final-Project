import os
import sys
import time
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from data_loader.loader import load_data, get_split

def display_ascii_image(image, height, width, title="Image"):
    # Reshape the flattened image
    img_2d = image.reshape(height, width)
    
    # Display using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img_2d, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def print_ascii_image(image, height, width):
    # Reshape the flattened image
    img_2d = image.reshape(height, width)
    
    # Print to console
    for row in range(height):
        line = ""
        for col in range(width):
            if img_2d[row, col] > 0.5:
                line += "#"
            else:
                line += " "
        print(line)

def load_model(model_type, dataset_type):
    models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    if not os.path.exists(models_dir):
        sys.exit(1)
        
    model_path = os.path.join(models_dir, f"{dataset_type}_{model_type}.pkl")
    if not os.path.exists(model_path):
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict_and_display(model, X, y, indices, dataset_type):
    # Set dimensions based on dataset type
    if dataset_type == 'digits':
        height, width = 28, 28
        label_names = [str(i) for i in range(10)]
    else:  # faces
        height, width = 70, 60
        label_names = ['Not Face', 'Face']
    
    # Make predictions for each selected sample
    for idx in indices:
        x_sample = X[idx].reshape(1, -1)
        true_label = y[idx]
        
        # Time the prediction
        start_time = time.time()
        pred_label = model.predict(x_sample)[0]
        pred_time = time.time() - start_time
        
        # Display result
        accuracy = 1.0 if pred_label == true_label else 0.0
        print(f"\nSample #{idx}:")
        print(f"True label: {label_names[true_label]}")
        print(f"Predicted label: {label_names[pred_label]}")
        print(f"Accuracy: {accuracy:.1f}")
        print(f"Prediction time: {pred_time*1000:.2f} ms")
        
        # Display image
        title = f"Sample #{idx} - True: {label_names[true_label]}, Pred: {label_names[pred_label]}"
        display_ascii_image(X[idx], height, width, title)
        
        # Also print ASCII representation
        print("\nASCII Representation:")
        print_ascii_image(X[idx], height, width)
        
        # Wait for user to press Enter before continuing
        input("\nPress Enter to continue...")

def main():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--model', choices=['perceptron', 'scratch', 'pytorch'], default='pytorch')
    parser.add_argument('--dataset', choices=['digits', 'faces'], default='digits')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    print(f"\nRunning demo with {args.model} model on {args.dataset} dataset...\n")
    
    # Load data
    if args.dataset == 'digits':
        (_, _), (X, y) = load_data()
    else:  # faces
        (X, y), (_, _) = load_data()
    
    # Split data (just to get some test samples)
    _, _, X_test, y_test = get_split(X, y, pct=80, test_pct=20, seed=42)
    
    # Load the pre-trained model
    print(f"Loading {args.model} model for {args.dataset}...")
    model = load_model(args.model, args.dataset)
    
    # Select random samples for prediction
    np.random.seed(42)
    indices = np.random.choice(len(X_test), min(args.num_samples, len(X_test)), replace=False)
    
    # Make predictions and display results
    predict_and_display(model, X_test, y_test, indices, args.dataset)
    
    print("\nDemo completed...")

if __name__ == "__main__":
    main() 
