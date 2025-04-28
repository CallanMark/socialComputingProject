import json
import csv
import argparse
import os

def process_results_to_csv(input_json_path, output_csv_path):
    """
    Process JSON results from model trials into a CSV file.
    
    Args:
        input_json_path (str): Path to the input JSON file containing trial results
        output_csv_path (str): Path to the output CSV file
    """
    # Read the JSON file
    with open(input_json_path, 'r') as f:
        results = json.load(f)
    
    # Extract the best parameters and accuracies
    best_params = results.get('best_params', {})
    best_val_accuracy = results.get('best_val_accuracy', 'N/A')
    best_test_accuracy = results.get('best_test_accuracy', 'N/A')
    
    # Get all trials
    all_trials = results.get('all_trials', [])
    
    # Prepare CSV headers (dynamically based on the parameters in the first trial)
    if all_trials:
        # Get all parameter keys from the first trial
        param_keys = sorted(all_trials[0].get('params', {}).keys())
        headers = ['trial_number'] + param_keys + ['val_accuracy', 'test_accuracy', 'state']
    else:
        # Fallback headers if no trials are found
        headers = ['trial_number', 'val_accuracy', 'test_accuracy', 'state']
    
    # Write to the CSV file
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers
        writer.writerow(headers)
        
        # Write trial data
        for trial in all_trials:
            row = [trial.get('number')]
            
            # Add parameter values in the same order as headers
            for key in param_keys:
                row.append(trial.get('params', {}).get(key, 'N/A'))
            
            # Add validation accuracy, test accuracy, and state
            row.append(trial.get('value', 'N/A'))
            row.append(trial.get('test_accuracy', 'N/A'))
            row.append(trial.get('state', 'N/A'))
            
            writer.writerow(row)
        
        

def main():
    parser = argparse.ArgumentParser(description='Process model trial results from JSON to CSV')
    parser.add_argument('input_json', help='Path to the input JSON file')
    parser.add_argument('--output', '-o', help='Path to the output CSV file (default: results.csv)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    output_path = args.output if args.output else 'results.csv'
    
    # Process results
    process_results_to_csv(args.input_json, output_path)

if __name__ == "__main__":
    main()