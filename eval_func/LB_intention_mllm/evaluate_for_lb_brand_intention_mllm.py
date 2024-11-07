import json
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

def main(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Extract true labels and predicted logits
    y_true = []
    logits = []
    for item in data:
        try:
            y_true.append(item['gt_label'])
            logits.append([float(x) for x in item['logits'].split('\t')])
        except:
            item = item[0]
            y_true.append(item['gt_label'])
            logits.append([float(x) for x in item['logits'].split('\t')])            

    # Convert labels to numbers
    labels = ['<A>', '<B>', '<C>', '<D>', '<E>', '<F>']
    y_true = [labels.index(label) for label in y_true]

    # Use argmax to compute overall six-class predictions
    logits = np.array(logits)
    y_pred = np.argmax(logits, axis=1)

    # Generate classification report as a dictionary
    report_dict = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    # Convert dictionary to DataFrame for easier formatting
    report_df = pd.DataFrame(report_dict).transpose()
    # Convert the support column to integers
    report_df['support'] = report_df['support'].astype(int)
    # Remove the accuracy row
    report_df = report_df.drop(index='accuracy')
    # Format the floating-point numbers to the desired number of decimal places
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].applymap(lambda x: f"{x:.4f}")

    overall_report = report_df
    overall_conf_matrix = confusion_matrix(y_true, y_pred)

    # Calculate P95R, P90R, P80R, P70R, R95P, R90P metrics for each class
    metrics = {label: {'P70R': None, 'P80R': None, 'P90R': None, 'P95R': None, 'R95P': None, 'R90P': None} for label in labels}

    for i in range(len(labels)):
        class_logits = logits[:, i]
        y_true_bin = (np.array(y_true) == i).astype(int)
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true_bin, class_logits)

        # Calculate P95R, P90R, P80R, P70R
        for precision_target in [0.70, 0.80, 0.90, 0.95]:
            best_index = np.argmin(np.abs(precisions - precision_target))
            if best_index < len(thresholds):
                metrics[labels[i]][f'P{int(precision_target * 100)}R'] = {
                    'Precision': precisions[best_index],
                    'Recall': recalls[best_index],
                    'F1-score': 2 * (precisions[best_index] * recalls[best_index]) / (precisions[best_index] + recalls[best_index] + 1e-9),
                    'Threshold': thresholds[best_index]
                }

        # Calculate R95P and R90P
        for recall_target in [0.95, 0.90]:
            best_index = np.argmin(np.abs(recalls - recall_target))
            if best_index < len(thresholds):
                metrics[labels[i]][f'R{int(recall_target * 100)}P'] = {
                    'Precision': precisions[best_index],
                    'Recall': recalls[best_index],
                    'F1-score': 2 * (precisions[best_index] * recalls[best_index]) / (precisions[best_index] + recalls[best_index] + 1e-9),
                    'Threshold': thresholds[best_index]
                }

    # Prepare output content
    output = []
    output.append("Overall PR:")
    output.append(overall_report.to_string())
    output.append("Overall Confusion Matrix:")
    output.append(str(overall_conf_matrix))
    output.append("")

    for label in labels:
        output.append(f"{label}:")
        for metric, values in metrics[label].items():
            if values is not None:
                output.append(f"  {metric} - Precision: {values['Precision']:.3f}, Recall: {values['Recall']:.3f}, F1-score: {values['F1-score']:.3f}, Threshold: {values['Threshold']:.3f}")
            else:
                output.append(f"  {metric} - Precision: , Recall: , F1-score: , Threshold: ")
        output.append("")

    output_text = "\n".join(output)

    # Print results
    print(output_text)

    # Save to metrics.txt
    metrics_file_path = os.path.join(os.path.dirname(file_path), 'metrics.txt')
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate metrics for LB-MLLM.')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file containing the test results.')
    args = parser.parse_args()
    main(args.file_path)
