import re
import matplotlib.pyplot as plt

def extract_and_plot(log_path, metric_name, save_path = None):
    # Regex to match any line with 'Epoch' in it
    pattern = re.compile(r'^.*Epoch.*$', re.MULTILINE)

    training_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    epochs = []

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 1
        e = 1        
        for line in file:
            match = pattern.search(line)

            if match:
                parts = match.group(0).split()
                print(parts)
                # Parse epoch number
                for i, val in enumerate(parts):
                    if val == "Epoch":
                        epoch_str = parts[i + 1].strip('[],')
                        print("EPOCH", epoch_str)
                        epochs.append(float(e))
                        e += 1 

                # Parse metrics
                for i, val in enumerate(parts):
                    if i + 2 < len(parts):
                        if val == "Training" and parts[i+1] == "Loss:":
                            print("VAL FOR TRAINING LOSS", parts[i+2])
                            try:
                                training_loss.append(float(parts[i+2]))
                            except:
                                training_loss.append(float(parts[i+2][:len(parts[i+2])-1]))
                        elif val == "Test" and parts[i+1] == "Loss:":
                            try:
                                test_loss.append(float(parts[i+2]))
                            except:
                                test_loss.append(float(parts[i+2][:len(parts[i+2])-1]))
                        elif (val == "Accuracy:") & (parts[i+2] == '|'):
                            try:
                                print("ENETERS IN THE BITCH")
                                train_accuracy.append(float(parts[i+1].strip(',')))
                            except:
                                pass
                        elif val == "Test" and parts[i+1] == "Accuracy:":
                            try:
                                test_accuracy.append(float(parts[i+2].strip(',')))
                            except:
                                pass

                line_count += 1

    # Print everything
    print("Epochs:", epochs)
    print("Training Loss:", training_loss)
    print("Test Loss:", test_loss)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Select the appropriate values to plot
    metric_map = {
        "Training Loss": training_loss,
        "Test Loss": test_loss,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy
    }

    if metric_name not in metric_map:
        print(f"Unknown metric '{metric_name}'. Choose from: {list(metric_map.keys())}")
        return

    values = metric_map[metric_name]

    # Make sure lengths match
    if len(epochs) != len(values):
        print("Warning: Epochs and metric values are of different lengths!")
        min_len = min(len(epochs), len(values))
        epochs = epochs[:min_len]
        values = values[:min_len]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values, marker='o')
    plt.title(f"{metric_name} Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True)
    filename = save_path or f"{metric_name.lower().replace(' ', '_')}_over_epochs.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to: {filename}")
    plt.close()

extract_and_plot("run_logs_nursery_lr_0.06_maxdepth_3_batch_32_numtrees_1_subsetselection_True_04-18-03-33_nummasked25.log", "Test Accuracy")
