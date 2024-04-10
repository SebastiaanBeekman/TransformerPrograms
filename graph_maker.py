import re
import numpy as np
import matplotlib.pyplot as plt

train_acc_pattern = re.compile(r'train: loss=\d+\.\d+, acc=([\d.]+),')
test_acc_pattern = re.compile(r'test: loss=\d+\.\d+, acc=([\d.]+),')
            
def get_acc(file):    
    for line in file:
        train_match = train_acc_pattern.search(line)
        if train_match:
            train_acc = float(train_match.group(1))
        
        test_match = test_acc_pattern.search(line)
        if test_match:
            test_acc = float(test_match.group(1))
    return train_acc, test_acc

if __name__ == "__main__":
    files = ["double_hist", "dyck1", "dyck2", "hist", "mostfreq", "reverse", "sort"]
    train_values = {file: [[], []] for file in files}
    x = [10, 20, 30 , 40]
    
    for file in files:
        for i in range(1, 5):
            acc = np.zeros((5, 2))
            for j in range(5):
                log_file_path = f"data/cluster/{file}/trainlength{i}0/s{j}/output.log"
            
                with open(log_file_path, 'r') as f:
                    acc[j] = get_acc(f)

            train_values[file][0].append(np.round(np.mean(acc[:, 1]), 4))
            train_values[file][1].append(np.round(np.std(acc[:, 1]), 4))
    
        plt.errorbar(x, train_values[file][0], yerr=train_values[file][1], label=file, marker='o', capsize=5)

    plt.rcParams.update({'font.size': 14})
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.xlabel('Train Length', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Length generalization: Test accuracy over different lengths')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.show()