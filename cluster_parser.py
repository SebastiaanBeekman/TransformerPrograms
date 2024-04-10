import re
import numpy as np
import pandas as pd

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
    for file in files:
        train_values = []
        test_values = []
        
        for i in range(1, 5):
            acc = np.zeros((5, 2))
            for j in range(5):
                log_file_path = f"data/cluster/{file}/trainlength{i}0/s{j}/output.log"
            
                with open(log_file_path, 'r') as f:
                    acc[j] = get_acc(f)

            train_values.append(np.round([np.mean(acc[:, 0]), acc[np.argmax(acc[:, 1]), 0], acc[np.argmin(acc[:, 1]), 0], np.median(acc[:, 0])], 4))
            test_values.append(np.round([np.mean(acc[:, 1]), np.max(acc[:, 1]), np.min(acc[:, 1]), np.median(acc[:, 1])], 4))

        # df = pd.DataFrame({'': ['Average', 'Max', 'Min', "median"], 'train': train_values, 'test': test_values})
        # df.to_excel('test.xlsx', index=False)
        df = pd.DataFrame({'': ['Average', 'Max', 'Min', "median"], 'train 1-10': train_values[0], 'test 1-10': test_values[0], 'train 1-20': train_values[1], 'test 21-30': test_values[1], 'train 1-30': train_values[2], 'test 31-40': test_values[2], 'train 1-40': train_values[3], 'test 41-50': test_values[3]})
        df.to_excel(f'data/excel/{file}.xlsx', index=False, float_format='%.4f')