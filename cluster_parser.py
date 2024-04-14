import re
import numpy as np
import pandas as pd

train_acc_pattern = re.compile(r'\(train\): loss=(nan|[\d.e-]+), acc=([\d.]+)')
val_acc_pattern = re.compile(r'\(val\): loss=(nan|[\d.e-]+), acc=([\d.]+)')
test_acc_pattern = re.compile(r'\(test\): loss=(nan|[\d.e-]+), acc=([\d.]+)')
            
def get_acc(file):
    train_acc, val_acc, test_acc = None, None, None
    for line in file:
        
        train_match = train_acc_pattern.search(line)
        if train_match:
            train_acc = float(train_match.group(2))
            
        val_match = val_acc_pattern.search(line)
        if val_match:
            val_acc = float(val_match.group(2))
        
        test_match = test_acc_pattern.search(line)
        if test_match:
            test_acc = float(test_match.group(2))
            
    if not train_acc or not val_acc or not test_acc:
        print("Error: Acc not found")
        print(file.name)
            
    return train_acc, val_acc, test_acc

def length_gen(files):
    for file in files:
        train_values = {file : [[], [], [], []] for file in files}
        test_values = {file : [[], [], [], []] for file in files}
        for i in range(1, 5):
            acc = np.zeros((5, 3))
            for j in range(5):
                log_file_path = f"data/cluster/length/{file}/trainlength{i}0/s{j}/output.log"
            
                with open(log_file_path, 'r') as f:
                    acc[j] = get_acc(f)

            train_values[file][0].append(np.round(np.mean(acc[:, 0])))
            train_values[file][1].append(np.round(acc[np.argmax(acc[:, 2]), 0]))
            train_values[file][2].append(np.round(acc[np.argmin(acc[:, 2]), 0]))
            train_values[file][3].append(np.round(np.median(acc[:, 0])))
            
            test_values[file][0].append(np.round(np.mean(acc[:, 2]), 4))
            test_values[file][1].append(np.round(np.max(acc[:, 2]), 4))
            test_values[file][2].append(np.round(np.min(acc[:, 2]), 4))
            test_values[file][3].append(np.round(np.median(acc[:, 2]), 4))

            
            train_values[file][0].append(np.round([np.mean(acc[:, 0]), acc[np.argmax(acc[:, 1]), 0], acc[np.argmin(acc[:, 1]), 0], np.median(acc[:, 0])], 4))
            test_values[file][1].append(np.round([np.mean(acc[:, 1]), np.max(acc[:, 1]), np.min(acc[:, 1]), np.median(acc[:, 1])], 4))
    
        # df = pd.DataFrame({'': ['Average', 'Max', 'Min', "median"], 'train 1-10': train_values[0], 'test 1-10': test_values[0], 'train 1-20': train_values[1], 'test 21-30': test_values[1], 'train 1-30': train_values[2], 'test 31-40': test_values[2], 'train 1-40': train_values[3], 'test 41-50': test_values[3]})
        # df.to_excel(f'data/excel/{file}.xlsx', index=False, float_format='%.4f')
        
def grid_search(files):
    for l in [1, 2, 3]: # Layers
        test_values = {file : [] for file in files}
        for file in files:
            for h in [2, 4, 8, 16]: # Heads
                for m in [2, 4, 8]: # MLPs 
                    acc = np.zeros((5, 3))
                    for i in range(5):
                        log_file_path = f"data/cluster/new/{file}/nlayers{l}heads{h}mlps{m}/s{i}/output.log"
                        try :
                            with open(log_file_path, 'r') as f:
                                acc[i] = get_acc(f)
                        except:
                            print(log_file_path)
                            continue
                    test_values[file].append(np.round(acc[np.argmax(acc[:, 1]), 2], 4))
        df = pd.DataFrame({file: test_values[file] for file in files}).transpose()
        df.to_excel(f'data/excel/new/nlayers{l}.xlsx', index=True, float_format='%.4f')


if __name__ == "__main__":
    files = ["addition", "addition_hints"]
    
    # length_gen(files)
    grid_search(files)