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
    font_size = 14
    x = [10, 20, 30, 40]

    lines = [
        (0, (1, 10)),
        (0, (1, 1)),
        (0, (1, 1)),
        (5, (10, 3)),
        (0, (5, 10)),
        (0, (5, 5)),
        (0, (5, 1)),
    ]

    fig = plt.figure(figsize=(6, 8))
    fig.subplots_adjust(left=0.11, right=0.72, top=0.9, bottom=0.065)
    ax_list = [None] * 7

    for idx, file in enumerate(files):
        for i in range(1, 5):
            acc = np.zeros((5, 2))
            for j in range(5):
                log_file_path = f"data/cluster/{file}/trainlength{i}0/s{j}/output.log"

                with open(log_file_path, 'r') as f:
                    acc[j] = get_acc(f)

            train_values[file][0].append(np.round(np.mean(acc[:, 1]), 4))
            train_values[file][1].append(np.round(np.std(acc[:, 1]), 4))

        ax_list[i] = plt.errorbar(
            x, train_values[file][0], yerr=train_values[file][1], label=file, marker='o', capsize=5, linestyle=lines[idx]
        )

    # plt.rcParams.update({'font.size': font_size})
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.01)
    plt.rcParams['axes.titley'] = 1.02
    plt.xlabel('Test Length')  # , fontsize=font_size)
    plt.ylabel('Accuracy (%)')  # , fontsize=font_size)
    plt.title('Length generalization: Test accuracy over different lengths')
    plt.grid(True)
    # plt.legend(title='Tasks', bbox_to_anchor=(1.02, 1.015))
    plt.savefig('test.png', transparent=True, dpi=600)
    plt.show()
