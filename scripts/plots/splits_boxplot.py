import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    split_sweep_path = "outputs/split_sweep/summary.csv"
    df = pd.read_csv(split_sweep_path)

    plt.figure(figsize=(8, 6))
    # boxplot of avg_auc grouped by option
    sns.boxplot(data=df, x="option", y="distance")
    # plt.title("Split Distance vs. Average AUC")
    plt.xlabel("Type")
    plt.ylabel("Average AUC across species")
    plt.legend(title="Split Option")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()