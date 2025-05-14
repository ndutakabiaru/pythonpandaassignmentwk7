import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def analyze_and_visualize_data():
    """
    Loads, analyzes, and visualizes the Iris dataset.
    """
    try:
        # 1. Load the Dataset
        iris = load_iris(as_frame=True)
        df = iris.frame

        # 2. Basic Data Analysis
        print("Basic statistics:")
        print(df.describe())

        print("\nMean of numerical columns grouped by species:")
        print(df.groupby('target').mean())

        # 3. Data Visualization
        sns.set_style("whitegrid")

        # Bar chart: Average petal length per species
        plt.figure(figsize=(8, 6))
        sns.barplot(x='target', y='petal length (cm)', data=df)
        plt.title('Avg Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Avg Petal Length')
        plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
        plt.show()

        # Scatter plot: Sepal length vs. petal length
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(title='Species', labels=iris.target_names)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_and_visualize_data()
