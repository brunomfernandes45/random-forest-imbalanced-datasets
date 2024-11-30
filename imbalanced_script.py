import openml
import pandas as pd

def check_class_imbalance(threshold=0.8):
    # Get datasets from OpenML Collection with ID 99
    datasets = openml.study.get_suite(99)  # Suite ID 99

    # List to store datasets with imbalance
    imbalanced_datasets = []

    print("Checking for class imbalance in OpenML Collection 99...\n")

    for dataset_id in datasets.data:
        try:
            # Download the dataset
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, attributes = dataset.get_data(target=dataset.default_target_attribute)

            # Check if the target column is categorical
            if not isinstance(y, pd.CategoricalDtype):
                y = pd.Categorical(y)

            # Calculate class distribution
            class_counts = y.value_counts(normalize=True)
            max_class_proportion = class_counts.max()

            # Check for imbalance
            if max_class_proportion >= threshold:
                imbalanced_datasets.append({
                    "Dataset ID": dataset_id,
                    "Dataset Name": dataset.name,
                    "Max Class Proportion": max_class_proportion,
                    "Class Distribution": class_counts.to_dict(),
                })

        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            continue

    # sort the imbalanced datasets by max class proportion descending
    imbalanced_datasets = sorted(imbalanced_datasets, key=lambda x: x["Max Class Proportion"], reverse=True)
    
    # Print results
    if imbalanced_datasets:
        print("Datasets with class imbalance greater than the threshold:\n")
        for dataset in imbalanced_datasets:
            print(f"Dataset ID: {dataset['Dataset ID']}")
            print(f"Dataset Name: {dataset['Dataset Name']}")
            print(f"Max Class Proportion: {dataset['Max Class Proportion']:.2f}")
            print("Class Distribution:")
            for cls, proportion in dataset["Class Distribution"].items():
                print(f"  {cls}: {proportion:.2%}")
            print()
    else:
        print("No datasets found with class imbalance greater than the threshold.")

if __name__ == "__main__":
    # Run the script
    check_class_imbalance(threshold=0.8)