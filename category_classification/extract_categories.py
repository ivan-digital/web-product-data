import csv

def get_category_names(map_file="PDC2020_map.tsv"):
    """Reads the mapping file and returns a sorted list of category names."""
    cluster2cat = {}
    with open(map_file, 'r', encoding='utf-8') as fh:
        for cid, cat in csv.reader(fh, delimiter='\t'):
            cluster2cat[int(cid)] = cat.strip()
    
    # Get unique category names and sort them to match the ClassLabel encoding
    sorted_labels = sorted(list(set(cluster2cat.values())))
    return sorted_labels

def main():
    """
    Extracts category names for the given indices from the confusion matrix.
    """
    category_indices = [0, 1, 2, 3, 4, 5]
    
    try:
        labels = get_category_names()
        
        print("Category names for the given indices:")
        for index in category_indices:
            if 0 <= index < len(labels):
                print(f"{index}: {labels[index]}")
            else:
                print(f"{index}: Index out of range")

    except FileNotFoundError:
        print(f"Error: The file 'PDC2020_map.tsv' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
