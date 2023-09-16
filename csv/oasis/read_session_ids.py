import os
import csv

def list_directories(path):
    """Return a list of directory names in the specified path."""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def write_to_csv(dir_list, output_filename):
    """Write the directory names to a CSV file."""
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Directory Name"])  # Write header
        for directory in dir_list:
            writer.writerow([directory])

def main():
    folder_path = input("Enter the path to the folder: ")
    output_file = input("Enter the name of the output CSV file: ")

    directories = list_directories(folder_path)
    write_to_csv(directories, output_file)

    print(f"Directory names written to {output_file}")

if __name__ == "__main__":
    main()
