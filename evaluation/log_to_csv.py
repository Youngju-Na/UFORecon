import csv

import argparse


# The path to your log file
parser = argparse.ArgumentParser()
parser.add_argument("--log_file_path", type=str, default='eval_final.log', help="log file path")
parser.add_argument("--csv_output_path", type=str, default='out.csv', help="csv output path")

# parse the arguments
args = parser.parse_args()
log_file_path = args.log_file_path
csv_output_path = args.csv_output_path

# Initialize an empty list to hold the data for each row
data = []

# Open and read the log file
with open(log_file_path, 'r') as file:
    for line in file:
        # Skip lines that don't start with the expected pattern
        if not line.startswith('INFO:root:scan:'):
            continue
        
        # Remove 'INFO:root:' prefix and split the line into parts
        parts = line.replace('INFO:root:', '').split('|')
        
        # Parse each part to extract the numeric values
        scan_value = int(parts[0].split(':')[1].strip())
        d2s_value = float(parts[1].split(':')[1].strip())
        s2d_value = float(parts[2].split(':')[1].strip())
        all_value = float(parts[3].split(':')[1].strip())
        
        # Add the parsed values to the data list as a new row
        data.append([scan_value, d2s_value, s2d_value, all_value])

# Write the data to a CSV file
with open(csv_output_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header
    csv_writer.writerow(['scan', 'd2s', 's2d', 'all'])
    
    # Write the data rows
    csv_writer.writerows(data)

print(f"CSV file has been created at {csv_output_path}.")