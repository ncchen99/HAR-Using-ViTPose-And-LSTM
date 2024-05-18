import csv
import yaml
import os
import pandas as pd 

# 讀取 info.csv
def read_info_csv(file_path, binary_category):
    binary_category = [categories for key, categories in binary_category.items()]
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        info = []
        category = None
        for row in reader:
            if len(row) == 4 and not row[3]:  # new category header
                if category:
                    info.append(category)
                id = row[1]
                for i, rrow in enumerate(binary_category):
                    if row[0] in rrow:
                        id = rrow.index(row[0])
                category = {
                    "name": row[0],
                    "id": row[1],
                    "count": int(row[2]),
                    "files": [],
                    "text": [[row[0], id, row[2], ""]]
                }
            else:  # file entry
                category["files"].append({
                    "filename": row[0],
                    "width": row[1],
                    "height": row[2],
                    "frames": row[3]
                })
                category["text"].append(row)
        if category:
            info.append(category)
    return info

# 讀取 data.csv
def read_data_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        data = list(reader)
    return headers, data

# 讀取 binary_category.yaml
def read_yaml(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)

# 分割 data.csv 並儲存成不同的檔案
def split_and_save_data(info, data, headers, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    index = 0
    for category_info in info:
        # category_info = next((cat for cat in info if cat["name"] == category), None)
        output_file_path = os.path.join(output_dir, f"{category_info['name']}.csv")
        with open(output_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for file_info in category_info["files"]:
                for row in data[index:index + int(file_info["frames"])]:
                    writer.writerow(row)
                index += int(file_info["frames"])

def combine_data(info, binary_category, output_dir):
    for key, categories in binary_category.items():
        # merging two csv files 
        df = pd.concat( 
            map(pd.read_csv, [f'{output_dir}/{categories[0]}.csv', f'{output_dir}/{categories[1]}.csv']), ignore_index=True) 
        df.to_csv(f'{output_dir}/data_{key}.csv', index=False)
        with open(f'{output_dir}/info_{key}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for category in categories:
                writer.writerows(list(cat for cat in info if cat["name"] == category)[0]["text"])
                
# 主函式
def main():
    info_file_path = "train_info.csv"
    data_file_path = "train_data.csv"
    yaml_file_path = "binary_category.yaml"
    output_dir = "output_train"

    binary_category = read_yaml(yaml_file_path)
    info = read_info_csv(info_file_path, binary_category)
    headers, data = read_data_csv(data_file_path)
    

    split_and_save_data(info, data, headers, output_dir)
    combine_data(info, binary_category, output_dir)
    
if __name__ == "__main__":
    main()


