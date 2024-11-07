import argparse
import yaml
import os

def update_json_paths(yaml_file, base_path):
    # 打开并解析 YAML 文件
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # 遍历数据并更新 json_path
    for dataset in data.get('datasets', []):
        if 'json_path' in dataset:
            # 拼接 json_path
            dataset['json_path'] = os.path.join(base_path, dataset['json_path'])

    return data

def save_updated_yaml(output_file, updated_data):
    with open(output_file, 'w') as file:
        yaml.dump(updated_data, file, default_flow_style=False, allow_unicode=True)

def main():
    """
    python3 ./scripts/CCExpert_data_scripts/add_CC_Foundation_local_absolute_path_to_yaml.py \
        --yaml_file="./scripts/CCExpert_data_scripts/cptdata_RSupsampled_template.yaml" \
        --base_path="/mnt/bn/chenhaobo-va-data/wangmingze/Research/CC-Foundation"
    python3 ./scripts/CCExpert_data_scripts/add_CC_Foundation_local_absolute_path_to_yaml.py \
        --yaml_file="./scripts/CCExpert_data_scripts/benchmark_LEVIR-CC_train_template.yaml" \
        --base_path="/mnt/bn/chenhaobo-va-data/wangmingze/Research/CC-Foundation"
    python3 ./scripts/CCExpert_data_scripts/add_CC_Foundation_local_absolute_path_to_yaml.py \
        --yaml_file="./scripts/CCExpert_data_scripts/benchmark_LEVIR-CC_test_template.yaml" \
        --base_path="/mnt/bn/chenhaobo-va-data/wangmingze/Research/CC-Foundation"
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Update json_paths in YAML file")
    parser.add_argument('--yaml_file', type=str, default="./scripts/CCExpert_data_scripts/cptdata_RSupsampled_template.yaml", help="Path to the YAML file")
    parser.add_argument('--base_path', type=str, help="Base path to prepend to json_path")
    parser.add_argument('--output_file', type=str, default="User Not input", help="Path to save the updated YAML file")
    
    args = parser.parse_args()

    # 更新 json_paths
    updated_data = update_json_paths(args.yaml_file, args.base_path)

    if args.output_file == "User Not input":
        output_file = args.yaml_file.replace("template.yaml", "absolute_path.yaml")
    else:
        output_file = args.output_file
    # 保存更新后的 YAML 文件
    save_updated_yaml(output_file, updated_data)
    print(f"Updated YAML saved to {output_file}")

if __name__ == "__main__":
    main()
