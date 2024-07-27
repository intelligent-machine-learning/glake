import json

def jsonl_to_json(input_file_path, output_file_path):
    """
    将JSONL文件转换为JSON数组文件。
    
    :param input_file_path: 输入的JSONL文件路径
    :param output_file_path: 输出的JSON文件路径
    """
    # 用于存放所有解析后的JSON对象
    data_list = []
    
    # 打开JSONL文件并逐行读取
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 每一行都是一个独立的JSON对象，使用json.loads解析
            json_obj = json.loads(line.strip())
            data_list.append(json_obj)
    
    # 将所有解析的JSON对象写入到一个新的JSON文件中，形成一个数组
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

# 示例使用
input_path_list = ['loogle_SD_mixup/loogle_SD_mixup_16k.jsonl', 'hotpotwikiqa_mixup/hotpotwikiqa_mixup_16k.jsonl']
output_path_list = ['loogle_SD_mixup/loogle_SD_mixup_16k.json', 'hotpotwikiqa_mixup/hotpotwikiqa_mixup_16k.json']

for i in range(len(input_path_list)):
    jsonl_to_json(input_path_list[i], output_path_list[i])

