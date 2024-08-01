import dask.dataframe as dd
import pdb
import pandas as pd
import glob
import os
import re

# 定义一个函数来检查每个值的数据类型
def check_types(df):
    return df['sequence_item_ids'].apply(lambda x: type(x).__name__)

def safe_split(x):
    if isinstance(x, str):
        # 使用正则表达式去除连续的逗号
        cleaned_str = re.sub(r',+', ',', x)
        # 去掉前后的逗号
        cleaned_str = cleaned_str.strip(',')
        return cleaned_str.split(',')
    elif isinstance(x, float):
        cleaned_str = re.sub(r',+', ',', str(x))
        cleaned_str = cleaned_str.strip(',')
        return cleaned_str.split(',')
    else:
        return x

def sum_clicks(clicks):
    return sum(map(int, clicks))

# 定义检查长度是否相同的函数
def lengths_match(row):
    if isinstance(row['sequence_item_ids_tovalid'], list) and isinstance(row['sequence_timestamps_tovalid'], list):
        return len(row['sequence_item_ids_tovalid']) == len(row['sequence_timestamps_tovalid']) == len(row['click_timestamps_tovalid']) == len(row['whethe_click_tovalid'])
    return Fals

def is_last_element_one(row):
    elements = row.split(',')
    return elements[-1] == '1'

def process_file(file):
    # 读取所有文件并合并到一个 DataFrame 中
    ddf = dd.read_csv(file, delimiter='\t', names=['uid', 'sequence_item_ids', 'whethe_click', 'sequence_timestamps'])
    print(len(ddf))
    

    ddf['whethe_click_sum'] = ddf['whethe_click'].apply(safe_split, meta=('x', 'object'))
    ddf['whethe_click_sum'] = ddf['whethe_click_sum'].apply(sum_clicks, meta=('whethe_click_sum', 'int'))
    
    # 删除click小于3的数据
    filtered_ddf = ddf[ddf['whethe_click_sum'] >= 3].reset_index(drop=True).drop(columns=['whethe_click_sum'])
    print(len(filtered_ddf))
    
    # 过滤缺失数据
    filtered_ddf['sequence_item_ids_tovalid'] = filtered_ddf['sequence_item_ids'].apply(safe_split, meta=('x', 'object'))
    filtered_ddf['sequence_timestamps_tovalid'] = filtered_ddf['sequence_timestamps'].apply(safe_split, meta=('x', 'object'))
    filtered_ddf = filtered_ddf[filtered_ddf['whethe_click'].apply(is_last_element_one)]
    filtered_ddf = filtered_ddf[filtered_ddf.apply(lambda row: lengths_match(row), axis=1, meta=('x', 'bool'))].drop(columns=['sequence_item_ids_tovalid', 'sequence_timestamps_tovalid', 'whethe_click_tovalid'])
    
    print(len(filtered_ddf))
    return filtered_ddf.compute()

file_pattern = '../../data/month/20240613/part-00000.gz'
files = sorted(glob.glob(file_pattern))
# 初始化空的 Pandas DataFrame 用于合并结果
output_directory = 'true_data/month'
os.makedirs(output_directory, exist_ok=True)

# # 逐个文件处理并合并结果，每处理10个文件保存一次
batch_size = 10
batch_counter = 0
part_counter = 0
for file in files: 
    print(file)
    filtered_df = process_file(file)
    pdb.set_trace()
    filtered_df.to_csv(f'{output_directory}/month/part-{part_counter}', index=False, header=False, sep='\t')





