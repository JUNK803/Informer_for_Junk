import pandas as pd
import os
import shutil
from pathlib import Path

def move_temperature_to_end(input_dir, backup=True):
    """
    将文件夹中所有 CSV 文件的 'temperature' 列移动到最后一列
    
    参数:
        input_dir (str): CSV 文件所在文件夹路径
        backup (bool): 是否备份原文件（默认备份）
    """
    # 1. 验证目录
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"目录不存在: {input_dir}")
    
    # 2. 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print("⚠️  警告: 目录中未找到 CSV 文件")
        return
    
    print(f"📁 检测到 {len(csv_files)} 个 CSV 文件，开始处理...\n")
    
    # 3. 处理每个文件
    success_count = 0
    error_count = 0
    not_found_count = 0
    
    for filename in sorted(csv_files):
        file_path = os.path.join(input_dir, filename)
        
        try:
            # 读取 CSV（自动识别索引列）
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            # --- A. 预处理：剔除非数值列 ---
            if 'weather_description' in df.columns:
                df = df.drop(columns=['weather_description'])

            # 已经有 wind_dir_sin、wind_dir_cos 表示风向了，此列为冗余
            if 'wind_direction' in df.columns:
                df = df.drop(columns=['wind_direction'])
            
            # 检查 temperature 列是否存在
            if 'temperature' not in df.columns:
                print(f"⚠️  跳过 {filename}: 未找到 'temperature' 列")
                not_found_count += 1
                continue
            
            # 记录原始列位置
            original_pos = df.columns.get_loc('temperature') + 1  # 1-indexed
            
            # === 方法1：使用 pop + append（推荐）===
            temp_col = df.pop('temperature')  # 移除并返回该列
            df['temperature'] = temp_col       # 添加到末尾
            
            # === 方法2：使用列重排（等效）===
            # cols = [col for col in df.columns if col != 'temperature'] + ['temperature']
            # df = df[cols]
            
            # 备份原文件
            if backup:
                backup_dir = os.path.join(input_dir, 'original_backup')
                Path(backup_dir).mkdir(exist_ok=True)
                shutil.copy2(file_path, os.path.join(backup_dir, filename))
            
            # 保存修改后的文件
            df.to_csv(file_path)
            
            new_pos = len(df.columns)
            print(f"✅ {filename}: 'temperature' 从第 {original_pos} 列 → 移动到第 {new_pos} 列")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {str(e)}")
            error_count += 1
    
    # 4. 生成处理报告
    print("\n" + "="*60)
    print("📊 处理完成报告")
    print("="*60)
    print(f"总文件数      : {len(csv_files)}")
    print(f"成功处理      : {success_count}")
    print(f"未找到列      : {not_found_count}")
    print(f"处理失败      : {error_count}")
    if backup:
        print(f"✓ 原始文件已备份至: {os.path.join(input_dir, 'original_backup')}")
    print("="*60)

# ============ 使用示例 ============
if __name__ == "__main__":
    # 修改为你的实际文件夹路径
    INPUT_FOLDER = '/Users/junk/vscode项目/20260129_数据处理和设计/Informer_copy/data'  # 例如: 'step2_4_city_scientific_imputed'
    
    # 执行（backup=True 会备份原文件，安全第一）
    move_temperature_to_end(input_dir=INPUT_FOLDER, backup=True)