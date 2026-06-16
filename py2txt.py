import os
import shutil

def process_files(source_folder, output_dir, recursive=True, keep_structure=True):
    """
    处理文件并生成 .txt 副本
    :param source_folder: 单个源文件夹路径
    :param output_dir: 统一输出目录
    :param recursive: 是否递归遍历子目录 True/False
    :param keep_structure: 是否保留原目录层级 True/False，False 则全部平铺
    """
    # 遍历目录
    if recursive:
        file_iterator = os.walk(source_folder)
    else:
        # 不递归，只遍历当前目录
        root = source_folder
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        file_iterator = [(root, [], files)]

    abs_source = os.path.abspath(source_folder)
    for root, _, files in file_iterator:
        for file in files:
            if file.endswith((".py", ".yaml")):
                src_path = os.path.join(root, file)
                dst_filename = file + ".txt"

                if keep_structure:
                    # 复刻原目录层级
                    rel_path = os.path.abspath(root)[len(abs_source):].lstrip(os.sep)
                    target_sub_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_sub_dir, exist_ok=True)
                    dst_path = os.path.join(target_sub_dir, dst_filename)
                else:
                    # 全部平铺到输出根目录
                    dst_path = os.path.join(output_dir, dst_filename)

                shutil.copy2(src_path, dst_path)
                print(f"完成：{src_path}  -->  {dst_path}")


if __name__ == "__main__":
    # ====================== 配置区域（按需修改）======================
    # 1. 多个源文件夹列表
    source_folders = [
        "F:/BGI/Project/PPI-Sifter/scripts/biogrid"
    ]

    # 2. 统一输出目录
    output_dir = "F:/BGI/Project/PPI-Sifter/txt_output"

    # 3. 是否递归遍历子目录：True=递归，False=仅当前目录
    IS_RECURSIVE = True

    # 4. 是否保留原目录层级：True=复刻层级，False=全部平铺
    KEEP_DIR_STRUCTURE = False
    # =================================================================

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已就绪：{output_dir}\n")

    # 批量处理所有源文件夹
    for folder in source_folders:
        if not os.path.isdir(folder):
            print(f"⚠️  无效文件夹，跳过：{folder}")
            continue

        print(f"===== 开始处理源目录：{folder} =====")
        process_files(
            source_folder=folder,
            output_dir=output_dir,
            recursive=IS_RECURSIVE,
            keep_structure=KEEP_DIR_STRUCTURE
        )

    print("\n✅ 所有文件处理完毕！")
