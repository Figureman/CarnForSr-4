import os  
from PIL import Image  
# 将一张图片切分成四个小块，并保存起来。
def split_image(image_path, output_dir, file_prefix):  

    img = Image.open(image_path)  
    width, height = img.size  
    # 计算每个小块的大小  
    half_width = width // 2  
    half_height = height // 2  
      
    # 切分图片并保存  
    for i in range(2):  
        for j in range(2):  
            box = (j * half_width, i * half_height, (j + 1) * half_width, (i + 1) * half_height)  
            piece = img.crop(box)  
              
            # 生成输出文件名  
            piece_index = i * 2 + j  
            filename_parts = os.path.splitext(os.path.basename(image_path))  
            output_filename = f"{filename_parts[0]}_{file_prefix}{piece_index}{filename_parts[1]}"  
              
            # 构造输出文件的完整路径  
            output_file_path = os.path.join(output_dir, output_filename)  
              
            # 保存图像块  
            piece.save(output_file_path)  
            print(f"Saved {output_file_path}")  
  
#  将文件夹中的所有图片切分成四个小块，并保存起来
def split_images_in_directory(input_dir, output_dir, file_prefix='piece'):  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
      
    # 遍历输入目录中的所有文件  
    for filename in os.listdir(input_dir):  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  
            # 构造完整的文件路径  
            image_path = os.path.join(input_dir, filename)  
            # 调用切分函数  
            try:  
                split_image(image_path, output_dir, file_prefix)  
            except ValueError as e:  
                print(f"Error splitting {image_path}: {e}")  
  
# 修改下述代码 来扩展数据集
input_folder = 'F:\\BiSheCoding\\NewRemember\\SR\\ClassSR\\demo_images'  # 输入文件夹路径
output_folder = 'F:\\BiSheCoding\\NewRemember\\SR\\test'  # 输出文件夹路径
split_images_in_directory(input_folder, output_folder, file_prefix='')  
  
