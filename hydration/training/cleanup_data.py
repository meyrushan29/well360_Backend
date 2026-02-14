import shutil
import os

source_dir = r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1\data"
backup_base = r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1\temp_data"

if not os.path.exists(backup_base):
    os.makedirs(backup_base)

dirs_to_move = ["fixed_images", "original_backup"]

for d in dirs_to_move:
    src = os.path.join(source_dir, d)
    dst = os.path.join(backup_base, d)
    
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    else:
        print(f"{src} does not exist.")
