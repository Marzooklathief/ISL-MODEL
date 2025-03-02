import os
import shutil

dataset_path = r"C:\Users\Marzook\Downloads\ISL IMG\train" 


if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset path '{dataset_path}' does not exist!")
    exit()

for img in os.listdir(dataset_path):
    if img.endswith(".jpg") or img.endswith(".png"):  
        class_name = img.split("_")[0]  
        class_folder = os.path.join(dataset_path, class_name)

        
        os.makedirs(class_folder, exist_ok=True)

        
        src_path = os.path.join(dataset_path, img)
        dest_path = os.path.join(class_folder, img)

        
        print(f"🔍 Checking: {src_path} → {dest_path}")

        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dest_path)
                print(f"✅ Moved: {img}")
            except Exception as e:
                print(f"❌ Error moving {img}: {e}")
        else:
            print(f"⚠️ Warning: Source file not found - {src_path}")

print("✅ Images organized successfully!")
