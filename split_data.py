import splitfolders
import os


input_folder = r"C:\Documents\Dataset\Original\MLprojectData"


output_folder = r"C:\Documents\Dataset\Original\TomatoPepper_Final_Split"

print(f"🔄 Starting Data Split...")
print(f"   Input:  {input_folder}")
print(f"   Output: {output_folder}")


try:
    splitfolders.ratio(
        input_folder, 
        output=output_folder, 
        seed=42, 
        ratio=(0.7, 0.15, 0.15), 
        group_prefix=None, 
        move=False 
    )
    print("\n Success! Data split into 'train', 'val', and 'test'.")
    print(f"   Check the folder: {output_folder}")

except Exception as e:
    print(f"\n Error during splitting: {e}")
    print("   Make sure the input path exists and isn't empty.")