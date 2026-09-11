import os
import shutil

if __name__ == "__main__":
    preprocess_path = ""
    out_path = ""
    os.makedirs(out_path, exist_ok=True)
    for filename in os.listdir(preprocess_path):
        if not os.path.exists(os.path.join(preprocess_path, filename, "pyavi", "speaking")):
            continue
        for segment in os.listdir(os.path.join(preprocess_path, filename, "pyavi", "speaking")):
            shutil.copy(
                os.path.join(preprocess_path, filename, "pyavi", "speaking", segment),
                os.path.join(out_path, filename + "_" + segment)
                )
