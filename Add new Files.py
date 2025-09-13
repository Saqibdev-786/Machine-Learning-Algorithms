import os

# Base directory (adjust if needed)
base_dir = "./Linear Regression"

# Files to create in each existing folder
files_to_create = {
    "Multiple Variables/formula": ["library.py", "README.md", "scratch.py", "usingNumpy.py"],
    "Multiple Variables/Gradient Descent": ["library.py", "README.md", "scratch.py", "usingNumpy.py"],
    "one Variable/formula": ["library.py", "README.md", "scratch.py", "usingNumpy.py"],
    "one Variable/Gradient Descent": ["library.py", "README.md", "scratch.py", "usingNumpy.py"]
}

def create_files(base_dir, files_to_create):
    for relative_path, files in files_to_create.items():
        full_path = os.path.join(base_dir, relative_path)
        
        # If the path ends with '.py' or '.md', treat as file
        if relative_path.endswith(('.py', '.md')):
            if not os.path.exists(full_path):
                with open(full_path, "w") as f:
                    f.write(f"# {os.path.basename(full_path)} placeholder\n")
        else:
            # It's a directory
            os.makedirs(full_path, exist_ok=True)
            for file in files:
                file_path = os.path.join(full_path, file)
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write(f"# {file} placeholder\n")

create_files(base_dir, files_to_create)
print(f"Files generated successfully inside '{base_dir}' as per your structure.")
