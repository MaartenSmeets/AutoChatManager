import os

def concatenate_files(output_file, root_dir, exclude_dirs=None, include_yaml_subfolder="config"):
    """
    Concatenates all Python files in the specified directory recursively into a single file,
    and includes YAML files from any folder named `config`.

    :param output_file: Path to the output file where concatenated content will be written.
    :param root_dir: Root directory to start searching for files.
    :param exclude_dirs: List of directories to exclude.
    :param include_yaml_subfolder: Subfolder name to include YAML files from (e.g., 'config').
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.venv', 'build', 'dist', '.idea', '.vscode']

    script_path = os.path.abspath(__file__)  # Get the absolute path of the current script

    with open(output_file, 'w') as output:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove excluded directories from the traversal
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # Handle Python files
                if filename.endswith('.py'):
                    if os.path.abspath(file_path) == script_path or os.path.getsize(file_path) == 0:
                        continue
                    
                    output.write(f"# File: {file_path}\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            output.write(content + '\n\n')
                    except Exception as e:
                        output.write(f"# Error reading {file_path}: {e}\n\n")
                
                # Handle YAML files from any 'config' folder in the path
                #if filename.endswith(('.yaml', '.yml')) and include_yaml_subfolder in dirpath.split(os.sep):
                #    if os.path.getsize(file_path) > 0:
                #        output.write(f"# YAML File: {file_path}\n")
                #        try:
                #            with open(file_path, 'r', encoding='utf-8') as file:
                #                content = file.read()
                #                output.write(content + '\n\n')
                #        except Exception as e:
                #            output.write(f"# Error reading {file_path}: {e}\n\n")

if __name__ == "__main__":
    output_file = "concatenated_files.py"
    root_dir = os.getcwd()  # Change this to the desired root directory if needed
    concatenate_files(output_file, root_dir)
    print(f"All Python and YAML files have been concatenated into {output_file}.")
