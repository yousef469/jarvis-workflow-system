import os

def replace_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        new_content = content
        new_content = new_content.replace('Jarvis', 'Jarvis')
        new_content = new_content.replace('jarvis', 'jarvis')
        new_content = new_content.replace('JARVIS', 'JARVIS')
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated content in: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    exclude_dirs = {'.git', 'node_modules', 'dist', 'dist-electron', 'jarvis_env_mac', '__pycache__', 'generated_images', 'generated_models'}
    
    # 1. Replace content
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(('.py', '.md', '.txt', '.json', '.html', '.css', '.tsx', '.ts', '.js', '.jsx')):
                replace_in_file(file_path)
    
    # 2. Rename files and directories
    for root, dirs, files in sorted(os.walk('.', topdown=False), reverse=True):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Rename files
        for file in files:
            if 'Jarvis' in file or 'jarvis' in file or 'JARVIS' in file:
                old_path = os.path.join(root, file)
                new_file = file.replace('Jarvis', 'Jarvis').replace('jarvis', 'jarvis').replace('JARVIS', 'JARVIS')
                new_path = os.path.join(root, new_file)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")
        
        # Rename directories
        for d in dirs:
            if 'Jarvis' in d or 'jarvis' in d or 'JARVIS' in d:
                old_path = os.path.join(root, d)
                new_d = d.replace('Jarvis', 'Jarvis').replace('jarvis', 'jarvis').replace('JARVIS', 'JARVIS')
                new_path = os.path.join(root, new_d)
                os.rename(old_path, new_path)
                print(f"Renamed dir: {old_path} -> {new_path}")

if __name__ == "__main__":
    main()
