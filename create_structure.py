import os
import re
import shutil

def parse_and_create_structure(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start and end of the tree block
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if "zero-error-system/" in line:
            start_index = i
            break
    
    if start_index == -1:
        print("Could not find start of tree structure")
        return

    # Find end of block
    for i in range(start_index, len(lines)):
        if lines[i].strip() == "```":
            end_index = i
            break
    
    if end_index == -1:
        end_index = len(lines)

    tree_lines = lines[start_index:end_index]
    
    # Stack to keep track of paths: (indentation_level, path_name)
    stack = []
    
    root_indent = -1

    for line in tree_lines:
        # Strip newline
        line = line.rstrip()
        if not line or line.strip() == "│":
            continue

        # Remove comments
        if "#" in line:
            line = line.split("#")[0].rstrip()

        # Find the tree prefix length (indentation)
        # We look for the first character that is NOT a tree char or space
        match = re.search(r'[^\s│├└─]', line)
        if not match:
            continue
        
        # The start index of the name (or icon) determines the indentation level
        indent = match.start()
        
        # Extract the name part
        content = line[indent:]
        
        # Check for folder icon and strip it for the name
        name = content
        if name.startswith("📁"):
            name = name.replace("📁", "", 1).strip()
        
        name = name.strip()
        
        # Determine if it is a directory
        # It is a directory if it ends with / or had the folder icon
        is_dir = content.startswith("📁") or name.endswith("/")
        
        # Clean name
        clean_name = name.rstrip("/")
        
        # If this is the root folder
        if not stack:
            root_indent = indent
            stack.append((indent, ".")) # Map root to current directory
            continue
            
        # Manage stack
        # Pop items that are deeper or same level as current
        while stack and stack[-1][0] >= indent:
            stack.pop()
            
        # Construct full path
        parent_path = stack[-1][1]
        full_path = os.path.join(parent_path, clean_name)
        
        print(f"Creating: {full_path} (Dir: {is_dir})")
        
        if is_dir:
            if os.path.exists(full_path) and not os.path.isdir(full_path):
                print(f"WARNING: {full_path} exists as file, removing to create directory")
                os.remove(full_path)
            os.makedirs(full_path, exist_ok=True)
            # Only add directories to the stack
            stack.append((indent, full_path))
        else:
            if os.path.exists(full_path) and os.path.isdir(full_path):
                 print(f"WARNING: {full_path} exists as directory, skipping file creation")
            else:
                parent_dir = os.path.dirname(full_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                
                # Create empty file if not exists
                if not os.path.exists(full_path):
                    with open(full_path, 'w') as f:
                        pass

if __name__ == "__main__":
    parse_and_create_structure("/home/niel/git/multiAgent/folder_structure.md")
