#!/bin/bash
# Puts all SNN project files in a single doc for easy comparison and documentation

output_file="./snn_project_doc.txt"

echo "<documents>" > $output_file

echo "<document index=\"1\">" >> $output_file
echo "<source>directory-structure.txt</source>" >> $output_file
echo "<document_content>" >> $output_file

# You might need to install tree first if you don't have it
# Use `brew install tree` on macOS or `apt-get install tree` on Ubuntu
tree -I 'snn_env|__pycache__|.git|.ipynb_checkpoints|venv|*.pyc|synthetic_checkpoints|.DS_Store' --dirsfirst -a >> $output_file

echo "</document_content>" >> $output_file
echo "</document>" >> $output_file

counter=2

process_file() {
    local file=$1
    echo "Processing: $file"
    echo "<document index=\"$counter\">" >> $output_file
    echo "<source>$file</source>" >> $output_file
    echo "<document_content>" >> $output_file
    
    cat "$file" >> $output_file
    
    echo "</document_content>" >> $output_file
    echo "</document>" >> $output_file
    
    ((counter++))
}

# Process Python files from the main project structure
find . -type f \( -name "*.py" -o -name "*.json" \) -not -path "*/snn_env/*" -not -path "*/__pycache__/*" -not -path "*/venv/*" -not -path "synthetic_checkpoints" | while read file; do
    process_file "$file"
done

# Process any notebook files
find . -type f -name "*.ipynb" -not -path "*/snn_env/*" -not -path "*/.ipynb_checkpoints/*" | while read file; do
    process_file "$file"
done

echo "</documents>" >> $output_file

echo "Documentation has been written to $output_file"