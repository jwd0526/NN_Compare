#!/bin/bash
#
# cleanup.sh - Clean up generated files in SNN project
#
# This script removes data, results, visualizations, checkpoints,
# and other temporary files to give you a clean slate.
#
# Usage: ./cleanup.sh [options]
#   --all             Remove everything (data, results, checkpoints, etc.)
#   --clean-slate     Remove EVERYTHING to bring repo to a completely clean state
#   --data            Remove only generated data files
#   --results         Remove only results files
#   --checkpoints     Remove only model checkpoints
#   --visualizations  Remove only visualization files
#   --temp            Remove only temporary files (__pycache__, .pyc)
#   --dry-run         Show what would be deleted without actually deleting
#   --force           Delete without asking for confirmation
#

# Ensure script is executable: chmod +x cleanup.sh

# Set color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}SNN Project Cleanup Script${NC}"
echo "============================="
echo ""

# Default settings
DELETE_DATA=false
DELETE_RESULTS=false
DELETE_CHECKPOINTS=false
DELETE_VISUALIZATIONS=false
DELETE_TEMP=false
CLEAN_SLATE=false
DRY_RUN=false
FORCE=false

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}No options specified. Use --help to see usage.${NC}"
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --all)
            DELETE_DATA=true
            DELETE_RESULTS=true
            DELETE_CHECKPOINTS=true
            DELETE_VISUALIZATIONS=true
            DELETE_TEMP=true
            ;;
        --clean-slate)
            DELETE_DATA=true
            DELETE_RESULTS=true
            DELETE_CHECKPOINTS=true
            DELETE_VISUALIZATIONS=true
            DELETE_TEMP=true
            CLEAN_SLATE=true
            ;;
        --data)
            DELETE_DATA=true
            ;;
        --results)
            DELETE_RESULTS=true
            ;;
        --checkpoints)
            DELETE_CHECKPOINTS=true
            ;;
        --visualizations)
            DELETE_VISUALIZATIONS=true
            ;;
        --temp)
            DELETE_TEMP=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
        --help)
            echo "Usage: ./cleanup.sh [options]"
            echo "  --all             Remove everything (data, results, checkpoints, etc.)"
            echo "  --clean-slate     Remove EVERYTHING to bring repo to a completely clean state"
            echo "  --data            Remove only generated data files"
            echo "  --results         Remove only results files"
            echo "  --checkpoints     Remove only model checkpoints"
            echo "  --visualizations  Remove only visualization files"
            echo "  --temp            Remove only temporary files (__pycache__, .pyc)"
            echo "  --dry-run         Show what would be deleted without actually deleting"
            echo "  --force           Delete without asking for confirmation"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help to see available options."
            exit 1
            ;;
    esac
done

# Function to delete directory with confirmation
delete_directory() {
    local dir=$1
    local name=$2
    
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Found $name directory:${NC} $dir"
        
        # Show some sample contents
        if [ -n "$(ls -A $dir 2>/dev/null)" ]; then
            echo "Sample contents:"
            ls -la $dir | head -n 5
            count=$(find $dir -type f | wc -l)
            echo "... and approximately $count files in total"
        else
            echo "Directory is empty."
        fi
        
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete $name directory: $dir"
        else
            if $FORCE; then
                echo -e "${RED}Deleting $name directory:${NC} $dir"
                rm -rf "$dir"
                echo -e "${GREEN}Deleted successfully!${NC}"
            else
                read -p "Are you sure you want to delete this $name directory? (y/n): " confirm
                if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                    echo -e "${RED}Deleting $name directory:${NC} $dir"
                    rm -rf "$dir"
                    echo -e "${GREEN}Deleted successfully!${NC}"
                else
                    echo -e "${BLUE}Skipping deletion of $name directory.${NC}"
                fi
            fi
        fi
        echo ""
    else
        echo -e "${BLUE}No $name directory found at:${NC} $dir"
        echo ""
    fi
}

# Function to delete files with confirmation
delete_files() {
    local pattern=$1
    local name=$2
    
    files=($(find . -name "$pattern" -type f))
    
    if [ ${#files[@]} -gt 0 ]; then
        echo -e "${YELLOW}Found ${#files[@]} $name files${NC}"
        echo "Sample files:"
        for ((i=0; i<$(min 5 ${#files[@]}); i++)); do
            echo "  ${files[i]}"
        done
        
        if [ ${#files[@]} -gt 5 ]; then
            echo "  ... and $((${#files[@]}-5)) more files"
        fi
        
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete ${#files[@]} $name files"
        else
            if $FORCE; then
                echo -e "${RED}Deleting $name files...${NC}"
                for file in "${files[@]}"; do
                    rm -f "$file"
                done
                echo -e "${GREEN}Deleted ${#files[@]} files successfully!${NC}"
            else
                read -p "Are you sure you want to delete these $name files? (y/n): " confirm
                if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                    echo -e "${RED}Deleting $name files...${NC}"
                    for file in "${files[@]}"; do
                        rm -f "$file"
                    done
                    echo -e "${GREEN}Deleted ${#files[@]} files successfully!${NC}"
                else
                    echo -e "${BLUE}Skipping deletion of $name files.${NC}"
                fi
            fi
        fi
        echo ""
    else
        echo -e "${BLUE}No $name files found matching:${NC} $pattern"
        echo ""
    fi
}

# Helper function to get minimum of two numbers
min() {
    if [ $1 -le $2 ]; then
        echo $1
    else
        echo $2
    fi
}

# Print run mode
if $DRY_RUN; then
    echo -e "${YELLOW}Running in DRY RUN mode. No files will be deleted.${NC}"
    echo ""
fi

# Main deletion operations
if $DELETE_DATA; then
    delete_directory "./data" "data"
    delete_files "*.npz" "data"
fi

if $DELETE_RESULTS; then
    delete_directory "./results" "results"
    delete_files "*_history.json" "results"
    delete_files "*_result.json" "results"
fi

if $DELETE_CHECKPOINTS; then
    delete_directory "./results/checkpoints" "checkpoints"
    delete_files "*.pt" "checkpoint"
fi

if $DELETE_VISUALIZATIONS; then
    delete_directory "./results/visualizations" "visualizations"
    delete_directory "./results/noise_analysis" "noise analysis"
    delete_directory "./results/benchmark" "benchmark"
    delete_files "*.png" "visualization"
fi

if $DELETE_TEMP; then
    temp_files=0
    
    # Clean up __pycache__ directories
    find . -name "__pycache__" -type d | while read dir; do
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete: $dir"
        else
            echo -e "${RED}Deleting:${NC} $dir"
            rm -rf "$dir"
            ((temp_files++))
        fi
    done
    
    # Clean up .pyc files
    find . -name "*.pyc" -type f | while read file; do
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete: $file"
        else
            echo -e "${RED}Deleting:${NC} $file"
            rm -f "$file"
            ((temp_files++))
        fi
    done
    
    # Clean up .DS_Store files
    find . -name ".DS_Store" -type f | while read file; do
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete: $file"
        else
            echo -e "${RED}Deleting:${NC} $file"
            rm -f "$file"
            ((temp_files++))
        fi
    done
    
    # Clean up .ipynb_checkpoints directories
    find . -name ".ipynb_checkpoints" -type d | while read dir; do
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY RUN]${NC} Would delete: $dir"
        else
            echo -e "${RED}Deleting:${NC} $dir"
            rm -rf "$dir"
            ((temp_files++))
        fi
    done
    
    echo -e "${GREEN}Temporary files cleanup complete.${NC}"
    echo ""
fi

# Additional clean slate operations
if $CLEAN_SLATE; then
    echo -e "\n${BLUE}Performing additional clean slate operations...${NC}"
    
    # Remove any generated script output files
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY RUN]${NC} Would remove script output files (*.out, *.log)"
    else
        if find . -name "*.out" -o -name "*.log" | grep -q .; then
            if $FORCE; then
                echo -e "${RED}Removing script output files${NC}"
                find . -name "*.out" -o -name "*.log" -exec rm -f {} \;
            else
                read -p "Are you sure you want to remove script output files? (y/n): " confirm
                if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                    echo -e "${RED}Removing script output files${NC}"
                    find . -name "*.out" -o -name "*.log" -exec rm -f {} \;
                else
                    echo -e "${BLUE}Skipping script output files.${NC}"
                fi
            fi
        else
            echo "No script output files found."
        fi
    fi
    
    # Remove any potential virtual environment caches
    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY RUN]${NC} Would clean pip cache"
    else
        if $FORCE; then
            echo -e "${RED}Cleaning pip cache${NC}"
            if command -v pip3 >/dev/null 2>&1; then
                pip3 cache purge >/dev/null 2>&1 || true
            fi
        else
            read -p "Are you sure you want to clean pip cache? (y/n): " confirm
            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                echo -e "${RED}Cleaning pip cache${NC}"
                if command -v pip3 >/dev/null 2>&1; then
                    pip3 cache purge >/dev/null 2>&1 || true
                fi
            else
                echo -e "${BLUE}Skipping pip cache cleanup.${NC}"
            fi
        fi
    fi
    
    echo -e "${GREEN}Clean slate operations completed.${NC}"
fi

# Summary
echo -e "\n${GREEN}Cleanup completed!${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}This was a dry run. No files were actually deleted.${NC}"
    echo "Run without --dry-run to perform actual deletion."
elif $CLEAN_SLATE; then
    echo -e "${GREEN}Repository has been reset to a clean slate state.${NC}"
fi