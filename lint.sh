#!/bin/bash

# Modern lint script for Surprise with improved error handling and modern tools
# This script runs code formatting, linting, and type checking

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run a command with error handling
run_check() {
    local tool=$1
    local description=$2
    shift 2

    print_step "Running $description..."

    if ! command_exists "$tool"; then
        print_error "$tool not found. Please install it: pip install $tool"
        return 1
    fi

    if "$@"; then
        print_success "$description completed successfully"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Show tool versions for debugging
print_step "Checking tool versions..."
echo "Python version: $(python --version)"

for tool in black usort flake8 ruff mypy; do
    if command_exists "$tool"; then
        echo "✓ $tool: $($tool --version 2>/dev/null | head -n1 || echo 'version unknown')"
    else
        echo "✗ $tool: not installed"
    fi
done

echo ""

# Define directories to check
DIRS_TO_CHECK=("surprise" "tests" "examples" "setup.py" "rebuild_extensions.py" "test_setup.py" "test_integration.py")

# Check if directories exist
EXISTING_DIRS=()
for dir in "${DIRS_TO_CHECK[@]}"; do
    if [[ -e "$dir" ]]; then
        EXISTING_DIRS+=("$dir")
    else
        print_warning "Directory/file '$dir' not found, skipping"
    fi
done

if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
    print_error "No directories or files found to lint"
    exit 1
fi

print_step "Will check: ${EXISTING_DIRS[*]}"
echo ""

# Exit code tracking
EXIT_CODE=0

# Run Ruff (fast modern linter) if available
if command_exists ruff; then
    print_step "Running Ruff (fast modern linter)..."
    if ruff check "${EXISTING_DIRS[@]}" --fix --exit-zero; then
        print_success "Ruff completed successfully"
    else
        print_warning "Ruff found issues (some may have been auto-fixed)"
    fi
    echo ""
fi

# Run import sorting with usort
if ! run_check usort "import sorting (usort)" usort format "${EXISTING_DIRS[@]}"; then
    EXIT_CODE=1
fi
echo ""

# Run code formatting with black
if ! run_check black "code formatting (black)" black "${EXISTING_DIRS[@]}"; then
    EXIT_CODE=1
fi
echo ""

# Run flake8 with configuration from pyproject.toml
print_step "Running flake8 (style checker)..."
if command_exists flake8; then
    # Use configuration from pyproject.toml via command line flags for compatibility
    FLAKE8_ARGS=(
        --max-line-length=88
        --ignore=E203,E231,E241,E402,W503,W504,F821,E501
        --exclude=.git,__pycache__,dist,build,*.egg-info,.venv,venv
        --statistics
        --count
    )

    if flake8 "${FLAKE8_ARGS[@]}" "${EXISTING_DIRS[@]}"; then
        print_success "flake8 completed successfully"
    else
        print_error "flake8 found issues"
        EXIT_CODE=1
    fi
else
    print_error "flake8 not found. Please install it: pip install flake8"
    EXIT_CODE=1
fi
echo ""

# Run mypy type checking on critical files
if command_exists mypy; then
    print_step "Running mypy (type checker) on critical files..."
    MYPY_FILES=()
    for file in setup.py rebuild_extensions.py test_setup.py; do
        if [[ -f "$file" ]]; then
            MYPY_FILES+=("$file")
        fi
    done

    if [[ ${#MYPY_FILES[@]} -gt 0 ]]; then
        if mypy "${MYPY_FILES[@]}" --ignore-missing-imports --no-strict-optional --check-untyped-defs; then
            print_success "mypy type checking completed successfully"
        else
            print_warning "mypy found type issues (non-critical)"
        fi
    else
        print_warning "No critical files found for mypy checking"
    fi
else
    print_warning "mypy not found. Install it for type checking: pip install mypy"
fi
echo ""

# Final summary
echo "========================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "All linting checks passed! 🎉"
    echo ""
    echo "Your code is ready for commit."
else
    print_error "Some linting checks failed! ❌"
    echo ""
    echo "Please fix the issues above before committing."
    echo ""
    echo "Quick fixes:"
    echo "  - Run 'black .' to auto-format code"
    echo "  - Run 'usort format .' to sort imports"
    echo "  - Run 'ruff check . --fix' to auto-fix many issues"
fi
echo "========================================"

exit $EXIT_CODE
