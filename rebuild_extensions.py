#!/usr/bin/env python3
"""
Rebuild script for Surprise Cython extensions with better compatibility.

This script helps rebuild the Cython extensions with proper numpy compatibility
to avoid binary incompatibility issues across different Python versions.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def clean_build_artifacts():
    """Remove existing build artifacts to ensure clean rebuild."""
    print("Cleaning build artifacts...")

    # Directories to clean
    dirs_to_clean = [
        "build",
        "dist",
        "scikit_surprise.egg-info",
        "surprise.egg-info"
    ]

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")

    # Find and remove .so files only in surprise directory
    surprise_dir = Path("surprise")
    if surprise_dir.exists():
        for so_file in surprise_dir.rglob("*.so"):
            so_file.unlink()
            print(f"Removed {so_file}")

    # Find and remove .c files generated from .pyx only in surprise directory
    if surprise_dir.exists():
        pyx_files = list(surprise_dir.rglob("*.pyx"))
        for pyx_file in pyx_files:
            c_file = pyx_file.with_suffix(".c")
            if c_file.exists():
                c_file.unlink()
                print(f"Removed {c_file}")


def rebuild_extensions():
    """Rebuild the Cython extensions."""
    print("\nRebuilding Cython extensions...")

    # Set environment variables for better compatibility
    env = os.environ.copy()
    env["CFLAGS"] = "-O3 -fno-strict-aliasing"
    env["NPY_NO_DEPRECATED_API"] = "NPY_1_7_API_VERSION"

    try:
        # Build in-place for development
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("Extensions rebuilt successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error rebuilding extensions: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def verify_installation():
    """Verify that the rebuilt extensions work correctly."""
    print("\nVerifying installation...")

    try:
        # Test basic imports
        import surprise
        print(f"✓ Basic imports successful (surprise version: {surprise.__version__})")

        # Test a simple algorithm
        from surprise import Dataset, SVD
        from surprise.model_selection import cross_validate

        # Load a small test dataset
        data = Dataset.load_builtin('ml-100k', prompt=False)

        # Test SVD algorithm
        algo = SVD(n_factors=5, n_epochs=5, verbose=False)
        results = cross_validate(algo, data, measures=['RMSE'], cv=2, verbose=False)

        if results['test_rmse'].mean() > 0:
            print("✓ Algorithm execution successful")
            return True
        else:
            print("✗ Algorithm execution failed")
            return False

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    """Main rebuild process."""
    print("Surprise Cython Extensions Rebuild Script")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("Error: setup.py not found. Please run this script from the Surprise root directory.")
        sys.exit(1)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {python_version}")

    # Clean build artifacts
    clean_build_artifacts()

    # Rebuild extensions
    if not rebuild_extensions():
        print("\nFailed to rebuild extensions. Please check the error messages above.")
        sys.exit(1)

    # Verify installation
    if verify_installation():
        print("\n" + "=" * 50)
        print("SUCCESS: Extensions rebuilt and verified successfully!")
        print("The package should now work correctly with your Python version.")
    else:
        print("\n" + "=" * 50)
        print("WARNING: Extensions rebuilt but verification failed.")
        print("There may still be compatibility issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
