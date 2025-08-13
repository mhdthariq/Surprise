#!/usr/bin/env python3
"""
Rebuild script for Surprise Cython extensions with better compatibility.

This script helps rebuild the Cython extensions with proper numpy compatibility
to avoid binary incompatibility issues across different Python versions.

Updated for Python 3.11-3.13 with enhanced error handling and optimization detection.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_artifacts():
    """Remove existing build artifacts to ensure clean rebuild."""
    print("Cleaning build artifacts...")

    # Directories to clean
    dirs_to_clean = ["build", "dist", "scikit_surprise.egg-info", "surprise.egg-info"]

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


def check_build_dependencies():
    """Check if all build dependencies are available."""
    print("Checking build dependencies...")

    missing_deps = []

    try:
        import numpy

        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")

    try:
        import Cython

        cython_version = getattr(Cython, "__version__", "unknown")
        print(f"✓ Cython {cython_version}")
    except ImportError:
        missing_deps.append("Cython")

    try:
        import setuptools

        print(f"✓ setuptools {setuptools.__version__}")
    except ImportError:
        missing_deps.append("setuptools")

    if missing_deps:
        print(f"\n✗ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them with: pip install " + " ".join(missing_deps))
        return False

    return True


def get_platform_specific_flags():
    """Get platform-specific compilation flags."""
    system = platform.system().lower()

    flags = {
        "CFLAGS": "-O3 -fno-strict-aliasing",
        "CXXFLAGS": "-O3 -fno-strict-aliasing",
        "NPY_NO_DEPRECATED_API": "NPY_1_7_API_VERSION",
    }

    if system in ("linux", "darwin"):
        # Unix-like systems
        flags["CFLAGS"] += " -ffast-math -Wall -Wno-unused-function"
        flags["CXXFLAGS"] += " -ffast-math -Wall -Wno-unused-function"

        # Check for OpenMP support
        try:
            result = subprocess.run(
                ["gcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                flags["CFLAGS"] += " -fopenmp"
                flags["CXXFLAGS"] += " -fopenmp"
                flags["LDFLAGS"] = "-fopenmp"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # GCC not available or timeout

    elif system == "windows":
        # Windows-specific flags
        flags["CFLAGS"] = "/O2 /fp:fast"
        flags["CXXFLAGS"] = "/O2 /fp:fast"

    return flags


def rebuild_extensions():
    """Rebuild the Cython extensions with enhanced configuration."""
    print("\nRebuilding Cython extensions...")

    if not check_build_dependencies():
        return False

    # Set environment variables for better compatibility
    env = os.environ.copy()
    platform_flags = get_platform_specific_flags()
    env.update(platform_flags)

    # Print configuration being used
    print("\nUsing compilation flags:")
    for key, value in platform_flags.items():
        print(f"  {key}={value}")

    # Build in-place for development
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]

    try:
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, env=env, check=True, capture_output=True, text=True
        )

        print("Extensions rebuilt successfully!")

        # Show any warnings if present
        if result.stderr and "warning" in result.stderr.lower():
            print("\nCompilation warnings:")
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error rebuilding extensions: {e}")
        print(f"Command: {' '.join(cmd)}")
        if e.stdout:
            print(f"stdout:\n{e.stdout}")
        if e.stderr:
            print(f"stderr:\n{e.stderr}")

        # Provide troubleshooting suggestions
        print("\nTroubleshooting suggestions:")
        if "Microsoft Visual C++" in str(e.stderr):
            print("- Install Microsoft Visual C++ Build Tools for Windows")
        elif "gcc" in str(e.stderr) or "clang" in str(e.stderr):
            print(
                "- Install development tools (build-essential on Ubuntu, Xcode on macOS)"
            )
        elif "numpy" in str(e.stderr):
            print("- Update numpy: pip install -U numpy")

        return False


def verify_installation():
    """Verify that the rebuilt extensions work correctly."""
    print("\nVerifying installation...")

    try:
        # Test basic imports
        print("Testing basic imports...")
        import surprise

        print(f"✓ Basic imports successful (surprise version: {surprise.__version__})")

        # Test Cython extensions specifically
        print("Testing Cython extensions...")

        extensions_to_test = [
            ("surprise.similarities", "similarities"),
            (
                "surprise.prediction_algorithms.matrix_factorization",
                "matrix_factorization",
            ),
            ("surprise.prediction_algorithms.optimize_baselines", "optimize_baselines"),
            ("surprise.prediction_algorithms.slope_one", "slope_one"),
            ("surprise.prediction_algorithms.co_clustering", "co_clustering"),
        ]

        for module_name, display_name in extensions_to_test:
            try:
                __import__(module_name)
                print(f"✓ {display_name} module loaded")
            except ImportError as e:
                print(f"✗ {display_name} module failed: {e}")
                return False

        # Test a simple algorithm
        print("Testing algorithm functionality...")
        from surprise import Dataset, SVD
        from surprise.model_selection import cross_validate

        # Load a small test dataset
        try:
            data = Dataset.load_builtin("ml-100k", prompt=False)
        except Exception:
            # If ml-100k is not available, create a dummy dataset
            import numpy as np
            import pandas as pd

            from surprise import Reader

            # Create minimal test data
            np.random.seed(42)
            users = np.repeat(range(10), 10)
            items = np.tile(range(10), 10)
            ratings = np.random.uniform(1, 5, 100)

            df = pd.DataFrame({"user": users, "item": items, "rating": ratings})
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

        # Test SVD algorithm
        algo = SVD(n_factors=5, n_epochs=5, verbose=False)
        results = cross_validate(algo, data, measures=["RMSE"], cv=2, verbose=False)

        if results["test_rmse"].mean() > 0:
            print("✓ Algorithm execution successful")
            print(f"  Mean RMSE: {results['test_rmse'].mean():.3f}")
            return True
        else:
            print("✗ Algorithm execution failed - unexpected RMSE result")
            return False

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("This usually indicates a compilation issue with Cython extensions.")
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
        print(
            "Error: setup.py not found. Please run this script from the Surprise root directory."
        )
        sys.exit(1)

    # Display system information
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"Python version: {python_version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Architecture: {platform.architecture()[0]}")

    # Check Python version compatibility
    if sys.version_info < (3, 11):
        print(f"Warning: Python {python_version} is not officially supported.")
        print("This package requires Python 3.11 or later.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ("y", "yes"):
            sys.exit(1)
    elif sys.version_info >= (3, 14):
        print(f"Info: Python {python_version} is newer than tested versions.")
        print("This should work but hasn't been extensively tested.")

    # Clean build artifacts
    clean_build_artifacts()

    # Rebuild extensions
    if not rebuild_extensions():
        print("\nFailed to rebuild extensions. Please check the error messages above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install numpy Cython setuptools")
        print("2. Update your compiler/build tools")
        print("3. Try a different Python version (3.11-3.13 recommended)")
        print("4. Check NUMPY_COMPATIBILITY.md for detailed troubleshooting")
        sys.exit(1)

    # Verify installation
    if verify_installation():
        print("\n" + "=" * 50)
        print("SUCCESS: Extensions rebuilt and verified successfully!")
        print("The package should now work correctly with your Python version.")
        print("\nYou can now use Surprise with:")
        print("  import surprise")
        print("  from surprise import SVD, Dataset")
    else:
        print("\n" + "=" * 50)
        print("WARNING: Extensions rebuilt but verification failed.")
        print("The compilation succeeded but there may be runtime issues.")
        print("Check the error messages above for troubleshooting.")
        print("\nYou can still try using the package, but some features may not work.")
        sys.exit(1)


if __name__ == "__main__":
    main()
