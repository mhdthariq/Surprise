#!/usr/bin/env python3
"""
Test script to verify that the updated setup.py works correctly
with Python 3.11-3.13 and modern numpy/Cython versions.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, cwd=None, timeout=300):
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_import_dependencies():
    """Test that all build dependencies can be imported."""
    print("Testing build dependencies...")

    dependencies = [
        ("numpy", "np"),
        ("Cython", "Cython"),
        ("setuptools", "setuptools"),
    ]

    for dep_name, import_name in dependencies:
        try:
            if import_name == "np":
                import numpy as np

                version = np.__version__
            elif import_name == "Cython":
                import Cython

                version = getattr(Cython, "__version__", "unknown")
            elif import_name == "setuptools":
                import setuptools

                version = setuptools.__version__
            else:
                version = "unknown"
            print(f"✓ {dep_name}: {version}")
        except ImportError:
            print(f"✗ {dep_name}: Not available")
            return False
        except Exception as e:
            print(f"✗ {dep_name}: Error - {e}")
            return False

    return True


def test_cython_compilation():
    """Test that Cython can compile a simple extension."""
    print("\nTesting Cython compilation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple Cython file
        pyx_content = """
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def simple_function(cnp.ndarray[cnp.float64_t, ndim=1] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.zeros(n)

    for i in range(n):
        result[i] = sqrt(arr[i] * arr[i])

    return result
"""

        setup_content = """
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "test_module",
        ["test_module.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3, force=True)
)
"""

        pyx_file = Path(temp_dir) / "test_module.pyx"
        setup_file = Path(temp_dir) / "setup.py"

        pyx_file.write_text(pyx_content)
        setup_file.write_text(setup_content)

        # Try to build
        success, stdout, stderr = run_command(
            [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=temp_dir
        )

        if success:
            print("✓ Cython compilation successful")
            return True
        else:
            print(f"✗ Cython compilation failed: {stderr}")
            return False


def test_surprise_build():
    """Test building Surprise extensions."""
    print("\nTesting Surprise extension build...")

    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print("✗ setup.py not found. Please run from Surprise root directory.")
        return False

    # Clean previous builds
    build_dirs = ["build", "dist", "*.egg-info"]
    for pattern in build_dirs:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)

    # Remove compiled extensions
    for so_file in Path("surprise").rglob("*.so"):
        so_file.unlink()

    # Try to build
    success, stdout, stderr = run_command(
        [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
    )

    if success:
        print("✓ Surprise extensions built successfully")

        # Check that .so files were created
        so_files = list(Path("surprise").rglob("*.so"))
        expected_extensions = [
            "similarities",
            "matrix_factorization",
            "optimize_baselines",
            "slope_one",
            "co_clustering",
        ]

        found_extensions = []
        for so_file in so_files:
            for ext in expected_extensions:
                if ext in so_file.name:
                    found_extensions.append(ext)

        missing = set(expected_extensions) - set(found_extensions)
        if missing:
            print(f"✗ Missing extensions: {missing}")
            return False
        else:
            print(f"✓ All {len(expected_extensions)} extensions compiled")
            return True
    else:
        print(f"✗ Surprise build failed: {stderr}")
        return False


def test_surprise_import():
    """Test importing Surprise after build."""
    print("\nTesting Surprise imports...")

    try:
        import surprise

        print(f"✓ Basic import successful (version: {surprise.__version__})")

        # Test specific extensions
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

        for module_name, ext_name in extensions_to_test:
            try:
                __import__(module_name)
                print(f"✓ {ext_name} extension imported")
            except ImportError as e:
                print(f"✗ {ext_name} extension failed: {e}")
                return False

        return True

    except ImportError as e:
        print(f"✗ Surprise import failed: {e}")
        return False


def test_surprise_functionality():
    """Test basic Surprise functionality."""
    print("\nTesting Surprise functionality...")

    try:
        import numpy as np
        import pandas as pd

        from surprise import Dataset, Reader, SVD
        from surprise.model_selection import train_test_split

        # Create test data
        np.random.seed(42)
        n_users, n_items = 20, 15
        user_ids = []
        item_ids = []
        ratings = []

        for user in range(n_users):
            for item in range(n_items):
                if np.random.random() > 0.7:  # Sparse data
                    rating = np.random.uniform(1, 5)
                    user_ids.append(user)
                    item_ids.append(item)
                    ratings.append(rating)

        df = pd.DataFrame({"userID": user_ids, "itemID": item_ids, "rating": ratings})
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

        # Split data
        trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

        # Test SVD algorithm
        algo = SVD(n_factors=10, n_epochs=20, random_state=42, verbose=False)
        algo.fit(trainset)

        # Make predictions
        predictions = algo.test(testset)

        if len(predictions) > 0:
            rmse = np.sqrt(
                np.mean([(pred.est - pred.r_ui) ** 2 for pred in predictions])
            )
            print(f"✓ Algorithm test successful (RMSE: {rmse:.3f})")
            return True
        else:
            print("✗ No predictions generated")
            return False

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Surprise Setup Test Suite")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    tests = [
        ("Import Dependencies", test_import_dependencies),
        ("Cython Compilation", test_cython_compilation),
        ("Surprise Build", test_surprise_build),
        ("Surprise Import", test_surprise_import),
        ("Surprise Functionality", test_surprise_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is working correctly.")
        return 0
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Check the output above.")
        print("\nTroubleshooting tips:")
        print(
            "1. Ensure all dependencies are installed: pip install numpy Cython setuptools"
        )
        print("2. Try rebuilding: python rebuild_extensions.py")
        print("3. Check for compiler issues (install build tools)")
        print("4. Consult NUMPY_COMPATIBILITY.md for detailed solutions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
