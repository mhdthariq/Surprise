#!/usr/bin/env python3
"""
Surprise Installation Verification Script

This script verifies that Surprise is properly installed and working
with the numpy compatibility fixes. Run this after installation to
ensure everything is working correctly.
"""

import sys
import traceback


def check_python_version():
    """Check if Python version is supported."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 11:
        print("  ✓ Python version is supported")
        return True
    else:
        print("  ✗ Python version not supported (requires 3.11+)")
        return False


def check_numpy():
    """Check if numpy is properly installed and version is compatible."""
    print("\nChecking numpy...")
    try:
        import numpy as np
        print(f"  Numpy version: {np.__version__}")

        # Check version compatibility
        major, minor = np.__version__.split('.')[:2]
        major, minor = int(major), int(minor)

        if (major == 1 and minor >= 21) or (major >= 2):
            print("  ✓ Numpy version is compatible")
            return True
        else:
            print("  ✗ Numpy version is not compatible (requires 1.21.0+)")
            return False

    except ImportError as e:
        print(f"  ✗ Numpy import failed: {e}")
        return False


def check_surprise_import():
    """Check if surprise can be imported successfully."""
    print("\nChecking Surprise import...")
    try:
        import surprise
        print(f"  Surprise version: {surprise.__version__}")
        print("  ✓ Surprise imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Surprise import failed: {e}")
        print("    This may indicate numpy compatibility issues")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error importing Surprise: {e}")
        return False


def check_cython_extensions():
    """Check if Cython extensions are working properly."""
    print("\nChecking Cython extensions...")
    try:
        # Test similarities module
        from surprise import similarities
        print("  ✓ similarities module loaded")

        # Test prediction algorithms
        from surprise.prediction_algorithms import matrix_factorization
        print("  ✓ matrix_factorization module loaded")

        from surprise.prediction_algorithms import optimize_baselines
        print("  ✓ optimize_baselines module loaded")

        from surprise.prediction_algorithms import slope_one
        print("  ✓ slope_one module loaded")

        from surprise.prediction_algorithms import co_clustering
        print("  ✓ co_clustering module loaded")

        print("  ✓ All Cython extensions loaded successfully")
        return True

    except ImportError as e:
        print(f"  ✗ Cython extension import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error with extensions: {e}")
        return False


def test_basic_functionality():
    """Test basic Surprise functionality."""
    print("\nTesting basic functionality...")
    try:
        from surprise import SVD, Dataset
        from surprise.model_selection import cross_validate

        # Load a small dataset
        print("  Loading dataset...")
        data = Dataset.load_builtin('ml-100k', prompt=False)
        print("  ✓ Dataset loaded")

        # Test algorithm
        print("  Testing SVD algorithm...")
        algo = SVD(n_factors=5, n_epochs=5, verbose=False)
        results = cross_validate(algo, data, measures=['RMSE'], cv=2, verbose=False)

        rmse = results['test_rmse'].mean()
        print(f"  ✓ SVD completed successfully (RMSE: {rmse:.3f})")

        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("Surprise Installation Verification")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Numpy Compatibility", check_numpy),
        ("Surprise Import", check_surprise_import),
        ("Cython Extensions", check_cython_extensions),
        ("Basic Functionality", test_basic_functionality),
    ]

    passed = 0
    total = len(checks)

    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} check failed with unexpected error: {e}")

    print("\n" + "=" * 50)
    print(f"Verification Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 SUCCESS: Surprise is properly installed and working!")
        print("\nYour installation has passed all compatibility checks.")
        print("You can now use Surprise for recommendation systems.")
        return True
    else:
        print("❌ ISSUES DETECTED: Some checks failed")
        print("\nPlease refer to NUMPY_COMPATIBILITY.md for troubleshooting:")
        print("- Try running: python rebuild_extensions.py")
        print("- Or reinstall with: pip install --force-reinstall .")
        print("- Check numpy version: python -c 'import numpy; print(numpy.__version__)'")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
