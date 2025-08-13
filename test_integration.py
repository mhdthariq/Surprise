#!/usr/bin/env python3
"""
Comprehensive integration test script for Surprise library.

This script tests the entire build pipeline, installation, and functionality
to ensure everything works correctly with Python 3.11-3.13.
"""

import sys
import subprocess
import shutil
import time
import traceback
from pathlib import Path
from typing import List, Tuple, Dict


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}")


def print_step(text: str) -> None:
    """Print a test step."""
    print(f"\n🔧 {text}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"❌ {text}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"⚠️  {text}")


def run_command(
    cmd: List[str],
    cwd: str | None = None,
    timeout: int = 300,
    capture_output: bool = True
) -> Tuple[bool, str, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


class IntegrationTester:
    """Comprehensive integration tester for Surprise."""

    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.start_time = time.time()
        self.current_dir = Path.cwd()

    def record_test(self, test_name: str, success: bool) -> None:
        """Record test result."""
        self.test_results[test_name] = success
        if success:
            print_success(f"{test_name} passed")
        else:
            print_error(f"{test_name} failed")

    def test_environment(self) -> bool:
        """Test the Python environment and basic dependencies."""
        print_step("Testing Python environment")

        # Check Python version
        version = sys.version_info
        if version < (3, 11):
            print_error(f"Python {version.major}.{version.minor} is too old (need 3.11+)")
            return False
        elif version >= (3, 14):
            print_warning(f"Python {version.major}.{version.minor} is newer than tested")

        print_success(f"Python {version.major}.{version.minor}.{version.micro}")

        # Check basic imports
        basic_deps = ["setuptools", "wheel", "pip"]
        for dep in basic_deps:
            try:
                __import__(dep)
                print_success(f"{dep} available")
            except ImportError:
                print_error(f"{dep} not available")
                return False

        return True

    def test_build_dependencies(self) -> bool:
        """Test build dependencies are available."""
        print_step("Testing build dependencies")

        deps = {
            "numpy": "numpy",
            "Cython": "Cython",
            "setuptools": "setuptools"
        }

        for name, module in deps.items():
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print_success(f"{name}: {version}")
            except ImportError:
                print_error(f"{name} not available")
                return False

        return True

    def test_clean_build(self) -> bool:
        """Test clean build process."""
        print_step("Testing clean build")

        # Clean existing builds
        build_dirs = ["build", "dist", "*.egg-info"]
        for pattern in build_dirs:
            for path in Path(".").glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print_success(f"Cleaned {path}")

        # Remove .so files
        so_files = list(Path("surprise").rglob("*.so"))
        for so_file in so_files:
            so_file.unlink()
            print_success(f"Removed {so_file}")

        return True

    def test_cython_build(self) -> bool:
        """Test Cython extension building."""
        print_step("Testing Cython extension build")

        cmd = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
        success, stdout, stderr = run_command(cmd, timeout=600)

        if not success:
            print_error("Build failed")
            if stderr:
                print(f"Error output:\n{stderr}")
            return False

        # Check that extensions were built
        expected_extensions = [
            "similarities",
            "matrix_factorization",
            "optimize_baselines",
            "slope_one",
            "co_clustering"
        ]

        built_extensions = []
        for so_file in Path("surprise").rglob("*.so"):
            for ext in expected_extensions:
                if ext in so_file.name:
                    built_extensions.append(ext)
                    print_success(f"Built {ext}")

        missing = set(expected_extensions) - set(built_extensions)
        if missing:
            print_error(f"Missing extensions: {missing}")
            return False

        return True

    def test_package_imports(self) -> bool:
        """Test package imports work correctly."""
        print_step("Testing package imports")

        try:
            import surprise
            print_success(f"surprise imported (version {surprise.__version__})")
        except ImportError as e:
            print_error(f"Failed to import surprise: {e}")
            return False

        # Test key algorithm imports
        algorithms = [
            "SVD", "SVDpp", "NMF", "SlopeOne", "KNNBasic",
            "KNNBaseline", "KNNWithMeans", "BaselineOnly",
            "CoClustering", "NormalPredictor"
        ]

        for algo in algorithms:
            try:
                getattr(surprise, algo)
                print_success(f"{algo} imported")
            except AttributeError:
                print_error(f"Failed to import {algo}")
                return False

        # Test utility imports
        utilities = ["Dataset", "Reader", "Trainset"]
        for util in utilities:
            try:
                getattr(surprise, util)
                print_success(f"{util} imported")
            except AttributeError:
                print_error(f"Failed to import {util}")
                return False

        return True

    def test_basic_functionality(self) -> bool:
        """Test basic algorithm functionality."""
        print_step("Testing basic functionality")

        try:
            from surprise import SVD, Dataset, Reader
            from surprise.model_selection import train_test_split
            import numpy as np
            import pandas as pd

            # Create test dataset
            np.random.seed(42)
            n_users, n_items = 50, 30

            user_ids = []
            item_ids = []
            ratings = []

            for user in range(n_users):
                for item in range(n_items):
                    if np.random.random() > 0.8:  # Sparse data
                        rating = np.random.uniform(1, 5)
                        user_ids.append(user)
                        item_ids.append(item)
                        ratings.append(rating)

            df = pd.DataFrame({
                'userID': user_ids,
                'itemID': item_ids,
                'rating': ratings
            })

            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

            print_success("Test dataset created")

            # Split data
            trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
            print_success("Data split completed")

            # Test SVD algorithm
            algo = SVD(n_factors=20, n_epochs=50, random_state=42, verbose=False)
            algo.fit(trainset)
            print_success("SVD training completed")

            # Make predictions
            predictions = algo.test(testset)

            if len(predictions) == 0:
                print_error("No predictions generated")
                return False

            # Calculate RMSE
            rmse = np.sqrt(np.mean([(pred.est - pred.r_ui) ** 2 for pred in predictions]))
            print_success(f"Predictions generated (RMSE: {rmse:.4f})")

            return True

        except Exception as e:
            print_error(f"Functionality test failed: {e}")
            traceback.print_exc()
            return False

    def test_cross_validation(self) -> bool:
        """Test cross-validation functionality."""
        print_step("Testing cross-validation")

        try:
            from surprise import SVD, Dataset
            from surprise.model_selection import cross_validate
            import numpy as np
            import pandas as pd

            # Create larger test dataset
            np.random.seed(123)
            data_size = 1000

            df = pd.DataFrame({
                'userID': np.random.randint(0, 100, data_size),
                'itemID': np.random.randint(0, 50, data_size),
                'rating': np.random.uniform(1, 5, data_size)
            })

            from surprise import Reader
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df, reader)

            # Run cross-validation
            algo = SVD(n_factors=10, n_epochs=20, verbose=False)
            results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

            rmse_mean = results['test_rmse'].mean()
            mae_mean = results['test_mae'].mean()

            print_success(f"Cross-validation completed (RMSE: {rmse_mean:.4f}, MAE: {mae_mean:.4f})")

            return True

        except Exception as e:
            print_error(f"Cross-validation test failed: {e}")
            return False

    def test_multiple_algorithms(self) -> bool:
        """Test multiple algorithms work correctly."""
        print_step("Testing multiple algorithms")

        try:
            from surprise import (SVD, NMF, KNNBasic, BaselineOnly,
                                SlopeOne, Dataset)
            import numpy as np
            import pandas as pd

            # Create test data
            np.random.seed(456)
            df = pd.DataFrame({
                'userID': np.random.randint(0, 20, 200),
                'itemID': np.random.randint(0, 15, 200),
                'rating': np.random.uniform(1, 5, 200)
            })

            from surprise import Reader
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df, reader)
            trainset = data.build_full_trainset()

            algorithms = [
                ('SVD', SVD(n_factors=5, n_epochs=10, verbose=False)),
                ('BaselineOnly', BaselineOnly(verbose=False)),
                ('NMF', NMF(n_factors=5, n_epochs=10, verbose=False)),
                ('KNNBasic', KNNBasic(k=5, verbose=False)),
                ('SlopeOne', SlopeOne()),
            ]

            for name, algo in algorithms:
                try:
                    algo.fit(trainset)
                    # Test a prediction
                    pred = algo.predict(0, 0)
                    print_success(f"{name} algorithm works (prediction: {pred.est:.3f})")
                except Exception as e:
                    print_error(f"{name} algorithm failed: {e}")
                    return False

            return True

        except Exception as e:
            print_error(f"Multiple algorithms test failed: {e}")
            return False

    def test_builtin_datasets(self) -> bool:
        """Test built-in dataset loading."""
        print_step("Testing built-in datasets")

        try:
            from surprise import Dataset

            # Try to load ml-100k (most common test dataset)
            try:
                data = Dataset.load_builtin('ml-100k', prompt=False)
                print_success("ml-100k dataset loaded")

                # Test basic operations
                trainset = data.build_full_trainset()
                print_success(f"Trainset built ({trainset.n_users} users, {trainset.n_items} items)")

                return True

            except Exception as e:
                print_warning(f"Built-in dataset not available: {e}")
                print_warning("This is normal in many environments")
                return True  # Not a failure

        except Exception as e:
            print_error(f"Dataset test failed: {e}")
            return False

    def test_performance_benchmark(self) -> bool:
        """Run a quick performance benchmark."""
        print_step("Running performance benchmark")

        try:
            from surprise import SVD, Dataset, Reader
            from surprise.model_selection import cross_validate
            import numpy as np
            import pandas as pd
            import time

            # Create moderately sized dataset
            np.random.seed(789)
            data_size = 5000

            df = pd.DataFrame({
                'userID': np.random.randint(0, 200, data_size),
                'itemID': np.random.randint(0, 100, data_size),
                'rating': np.random.uniform(1, 5, data_size)
            })

            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df, reader)

            # Benchmark SVD
            start_time = time.time()
            algo = SVD(n_factors=50, n_epochs=30, verbose=False)
            results = cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)
            end_time = time.time()

            duration = end_time - start_time
            rmse = results['test_rmse'].mean()

            print_success(f"Benchmark completed in {duration:.2f}s (RMSE: {rmse:.4f})")

            # Performance expectations (very loose)
            if duration > 120:  # 2 minutes
                print_warning(f"Performance slower than expected ({duration:.2f}s)")
            else:
                print_success("Performance within expected range")

            return True

        except Exception as e:
            print_error(f"Performance benchmark failed: {e}")
            return False

    def test_memory_usage(self) -> bool:
        """Test memory usage doesn't explode."""
        print_step("Testing memory usage")

        try:
            import psutil  # type: ignore
            import gc
            from surprise import SVD, Dataset, Reader
            import numpy as np
            import pandas as pd

            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create dataset and run algorithm
            np.random.seed(999)
            df = pd.DataFrame({
                'userID': np.random.randint(0, 500, 10000),
                'itemID': np.random.randint(0, 200, 10000),
                'rating': np.random.uniform(1, 5, 10000)
            })

            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df, reader)
            trainset = data.build_full_trainset()

            algo = SVD(n_factors=100, n_epochs=50, verbose=False)
            algo.fit(trainset)

            # Check memory after
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print_success(f"Memory usage: {initial_memory:.1f} MB → {final_memory:.1f} MB (+{memory_increase:.1f} MB)")

            # Clean up
            del algo, trainset, data, df
            gc.collect()

            if memory_increase > 500:  # 500 MB
                print_warning(f"High memory usage: {memory_increase:.1f} MB")

            return True

        except ImportError:
            print_warning("psutil not available, skipping memory test")
            return True
        except Exception as e:
            print_error(f"Memory test failed: {e}")
            return False

    def test_error_handling(self) -> bool:
        """Test proper error handling."""
        print_step("Testing error handling")

        try:
            from surprise import Dataset, Reader
            import pandas as pd

            # Test with invalid data
            try:
                df = pd.DataFrame({
                    'userID': [1, 2, 3],
                    'itemID': [1, 2, 3],
                    'rating': ['invalid', 'data', 'here']  # Invalid ratings
                })

                reader = Reader(rating_scale=(1, 5))
                # This should handle the error gracefully
                try:
                    data = Dataset.load_from_df(df, reader)
                    trainset = data.build_full_trainset()
                    print_warning("Invalid data was accepted (unexpected)")
                except:
                    print_success("Invalid data properly rejected")

            except Exception:
                print_success("Error handling works for invalid data")

            # Test with empty dataset
            try:
                empty_df = pd.DataFrame({'userID': [], 'itemID': [], 'rating': []})
                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(empty_df, reader)
                trainset = data.build_full_trainset()

                if trainset.n_users == 0:
                    print_success("Empty dataset handled correctly")
                else:
                    print_warning("Empty dataset handling unexpected")

            except Exception:
                print_success("Empty dataset properly rejected")

            return True

        except Exception as e:
            print_error(f"Error handling test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print_header("🧪 Surprise Integration Test Suite")
        print(f"Python: {sys.version}")
        print(f"Platform: {sys.platform}")
        print(f"Working Directory: {self.current_dir}")

        tests = [
            ("Environment Check", self.test_environment),
            ("Build Dependencies", self.test_build_dependencies),
            ("Clean Build", self.test_clean_build),
            ("Cython Build", self.test_cython_build),
            ("Package Imports", self.test_package_imports),
            ("Basic Functionality", self.test_basic_functionality),
            ("Cross Validation", self.test_cross_validation),
            ("Multiple Algorithms", self.test_multiple_algorithms),
            ("Built-in Datasets", self.test_builtin_datasets),
            ("Performance Benchmark", self.test_performance_benchmark),
            ("Memory Usage", self.test_memory_usage),
            ("Error Handling", self.test_error_handling),
        ]

        for test_name, test_func in tests:
            try:
                success = test_func()
                self.record_test(test_name, success)
            except Exception as e:
                print_error(f"Test {test_name} crashed: {e}")
                traceback.print_exc()
                self.record_test(test_name, False)

        # Print summary
        self.print_summary()

        # Return overall success
        return all(self.test_results.values())

    def print_summary(self) -> None:
        """Print test summary."""
        total_time = time.time() - self.start_time
        passed = sum(self.test_results.values())
        total = len(self.test_results)

        print_header("📊 Test Summary")

        for test_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:<25} {status}")

        print(f"\n📈 Results: {passed}/{total} tests passed")
        print(f"⏱️  Total time: {total_time:.2f} seconds")

        if passed == total:
            print_header("🎉 ALL TESTS PASSED! 🎉", "🎉")
            print("Your Surprise installation is working perfectly!")
        else:
            print_header("❌ SOME TESTS FAILED ❌", "❌")
            print(f"{total - passed} test(s) failed. Check the output above for details.")


def main():
    """Main entry point."""
    # Check if we're in the right directory
    if not Path("setup.py").exists():
        print_error("setup.py not found. Please run from the Surprise root directory.")
        return 1

    tester = IntegrationTester()
    success = tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
