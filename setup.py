# Build script - setuptools and Cython are available as build dependencies

import numpy as np
from Cython.Build import cythonize  # type: ignore[import-untyped]
from setuptools import Extension, setup  # type: ignore[import-untyped]

"""
Modern setup.py for Surprise with Python 3.11-3.13 compatibility.

This setup.py has been updated to work better with modern Python versions and
includes improved compatibility with numpy 2.x and Cython 3.x.

Key improvements:
- Enhanced numpy compatibility macros
- Better Cython compiler directives
- Improved error handling for different platforms
- Support for newer numpy C API
- Better memory management directives
"""


def get_numpy_version_info():
    """Get numpy version for compatibility checking."""
    try:
        numpy_version = np.__version__
        major, minor = map(int, numpy_version.split(".")[:2])
        return major, minor
    except Exception:
        # Fallback for unexpected version formats
        return 1, 21


def get_compiler_directives():
    """Get optimized compiler directives based on environment."""
    numpy_major, numpy_minor = get_numpy_version_info()

    base_directives = {
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "embedsignature": True,
        "always_allow_keywords": True,
        "cdivision": True,  # Faster division, but be careful with negative numbers
        "overflowcheck": False,  # Disable overflow checking for performance
        "profile": False,  # Disable profiling unless debugging
        "linetrace": False,  # Disable line tracing unless debugging
    }

    # Add numpy-specific directives for numpy 2.x compatibility
    if numpy_major >= 2:
        base_directives.update(
            {
                "np_pythran": False,  # Disable pythran for better compatibility
            }
        )

    return base_directives


def get_define_macros():
    """Get preprocessor macros for numpy compatibility."""
    numpy_major, numpy_minor = get_numpy_version_info()

    # Base macros for numpy compatibility
    macros: list[tuple[str, str | None]] = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("CYTHON_USE_TYPE_SLOTS", None),
        ("CYTHON_FAST_THREAD_STATE", "1"),  # Better threading performance
    ]

    # Additional macros for numpy 2.x
    if numpy_major >= 2:
        macros.extend(
            [
                ("NPY_TARGET_VERSION", "NPY_1_22_API_VERSION"),
                (
                    "CYTHON_LIMITED_API",
                    "1",
                ),  # Enable limited API for better ABI stability
            ]
        )

    # Platform-specific optimizations
    import sys

    if sys.platform.startswith("linux"):
        macros.append(("_GNU_SOURCE", None))
    elif sys.platform == "darwin":
        macros.append(("_DARWIN_C_SOURCE", None))

    return macros


def get_include_dirs():
    """Get include directories with proper numpy headers."""
    include_dirs = [np.get_include()]

    # Add additional include directories if they exist
    import os
    import sys

    # Add Python include directory for better compatibility
    python_include = (
        f"{sys.prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}"
    )
    if os.path.exists(python_include):
        include_dirs.append(python_include)

    return include_dirs


def get_extra_compile_args():
    """Get platform-specific compilation arguments."""
    import sys

    import platform

    args = []

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        args.extend(
            [
                "-O3",  # Maximum optimization
                "-ffast-math",  # Faster math operations
                "-fno-strict-aliasing",  # Prevent strict aliasing issues
                "-Wall",  # Enable warnings
                "-Wno-unused-function",  # Ignore unused function warnings
                "-Wno-unused-variable",  # Ignore unused variable warnings
                "-Wno-unreachable-code",  # Ignore unreachable code warnings (Apple Silicon)
            ]
        )

        # Handle Apple Silicon CPU targeting issues
        if sys.platform == "darwin":
            # Check if we're on Apple Silicon
            if platform.machine() in ["arm64", "aarch64"] or "arm" in platform.machine().lower():
                # Use generic ARM64 optimization instead of specific CPU models
                args.append("-mcpu=apple-a14")  # Safe fallback for Apple Silicon
            else:
                # Intel Mac
                args.append("-march=native")
        else:
            # Linux - use native optimization
            args.append("-march=native")

        # Add OpenMP support if available
        try:
            import subprocess

            result = subprocess.run(
                ["gcc", "-fopenmp", "-E", "-"], input="", text=True, capture_output=True
            )
            if result.returncode == 0:
                args.append("-fopenmp")
        except Exception:
            pass  # OpenMP not available

    elif sys.platform == "win32":
        args.extend(
            [
                "/O2",  # Optimize for speed
                "/fp:fast",  # Fast floating point
                "/GL",  # Whole program optimization
            ]
        )

    return args


def get_extra_link_args():
    """Get platform-specific linking arguments."""
    import sys
    import platform

    args = []

    if sys.platform.startswith("linux") or sys.platform == "darwin":
        args.extend(
            [
                "-O3",
            ]
        )

        # Only add LTO on platforms where it's stable
        if sys.platform.startswith("linux"):
            args.append("-flto")  # Link-time optimization
        elif sys.platform == "darwin":
            # Be more conservative with LTO on macOS, especially Apple Silicon
            if not (platform.machine() in ["arm64", "aarch64"] or "arm" in platform.machine().lower()):
                args.append("-flto")

        # Add OpenMP linking if available
        try:
            import subprocess

            result = subprocess.run(
                ["gcc", "-fopenmp", "-E", "-"], input="", text=True, capture_output=True
            )
            if result.returncode == 0:
                args.append("-fopenmp")
        except Exception:
            pass

    elif sys.platform == "win32":
        args.extend(
            [
                "/LTCG",  # Link-time code generation
            ]
        )

    return args


# Get configuration based on environment
define_macros = get_define_macros()
include_dirs = get_include_dirs()
extra_compile_args = get_extra_compile_args()
extra_link_args = get_extra_link_args()

# Define extensions with improved configuration
extensions = [
    Extension(
        name="surprise.similarities",
        sources=["surprise/similarities.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="surprise.prediction_algorithms.matrix_factorization",
        sources=["surprise/prediction_algorithms/matrix_factorization.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="surprise.prediction_algorithms.optimize_baselines",
        sources=["surprise/prediction_algorithms/optimize_baselines.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="surprise.prediction_algorithms.slope_one",
        sources=["surprise/prediction_algorithms/slope_one.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="surprise.prediction_algorithms.co_clustering",
        sources=["surprise/prediction_algorithms/co_clustering.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# Cythonize with improved directives
extensions = cythonize(
    extensions,
    compiler_directives=get_compiler_directives(),
    # Force rebuild to ensure compatibility
    force=True,
    # Add annotation for debugging (disable in production)
    annotate=False,
    # Parallel compilation if available
    nthreads=0,  # Use all available cores
)

# Enhanced setup call with better error handling
if __name__ == "__main__":
    try:
        setup(ext_modules=extensions)
    except Exception as e:
        import sys

        print(f"Error during setup: {e}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print(
            "1. Ensure you have the latest numpy installed: pip install -U numpy",
            file=sys.stderr,
        )
        print("2. Try rebuilding: python rebuild_extensions.py", file=sys.stderr)
        print("3. Check NUMPY_COMPATIBILITY.md for more solutions", file=sys.stderr)
        sys.exit(1)
