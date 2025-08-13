#!/usr/bin/env python3
"""
Workflow status verification script for Surprise library.

This script verifies that all GitHub workflows are properly configured
and that the test files are correctly integrated.
"""

import sys
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:
    yaml = None


class WorkflowVerifier:
    """Verifies GitHub workflow configuration and test integration."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.workflows_dir = self.project_root / ".github" / "workflows"
        self.test_files = ["test_setup.py", "test_integration.py"]
        self.workflow_files = []
        self.issues = []

    def print_header(self, text: str) -> None:
        """Print formatted header."""
        print(f"\n{'=' * 60}")
        print(f" {text}")
        print(f"{'=' * 60}")

    def print_success(self, text: str) -> None:
        """Print success message."""
        print(f"✅ {text}")

    def print_warning(self, text: str) -> None:
        """Print warning message."""
        print(f"⚠️  {text}")

    def print_error(self, text: str) -> None:
        """Print error message."""
        print(f"❌ {text}")

    def check_test_files_exist(self) -> bool:
        """Check that test files exist and are executable."""
        print("\n🔍 Checking test files...")

        all_exist = True
        for test_file in self.test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                self.print_success(f"{test_file} exists")

                # Check if file is executable (has main function)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'if __name__ == "__main__"' in content:
                            self.print_success(f"{test_file} is executable")
                        else:
                            self.print_warning(f"{test_file} may not be executable")
                except Exception as e:
                    self.print_error(f"Error reading {test_file}: {e}")
                    all_exist = False
            else:
                self.print_error(f"{test_file} not found")
                all_exist = False

        return all_exist

    def load_workflow_files(self) -> bool:
        """Load all workflow files."""
        print("\n🔍 Loading workflow files...")

        if not self.workflows_dir.exists():
            self.print_error(f"Workflows directory not found: {self.workflows_dir}")
            return False

        if yaml is None:
            self.print_error("PyYAML not available. Install with: pip install PyYAML")
            return False

        yaml_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))

        for file_path in yaml_files:
            try:
                with open(file_path, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                    self.workflow_files.append({
                        'name': file_path.name,
                        'path': file_path,
                        'data': workflow_data
                    })
                self.print_success(f"Loaded {file_path.name}")
            except Exception as e:
                self.print_error(f"Error loading {file_path.name}: {e}")
                return False

        return len(self.workflow_files) > 0

    def check_workflow_structure(self) -> bool:
        """Check basic workflow structure."""
        print("\n🔍 Checking workflow structure...")

        all_valid = True
        required_workflows = {
            'test.yml': 'Comprehensive testing',
            'lint.yml': 'Code quality',
            'benchmarks.yml': 'Performance testing',
            'build_sdist.yml': 'Package building'
        }

        found_workflows = set(wf['name'] for wf in self.workflow_files)

        for required, description in required_workflows.items():
            if required in found_workflows:
                self.print_success(f"{required} ({description}) present")
            else:
                self.print_error(f"{required} ({description}) missing")
                all_valid = False

        return all_valid

    def check_test_integration(self) -> bool:
        """Check that test files are properly integrated in workflows."""
        print("\n🔍 Checking test file integration in workflows...")

        test_usage = {test_file: [] for test_file in self.test_files}

        for workflow in self.workflow_files:
            workflow_content = str(workflow['data'])

            for test_file in self.test_files:
                if test_file in workflow_content:
                    test_usage[test_file].append(workflow['name'])

        all_integrated = True
        for test_file, workflows in test_usage.items():
            if workflows:
                self.print_success(f"{test_file} used in: {', '.join(workflows)}")
            else:
                self.print_warning(f"{test_file} not used in any workflow")
                # This is not an error since test_integration.py might be optional

        return all_integrated

    def check_python_versions(self) -> bool:
        """Check Python version matrix in workflows."""
        print("\n🔍 Checking Python version support...")

        expected_versions = set(["3.11", "3.12", "3.13"])
        all_correct = True

        for workflow in self.workflow_files:
            workflow_name = workflow['name']
            workflow_data = workflow['data']

            # Look for Python version matrices
            python_versions: set = set()

            def extract_python_versions(obj):
                if isinstance(obj, dict):
                    if 'python-version' in obj:
                        versions = obj['python-version']
                        if isinstance(versions, list):
                            python_versions.update(str(v).strip('"\'') for v in versions)
                        else:
                            python_versions.add(str(versions).strip('"\''))
                    for value in obj.values():
                        extract_python_versions(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_python_versions(item)

            extract_python_versions(workflow_data)

            if python_versions:
                missing = expected_versions - python_versions
                extra = python_versions - expected_versions

                if not missing and not extra:
                    self.print_success(f"{workflow_name}: Python versions correct")
                else:
                    if missing:
                        self.print_warning(f"{workflow_name}: Missing Python versions: {missing}")
                    if extra:
                        self.print_warning(f"{workflow_name}: Extra Python versions: {extra}")
                    all_correct = False
            else:
                self.print_warning(f"{workflow_name}: No Python version matrix found")

        return all_correct

    def check_workflow_triggers(self) -> bool:
        """Check that workflows have appropriate triggers."""
        print("\n🔍 Checking workflow triggers...")

        all_correct = True
        expected_triggers = set(['push', 'pull_request'])

        for workflow in self.workflow_files:
            workflow_name = workflow['name']
            workflow_data = workflow['data']

            if 'on' in workflow_data:
                triggers: set = set()
                on_config = workflow_data['on']

                if isinstance(on_config, dict):
                    triggers.update(on_config.keys())
                elif isinstance(on_config, list):
                    triggers.update(on_config)
                elif isinstance(on_config, str):
                    triggers.add(on_config)

                missing_triggers = expected_triggers - triggers

                if not missing_triggers:
                    self.print_success(f"{workflow_name}: Has required triggers")
                else:
                    self.print_warning(f"{workflow_name}: Missing triggers: {missing_triggers}")
            else:
                self.print_error(f"{workflow_name}: No triggers defined")
                all_correct = False

        return all_correct

    def check_caching_configuration(self) -> bool:
        """Check that workflows use proper caching."""
        print("\n🔍 Checking caching configuration...")

        cache_found = False

        for workflow in self.workflow_files:
            workflow_content = str(workflow['data'])

            if 'actions/cache@v4' in workflow_content or 'cache' in workflow_content.lower():
                self.print_success(f"{workflow['name']}: Uses caching")
                cache_found = True
            else:
                self.print_warning(f"{workflow['name']}: No caching configured")

        return cache_found

    def check_artifact_handling(self) -> bool:
        """Check artifact upload/download configuration."""
        print("\n🔍 Checking artifact handling...")

        artifact_workflows = []

        for workflow in self.workflow_files:
            workflow_content = str(workflow['data'])

            if 'upload-artifact' in workflow_content or 'download-artifact' in workflow_content:
                artifact_workflows.append(workflow['name'])

        if artifact_workflows:
            self.print_success(f"Artifacts configured in: {', '.join(artifact_workflows)}")
            return True
        else:
            self.print_warning("No artifact handling found in workflows")
            return False

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for workflow improvements."""
        recommendations = []

        # Check if test_integration.py is used
        integration_used = any('test_integration.py' in str(wf['data']) for wf in self.workflow_files)
        if not integration_used:
            recommendations.append(
                "Consider adding test_integration.py to workflows for comprehensive testing"
            )

        # Check for performance testing
        perf_testing = any('benchmark' in wf['name'].lower() or 'performance' in str(wf['data']).lower()
                          for wf in self.workflow_files)
        if not perf_testing:
            recommendations.append(
                "Consider adding performance regression testing to workflows"
            )

        # Check for cross-platform testing
        cross_platform = any('matrix' in str(wf['data']) and 'os' in str(wf['data'])
                            for wf in self.workflow_files)
        if not cross_platform:
            recommendations.append(
                "Consider adding cross-platform testing (Linux, macOS, Windows)"
            )

        return recommendations

    def verify_all(self) -> bool:
        """Run all verification checks."""
        self.print_header("GitHub Workflows Verification")
        print(f"Project root: {self.project_root}")

        checks = [
            ("Test Files", self.check_test_files_exist),
            ("Workflow Loading", self.load_workflow_files),
            ("Workflow Structure", self.check_workflow_structure),
            ("Test Integration", self.check_test_integration),
            ("Python Versions", self.check_python_versions),
            ("Workflow Triggers", self.check_workflow_triggers),
            ("Caching Config", self.check_caching_configuration),
            ("Artifact Handling", self.check_artifact_handling),
        ]

        results = {}
        for check_name, check_func in checks:
            try:
                results[check_name] = check_func()
            except Exception as e:
                self.print_error(f"Error in {check_name}: {e}")
                results[check_name] = False

        # Generate summary
        self.print_header("Verification Summary")

        passed = 0
        total = len(results)

        for check_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{check_name:<20} {status}")
            if success:
                passed += 1

        print(f"\nResults: {passed}/{total} checks passed")

        # Generate recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            self.print_header("Recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")

        overall_success = passed == total

        if overall_success:
            self.print_header("🎉 All Workflow Checks Passed! 🎉")
            print("Your GitHub workflows are properly configured!")
        else:
            self.print_header("⚠️  Some Issues Found")
            print("Please review the failed checks above.")

        return overall_success


def main():
    """Main entry point."""
    if not Path("setup.py").exists():
        print("❌ Error: setup.py not found. Please run from the Surprise root directory.")
        return 1

    verifier = WorkflowVerifier()
    success = verifier.verify_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
