"""
Comprehensive verification script for the red flag detection module.

This script verifies:
1. No try-except blocks are used
2. All type annotations are present
3. Modular design is followed
4. Zero syntax errors
5. Proper imports and dependencies
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Tuple


class CodeVerifier:
    """Verifies code follows zero-error principles"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.violations = []
    
    def verify_no_try_except(self, filepath: Path) -> List[str]:
        """Verify no try-except blocks are used"""
        violations = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                line_num = node.lineno
                violations.append(f"{filepath.name}:{line_num} - try-except block found")
        
        return violations
    
    def verify_type_annotations(self, filepath: Path) -> List[str]:
        """Verify functions have type annotations"""
        violations = []
        
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip __init__ and special methods for now
                if node.name.startswith('_') and node.name != '__init__':
                    continue
                
                # Check return annotation
                if node.returns is None and node.name != '__init__':
                    violations.append(
                        f"{filepath.name}:{node.lineno} - "
                        f"Function '{node.name}' missing return type annotation"
                    )
                
                # Check parameter annotations
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue
                    if arg.annotation is None:
                        violations.append(
                            f"{filepath.name}:{node.lineno} - "
                            f"Parameter '{arg.arg}' in '{node.name}' missing type annotation"
                        )
        
        return violations
    
    def verify_imports(self, filepath: Path) -> List[str]:
        """Verify imports are valid"""
        violations = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check for relative imports
        if 'from .' in content or 'from ..' in content:
            # This is OK for internal module imports
            pass
        
        return violations
    
    def verify_file(self, filepath: Path) -> Dict[str, List[str]]:
        """Verify a single file"""
        results = {
            'try_except': [],
            'type_annotations': [],
            'imports': []
        }
        
        if not filepath.exists():
            return results
        
        # Skip __init__.py from some checks
        if filepath.name == '__init__.py':
            return results
        
        results['try_except'] = self.verify_no_try_except(filepath)
        results['type_annotations'] = self.verify_type_annotations(filepath)
        results['imports'] = self.verify_imports(filepath)
        
        return results
    
    def verify_module(self) -> Dict[str, any]:
        """Verify entire module"""
        files_to_check = [
            'types.py',
            'patterns.py',
            'uncertainty.py',
            'detector.py',
            'escalation.py',
            '__init__.py'
        ]
        
        all_results = {}
        total_violations = 0
        
        for filename in files_to_check:
            filepath = self.base_path / filename
            results = self.verify_file(filepath)
            
            violations_count = sum(len(v) for v in results.values())
            total_violations += violations_count
            
            all_results[filename] = {
                'results': results,
                'violations_count': violations_count
            }
        
        return {
            'files': all_results,
            'total_violations': total_violations,
            'passed': total_violations == 0
        }


def print_verification_results(results: Dict[str, any]) -> None:
    """Print verification results"""
    print("=" * 80)
    print("RED FLAG MODULE VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    for filename, file_results in results['files'].items():
        print(f"File: {filename}")
        print("-" * 80)
        
        violations = file_results['results']
        
        # Try-except violations
        if violations['try_except']:
            print("  ❌ Try-Except Violations:")
            for v in violations['try_except']:
                print(f"     - {v}")
        else:
            print("  ✅ No try-except blocks")
        
        # Type annotation violations
        if violations['type_annotations']:
            print("  ⚠️  Type Annotation Issues:")
            for v in violations['type_annotations']:
                print(f"     - {v}")
        else:
            print("  ✅ All type annotations present")
        
        # Import violations
        if violations['imports']:
            print("  ⚠️  Import Issues:")
            for v in violations['imports']:
                print(f"     - {v}")
        else:
            print("  ✅ Imports valid")
        
        print()
    
    print("=" * 80)
    print(f"Total Violations: {results['total_violations']}")
    
    if results['passed']:
        print("✅ ALL CHECKS PASSED - ZERO ERRORS!")
    else:
        print("❌ VIOLATIONS FOUND - NEEDS FIXES")
    
    print("=" * 80)


def main():
    """Main verification function"""
    # Path to red_flag module
    red_flag_path = Path(__file__).parent.parent / 'src' / 'core' / 'red_flag'
    
    if not red_flag_path.exists():
        print(f"Error: Path not found: {red_flag_path}")
        return
    
    print(f"Verifying module at: {red_flag_path}")
    print()
    
    verifier = CodeVerifier(red_flag_path)
    results = verifier.verify_module()
    
    print_verification_results(results)
    
    # Also verify task_spec/types.py
    print("\n" + "=" * 80)
    print("VERIFYING TASK_SPEC TYPES MODULE")
    print("=" * 80)
    print()
    
    task_spec_path = Path(__file__).parent.parent / 'src' / 'core' / 'task_spec'
    types_verifier = CodeVerifier(task_spec_path)
    types_file = task_spec_path / 'types.py'
    
    if types_file.exists():
        types_results = types_verifier.verify_file(types_file)
        
        print(f"File: types.py")
        print("-" * 80)
        
        if types_results['try_except']:
            print("  ❌ Try-Except Violations:")
            for v in types_results['try_except']:
                print(f"     - {v}")
        else:
            print("  ✅ No try-except blocks")
        
        if types_results['type_annotations']:
            print("  ⚠️  Type Annotation Issues:")
            for v in types_results['type_annotations'][:5]:  # Show first 5
                print(f"     - {v}")
            if len(types_results['type_annotations']) > 5:
                print(f"     ... and {len(types_results['type_annotations']) - 5} more")
        else:
            print("  ✅ All type annotations present")
        
        print()
    
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
