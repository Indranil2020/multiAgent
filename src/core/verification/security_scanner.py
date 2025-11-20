"""
Security Scanner - Layer 7 of Verification Stack

This module provides security scanning for code vulnerabilities.
It detects common security issues and potential exploits.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set
import ast
import time
import re

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationIssue
)


@dataclass
class SecurityScannerConfig:
    """Configuration for security scanner"""
    strict_mode: bool = True
    check_injection: bool = True
    check_dangerous_functions: bool = True
    check_hardcoded_secrets: bool = True
    check_insecure_random: bool = True
    check_path_traversal: bool = True


class SecurityScanner:
    """
    Scans code for security vulnerabilities.
    
    This scanner detects common security issues including injection vulnerabilities,
    use of dangerous functions, hardcoded secrets, and other security risks.
    """
    
    def __init__(self, config: Optional[SecurityScannerConfig] = None):
        """
        Initialize the security scanner.
        
        Args:
            config: Configuration for the scanner
        """
        self.config = config if config is not None else SecurityScannerConfig()
        
        # Define dangerous patterns
        self.dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'os.system', 'subprocess.call', 'subprocess.Popen',
            'pickle.loads', 'yaml.load', 'marshal.loads'
        }
        
        self.insecure_random_functions = {
            'random.random', 'random.randint', 'random.choice'
        }
        
        self.secret_patterns = [
            r'password\s*=\s*["\'].*["\']',
            r'api_key\s*=\s*["\'].*["\']',
            r'secret\s*=\s*["\'].*["\']',
            r'token\s*=\s*["\'].*["\']',
            r'aws_access_key',
            r'private_key'
        ]
    
    def verify(self, code: str, code_id: str = "unknown") -> VerificationResult:
        """
        Scan code for security vulnerabilities.
        
        Args:
            code: Source code to scan
            code_id: Identifier for the code being scanned
            
        Returns:
            VerificationResult with security scan status
        """
        start_time = time.time()
        
        # Validate input
        if not isinstance(code, str):
            return self._create_error_result(
                "Code must be a string",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if len(code.strip()) == 0:
            return self._create_error_result(
                "Code cannot be empty",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse code
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for security scanning",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Perform security checks
        issues = []
        
        # Check for dangerous functions
        if self.config.check_dangerous_functions:
            dangerous_issues = self._check_dangerous_functions(tree)
            issues.extend(dangerous_issues)
        
        # Check for injection vulnerabilities
        if self.config.check_injection:
            injection_issues = self._check_injection_vulnerabilities(tree, code)
            issues.extend(injection_issues)
        
        # Check for hardcoded secrets
        if self.config.check_hardcoded_secrets:
            secret_issues = self._check_hardcoded_secrets(code)
            issues.extend(secret_issues)
        
        # Check for insecure random
        if self.config.check_insecure_random:
            random_issues = self._check_insecure_random(tree)
            issues.extend(random_issues)
        
        # Check for path traversal
        if self.config.check_path_traversal:
            path_issues = self._check_path_traversal(tree)
            issues.extend(path_issues)
        
        # Additional security checks
        additional_issues = self._check_additional_security(tree)
        issues.extend(additional_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        
        if critical_issues == 0 and high_issues == 0:
            return VerificationResult(
                layer=VerificationLayer.SECURITY,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"Security scan passed with {len(issues)} minor issue(s)",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "total_issues": len(issues),
                    "critical_issues": critical_issues,
                    "high_issues": high_issues
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.SECURITY,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Security scan failed with {critical_issues} critical and {high_issues} high severity issues",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "total_issues": len(issues),
                    "critical_issues": critical_issues,
                    "high_issues": high_issues
                }
            )
            return result
    
    def _parse_code(self, code: str) -> Optional[ast.AST]:
        """Parse code into AST"""
        parsed_tree = None
        
        if self._can_parse(code):
            parsed_tree = ast.parse(code)
        
        return parsed_tree
    
    def _can_parse(self, code: str) -> bool:
        """Check if code can be parsed"""
        return code and isinstance(code, str) and len(code.strip()) > 0
    
    def _check_dangerous_functions(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for use of dangerous functions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if func_name in self.dangerous_functions:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.SECURITY,
                        severity="critical",
                        message=f"Dangerous function '{func_name}' detected",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion=f"Avoid using '{func_name}' as it poses security risks"
                    ))
        
        return issues
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle module.function calls
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return ""
    
    def _check_injection_vulnerabilities(self, tree: ast.AST, code: str) -> List[VerificationIssue]:
        """Check for injection vulnerabilities"""
        issues = []
        
        # Check for SQL injection
        sql_issues = self._check_sql_injection(tree)
        issues.extend(sql_issues)
        
        # Check for command injection
        command_issues = self._check_command_injection(tree)
        issues.extend(command_issues)
        
        # Check for code injection
        code_issues = self._check_code_injection(tree)
        issues.extend(code_issues)
        
        return issues
    
    def _check_sql_injection(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for SQL injection vulnerabilities"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for string formatting in SQL queries
                if self._is_sql_query(node):
                    if self._uses_string_formatting(node):
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.SECURITY,
                            severity="critical",
                            message="Potential SQL injection vulnerability detected",
                            line_number=node.lineno if hasattr(node, 'lineno') else None,
                            suggestion="Use parameterized queries instead of string formatting"
                        ))
        
        return issues
    
    def _is_sql_query(self, node: ast.Call) -> bool:
        """Check if call is likely a SQL query"""
        # Check for common SQL keywords in string arguments
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']
                if any(keyword in arg.value.upper() for keyword in sql_keywords):
                    return True
        
        return False
    
    def _uses_string_formatting(self, node: ast.Call) -> bool:
        """Check if node uses string formatting"""
        for arg in node.args:
            if isinstance(arg, ast.JoinedStr):  # f-string
                return True
            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):  # % formatting
                return True
            if isinstance(arg, ast.Call):
                if isinstance(arg.func, ast.Attribute) and arg.func.attr == 'format':
                    return True
        
        return False
    
    def _check_command_injection(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for command injection vulnerabilities"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if 'system' in func_name or 'Popen' in func_name or 'call' in func_name:
                    # Check if using shell=True
                    if self._uses_shell_true(node):
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.SECURITY,
                            severity="critical",
                            message="Command injection vulnerability: shell=True detected",
                            line_number=node.lineno if hasattr(node, 'lineno') else None,
                            suggestion="Avoid shell=True or properly sanitize inputs"
                        ))
        
        return issues
    
    def _uses_shell_true(self, node: ast.Call) -> bool:
        """Check if call uses shell=True"""
        for keyword in node.keywords:
            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    return True
        
        return False
    
    def _check_code_injection(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for code injection vulnerabilities"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if func_name in ['eval', 'exec', 'compile']:
                    # Check if input is from external source
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.SECURITY,
                        severity="critical",
                        message=f"Code injection risk: '{func_name}' with potentially untrusted input",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion=f"Avoid using '{func_name}' with user input"
                    ))
        
        return issues
    
    def _check_hardcoded_secrets(self, code: str) -> List[VerificationIssue]:
        """Check for hardcoded secrets"""
        issues = []
        
        for pattern in self.secret_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            
            for match in matches:
                # Calculate line number
                line_number = code[:match.start()].count('\n') + 1
                
                issues.append(VerificationIssue(
                    layer=VerificationLayer.SECURITY,
                    severity="high",
                    message="Potential hardcoded secret detected",
                    line_number=line_number,
                    code_snippet=match.group(),
                    suggestion="Use environment variables or secure secret management"
                ))
        
        return issues
    
    def _check_insecure_random(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for insecure random number generation"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if func_name in self.insecure_random_functions:
                    # Check if used in security context
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.SECURITY,
                        severity="medium",
                        message=f"Insecure random function '{func_name}' detected",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion="Use secrets module for cryptographic purposes"
                    ))
        
        return issues
    
    def _check_path_traversal(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for path traversal vulnerabilities"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                # Check for file operations
                if 'open' in func_name or 'read' in func_name or 'write' in func_name:
                    # Check if path comes from user input
                    if self._uses_user_input(node):
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.SECURITY,
                            severity="high",
                            message="Potential path traversal vulnerability",
                            line_number=node.lineno if hasattr(node, 'lineno') else None,
                            suggestion="Validate and sanitize file paths from user input"
                        ))
        
        return issues
    
    def _uses_user_input(self, node: ast.Call) -> bool:
        """Check if call uses user input"""
        # Simplified check - look for input() or request parameters
        for arg in node.args:
            if isinstance(arg, ast.Call):
                func_name = self._get_function_name(arg)
                if 'input' in func_name or 'request' in func_name:
                    return True
        
        return False
    
    def _check_additional_security(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for additional security issues"""
        issues = []
        
        # Check for assert statements (can be disabled with -O)
        assert_issues = self._check_assert_for_security(tree)
        issues.extend(assert_issues)
        
        # Check for pickle usage
        pickle_issues = self._check_pickle_usage(tree)
        issues.extend(pickle_issues)
        
        # Check for weak cryptography
        crypto_issues = self._check_weak_cryptography(tree)
        issues.extend(crypto_issues)
        
        return issues
    
    def _check_assert_for_security(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for security-critical assert statements"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # Check if assert is used for security checks
                # This is heuristic - look for security-related keywords
                assert_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
                
                security_keywords = ['auth', 'permission', 'admin', 'password', 'token']
                if any(keyword in assert_str.lower() for keyword in security_keywords):
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.SECURITY,
                        severity="medium",
                        message="Assert used for security check (can be disabled with -O)",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion="Use explicit if statements for security checks"
                    ))
        
        return issues
    
    def _check_pickle_usage(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for unsafe pickle usage"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if 'pickle.loads' in func_name or 'pickle.load' in func_name:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.SECURITY,
                        severity="high",
                        message="Unsafe pickle deserialization detected",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion="Avoid unpickling untrusted data or use safer serialization"
                    ))
        
        return issues
    
    def _check_weak_cryptography(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for weak cryptographic algorithms"""
        issues = []
        
        weak_algorithms = ['md5', 'sha1', 'des', 'rc4']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node).lower()
                
                for weak_algo in weak_algorithms:
                    if weak_algo in func_name:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.SECURITY,
                            severity="medium",
                            message=f"Weak cryptographic algorithm '{weak_algo}' detected",
                            line_number=node.lineno if hasattr(node, 'lineno') else None,
                            suggestion=f"Use stronger algorithms like SHA-256 or AES instead of {weak_algo}"
                        ))
        
        return issues
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.SECURITY,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
