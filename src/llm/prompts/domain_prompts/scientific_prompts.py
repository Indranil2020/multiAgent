"""
Scientific computing prompt templates for numerical analysis, simulations, and data science.

This module provides comprehensive prompts for scientific software development including:
- Numerical computing with NumPy/SciPy
- Scientific simulations and modeling
- Statistical analysis and hypothesis testing
- Machine learning and data science pipelines
- Optimization algorithms and computational methods

Dependencies:
    numpy>=1.24.0: Core numerical computing library
    scipy>=1.10.0: Scientific computing and technical computing
    pandas>=2.0.0: Data manipulation and analysis
    matplotlib>=3.7.0: Plotting and visualization
    scikit-learn>=1.3.0: Machine learning library
    numba>=0.57.0: JIT compilation for numerical functions
    joblib>=1.3.0: Parallel computing utilities
    h5py>=3.8.0: HDF5 file format for large datasets
    xarray>=2023.1.0: N-dimensional labeled arrays
    sympy>=1.12: Symbolic mathematics
    statsmodels>=0.14.0: Statistical modeling
    seaborn>=0.12.0: Statistical data visualization

All templates enforce zero-error principles with complete implementations.
"""

from typing import Optional, List
from llm.prompts.base_prompts import PromptTemplate


NUMERICAL_COMPUTING_PROMPT = PromptTemplate(
    template_id="numerical_computing",
    name="Numerical Computing Prompt",
    template_text="""Generate a complete numerical computing implementation using NumPy/SciPy.

COMPUTATION NAME: {computation_name}
DESCRIPTION: {description}
MATHEMATICAL FORMULATION: {math_formulation}
INPUT SPECIFICATIONS: {inputs}
OUTPUT SPECIFICATIONS: {outputs}
NUMERICAL REQUIREMENTS: {numerical_requirements}
PERFORMANCE CONSTRAINTS: {performance_constraints}

REQUIRED DEPENDENCIES:
- numpy>=1.24.0
- scipy>=1.10.0
- numba>=0.57.0 (for performance-critical sections)

IMPLEMENTATION REQUIREMENTS:

1. NUMERICAL ACCURACY:
   - Use appropriate data types (float64 for high precision, float32 for memory/speed)
   - Handle numerical stability (avoid division by zero, overflow, underflow)
   - Implement error bounds and tolerance checking
   - Validate convergence criteria for iterative methods
   - Document expected numerical precision

2. VECTORIZATION:
   - Use NumPy broadcasting instead of loops
   - Leverage BLAS/LAPACK through NumPy
   - Profile and optimize hot paths with numba.jit
   - Minimize array copies (use views where possible)
   - Use in-place operations when safe

3. ERROR HANDLING:
   - Validate array shapes and dimensions
   - Check for NaN/Inf values in inputs and outputs
   - Handle singular matrices and ill-conditioned systems
   - Raise informative errors for invalid inputs
   - Implement numerical warnings for precision loss

4. PERFORMANCE OPTIMIZATION:
   - Use contiguous arrays (C-order vs F-order)
   - Minimize memory allocations
   - Consider cache locality
   - Parallelize independent operations
   - Use specialized NumPy functions (einsum, dot, etc.)

5. TESTING REQUIREMENTS:
   - Unit tests with known analytical solutions
   - Numerical convergence tests
   - Edge cases (empty arrays, single element, large arrays)
   - Stability tests (perturbation analysis)
   - Performance benchmarks

CODE STRUCTURE:

```python
import numpy as np
from scipy import linalg, optimize, integrate, special
from numba import jit, prange
from typing import Union, Tuple, Optional
import warnings


def validate_input_{computation_name}(
    array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_finite: bool = True
) -> None:
    \"\"\"
    Validate numerical input arrays.

    Args:
        array: Input array to validate
        expected_shape: Expected shape (None for any shape)
        check_finite: Check for NaN/Inf values

    Raises:
        ValueError: If validation fails
    \"\"\"
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy array, got {{type(array).__name__}}")

    if check_finite and not np.all(np.isfinite(array)):
        raise ValueError("Array contains NaN or Inf values")

    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"Expected shape {{expected_shape}}, got {{array.shape}}")


@jit(nopython=True, cache=True)
def _optimized_kernel(data: np.ndarray) -> np.ndarray:
    \"\"\"
    Numba-optimized computational kernel.

    Use @jit decorator for performance-critical numerical operations.
    Supports parallel execution with prange for independent iterations.
    \"\"\"
    # Implement core numerical computation here
    # This function will be compiled to machine code by Numba
    pass


def {computation_name}(
    # Input parameters here
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    \"\"\"
    {description}

    Mathematical Formulation:
    {math_formulation}

    Args:
        # Document all input parameters
        # Include shape requirements, valid ranges, units

    Returns:
        # Document return values
        # Include shape, data type, interpretation

    Raises:
        ValueError: If input validation fails
        RuntimeError: If computation fails to converge

    Examples:
        >>> # Provide working numerical examples
        >>> x = np.linspace(0, 1, 100)
        >>> result = {computation_name}(x)
        >>> assert result.shape == x.shape
        >>> assert np.all(np.isfinite(result))

    Notes:
        - Numerical complexity: O(...)
        - Memory complexity: O(...)
        - Typical convergence: ... iterations
        - Numerical stability: ...
    \"\"\"
    # 1. Input validation
    # Validate all inputs with detailed error messages

    # 2. Handle special cases
    # Empty arrays, single elements, edge conditions

    # 3. Initialize computation
    # Allocate output arrays with correct dtype and shape
    # Set up convergence criteria if iterative

    # 4. Core computation
    # Use vectorized NumPy operations
    # Call Numba-optimized kernels for hot paths
    # Implement algorithm with numerical stability in mind

    # 5. Convergence/error checking
    # For iterative methods, check convergence
    # Verify output is finite and within expected range
    # Issue warnings if precision is questionable

    # 6. Return results
    # Return with documented shape and dtype
    pass


# Example implementations for reference:

def solve_linear_system_stable(
    A: np.ndarray,
    b: np.ndarray,
    rcond: float = 1e-15
) -> Tuple[np.ndarray, dict]:
    \"\"\"
    Solve linear system Ax = b with stability checks.

    Uses SVD for ill-conditioned systems, LU decomposition for well-conditioned.

    Args:
        A: Coefficient matrix (n, n)
        b: Right-hand side (n,) or (n, m)
        rcond: Cutoff for small singular values

    Returns:
        x: Solution vector(s)
        info: Dictionary with condition number, residual, method used

    Examples:
        >>> A = np.array([[3, 1], [1, 2]], dtype=float)
        >>> b = np.array([9, 8], dtype=float)
        >>> x, info = solve_linear_system_stable(A, b)
        >>> np.allclose(A @ x, b)
        True
    \"\"\"
    validate_input_numerical_computing(A, check_finite=True)
    validate_input_numerical_computing(b, check_finite=True)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {{A.shape}}")

    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: A{{A.shape}}, b{{b.shape}}")

    # Compute condition number
    cond_number = np.linalg.cond(A)

    info = {{'condition_number': cond_number}}

    # Choose method based on condition number
    if cond_number > 1e10:
        warnings.warn(f"Matrix is ill-conditioned (κ={{cond_number:.2e}}), using SVD")
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
        info['method'] = 'svd'
        info['rank'] = rank
        info['singular_values'] = s
    else:
        # Use LU decomposition for well-conditioned systems
        x = linalg.solve(A, b, assume_a='gen', check_finite=False)
        info['method'] = 'lu'

    # Compute residual
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    info['relative_residual'] = residual

    if residual > 1e-6:
        warnings.warn(f"Large residual: {{residual:.2e}}")

    return x, info


def integrate_adaptive_quadrature(
    func: callable,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 1000
) -> Tuple[float, dict]:
    \"\"\"
    Adaptive quadrature integration with error estimation.

    Args:
        func: Function to integrate (must be vectorized)
        a: Lower bound
        b: Upper bound
        tol: Absolute tolerance
        max_iter: Maximum iterations

    Returns:
        integral: Numerical integral value
        info: Convergence information

    Examples:
        >>> def f(x): return np.sin(x)
        >>> result, info = integrate_adaptive_quadrature(f, 0, np.pi)
        >>> np.isclose(result, 2.0, atol=1e-8)
        True
    \"\"\"
    result, error = integrate.quad(func, a, b, epsabs=tol, limit=max_iter)

    info = {{
        'estimated_error': error,
        'tolerance': tol,
        'converged': error < tol
    }}

    if not info['converged']:
        warnings.warn(f"Integration may not have converged: error={{error:.2e}}")

    return result, info
```

FORBIDDEN PRACTICES:
- NO manual loops for array operations (use vectorization)
- NO ignoring numerical warnings
- NO unchecked array operations that could overflow
- NO missing input validation
- NO returning NaN/Inf without errors
- NO hardcoded array sizes
- NO single precision (float32) without justification

OUTPUT FORMAT:
Provide complete, production-ready numerical computing code with:
1. All imports at the top
2. Input validation functions
3. Numba-optimized kernels if needed
4. Main computation function(s)
5. Comprehensive docstrings with mathematical formulation
6. Type hints
7. Example usage in docstrings
8. Unit tests demonstrating correctness
""",
    variables=[
        "computation_name",
        "description",
        "math_formulation",
        "inputs",
        "outputs",
        "numerical_requirements",
        "performance_constraints"
    ]
)


SCIENTIFIC_SIMULATION_PROMPT = PromptTemplate(
    template_id="scientific_simulation",
    name="Scientific Simulation Prompt",
    template_text="""Generate a complete scientific simulation implementation.

SIMULATION NAME: {simulation_name}
DESCRIPTION: {description}
PHYSICAL/MATHEMATICAL MODEL: {model_description}
STATE VARIABLES: {state_variables}
PARAMETERS: {parameters}
INITIAL CONDITIONS: {initial_conditions}
TIME/ITERATION REQUIREMENTS: {time_requirements}

REQUIRED DEPENDENCIES:
- numpy>=1.24.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- numba>=0.57.0
- h5py>=3.8.0 (for data storage)

SIMULATION REQUIREMENTS:

1. STATE MANAGEMENT:
   - Explicitly define all state variables
   - Use appropriate data structures (arrays, dataclasses)
   - Document units for all physical quantities
   - Implement state validation and bounds checking
   - Support state serialization/deserialization

2. NUMERICAL INTEGRATION:
   - Choose appropriate integrator (Euler, RK4, adaptive methods)
   - Implement timestep control
   - Monitor energy/conserved quantities
   - Handle stiff systems appropriately
   - Document integration scheme and stability

3. BOUNDARY CONDITIONS:
   - Implement all boundary conditions explicitly
   - Handle periodic boundaries correctly
   - Validate boundary consistency
   - Document boundary treatment

4. OBSERVABLES AND DIAGNOSTICS:
   - Compute relevant physical observables
   - Track conservation laws
   - Implement checkpointing for long simulations
   - Provide progress monitoring
   - Store results efficiently (HDF5 for large data)

5. VERIFICATION:
   - Test against analytical solutions where available
   - Verify conservation laws numerically
   - Check order of accuracy (convergence tests)
   - Validate with published results
   - Document expected behaviors

CODE STRUCTURE:

```python
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import h5py
from pathlib import Path
import warnings


@dataclass
class SimulationParameters:
    \"\"\"
    Physical and numerical parameters for simulation.

    All physical quantities should include units in docstrings.
    \"\"\"
    # Physical parameters
    # Include units, valid ranges, physical meaning

    # Numerical parameters
    dt: float  # Timestep (seconds)
    t_max: float  # Maximum simulation time (seconds)
    output_interval: int  # Save state every N steps

    # Tolerances
    energy_tolerance: float = 1e-6  # Relative energy conservation

    def __post_init__(self):
        \"\"\"Validate parameters.\"\"\"
        if self.dt <= 0:
            raise ValueError("Timestep must be positive")
        if self.t_max <= 0:
            raise ValueError("Maximum time must be positive")


@dataclass
class SimulationState:
    \"\"\"
    Complete simulation state at a given time.

    Includes all variables needed to continue simulation.
    \"\"\"
    time: float
    # State variables here (positions, velocities, fields, etc.)
    # Use numpy arrays with documented shapes

    # Conserved quantities for validation
    energy: float = 0.0
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def validate(self) -> None:
        \"\"\"Validate state consistency.\"\"\"
        # Check array shapes
        # Verify physical constraints
        # Check for NaN/Inf
        pass

    def save(self, filepath: Path) -> None:
        \"\"\"Save state to HDF5 file.\"\"\"
        with h5py.File(filepath, 'w') as f:
            f.attrs['time'] = self.time
            f.attrs['energy'] = self.energy
            # Save all array data
            # f.create_dataset('positions', data=self.positions)

    @classmethod
    def load(cls, filepath: Path) -> 'SimulationState':
        \"\"\"Load state from HDF5 file.\"\"\"
        with h5py.File(filepath, 'r') as f:
            time = f.attrs['time']
            energy = f.attrs['energy']
            # Load all array data
            # positions = f['positions'][:]
            # return cls(time=time, positions=positions, energy=energy)
        pass


class {simulation_name}:
    \"\"\"
    {description}

    Physical Model:
    {model_description}

    State Variables:
    {state_variables}

    Governing Equations:
    # Document differential equations or update rules

    Attributes:
        params: Simulation parameters
        state: Current simulation state
        history: List of saved states

    Examples:
        >>> params = SimulationParameters(dt=0.01, t_max=10.0)
        >>> sim = {simulation_name}(params)
        >>> sim.initialize()
        >>> sim.run()
        >>> sim.plot_results()
    \"\"\"

    def __init__(self, params: SimulationParameters):
        \"\"\"
        Initialize simulation.

        Args:
            params: Simulation parameters
        \"\"\"
        self.params = params
        self.state: Optional[SimulationState] = None
        self.history: List[SimulationState] = []
        self.initial_energy: Optional[float] = None

    def initialize(self, initial_state: Optional[SimulationState] = None) -> None:
        \"\"\"
        Initialize or reset simulation state.

        Args:
            initial_state: Custom initial state (None for default)
        \"\"\"
        if initial_state is None:
            # Create default initial state
            # {initial_conditions}
            self.state = SimulationState(time=0.0)
            # Initialize state variables
        else:
            self.state = initial_state

        self.state.validate()
        self.initial_energy = self.compute_energy(self.state)
        self.history = [self.state]

    def compute_energy(self, state: SimulationState) -> float:
        \"\"\"
        Compute total energy (kinetic + potential).

        Args:
            state: Simulation state

        Returns:
            Total energy
        \"\"\"
        # Implement energy calculation
        # Should match physical model
        kinetic_energy = 0.0
        potential_energy = 0.0
        return kinetic_energy + potential_energy

    def compute_forces(self, state: SimulationState) -> np.ndarray:
        \"\"\"
        Compute forces/accelerations from current state.

        Args:
            state: Current simulation state

        Returns:
            Forces or accelerations array
        \"\"\"
        # Implement force calculation based on physical model
        # Use vectorized operations for particle systems
        # Consider using spatial data structures (KDTree) for neighbor searches
        pass

    def step(self, dt: float) -> None:
        \"\"\"
        Advance simulation by one timestep.

        Args:
            dt: Timestep size

        Raises:
            RuntimeError: If integration fails or conservation is violated
        \"\"\"
        if self.state is None:
            raise RuntimeError("Simulation not initialized")

        # Implement integration scheme
        # Common choices:
        # - Euler: simple but unstable
        # - Velocity Verlet: good for conservative systems
        # - RK4: general purpose, 4th order accurate
        # - Adaptive methods: scipy.integrate.solve_ivp

        # Example: Velocity Verlet for molecular dynamics
        # 1. Compute forces at current state
        # forces = self.compute_forces(self.state)

        # 2. Update positions (half step)
        # self.state.positions += self.state.velocities * dt + 0.5 * forces * dt**2

        # 3. Compute forces at new positions
        # new_forces = self.compute_forces(self.state)

        # 4. Update velocities
        # self.state.velocities += 0.5 * (forces + new_forces) * dt

        # 5. Update time
        self.state.time += dt

        # 6. Update conserved quantities
        self.state.energy = self.compute_energy(self.state)

        # 7. Check conservation
        if self.initial_energy is not None:
            rel_error = abs(self.state.energy - self.initial_energy) / abs(self.initial_energy)
            if rel_error > self.params.energy_tolerance:
                warnings.warn(
                    f"Energy conservation violated: "
                    f"ΔE/E = {{rel_error:.2e}} > {{self.params.energy_tolerance:.2e}}"
                )

        # 8. Validate state
        self.state.validate()

    def run(self, callback: Optional[Callable] = None) -> None:
        \"\"\"
        Run complete simulation.

        Args:
            callback: Optional function called after each output interval
                     Signature: callback(state: SimulationState) -> None
        \"\"\"
        if self.state is None:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        n_steps = int(self.params.t_max / self.params.dt)

        print(f"Running simulation: {{n_steps}} steps, dt={{self.params.dt}}")

        for step_idx in range(n_steps):
            self.step(self.params.dt)

            # Save state at output intervals
            if (step_idx + 1) % self.params.output_interval == 0:
                self.history.append(self.state)

                if callback is not None:
                    callback(self.state)

                # Progress report
                progress = 100 * (step_idx + 1) / n_steps
                print(f"Progress: {{progress:.1f}}% | "
                      f"t={{self.state.time:.3f}} | "
                      f"E={{self.state.energy:.6e}}")

        print("Simulation complete")

    def save_trajectory(self, filepath: Path) -> None:
        \"\"\"
        Save complete trajectory to HDF5.

        Args:
            filepath: Output file path
        \"\"\"
        with h5py.File(filepath, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            params_group.attrs['dt'] = self.params.dt
            params_group.attrs['t_max'] = self.params.t_max

            # Save trajectory
            n_frames = len(self.history)
            # Create datasets for time series
            # times = np.array([state.time for state in self.history])
            # f.create_dataset('times', data=times)

            # Save state arrays
            # Example: positions shape (n_frames, n_particles, 3)

        print(f"Trajectory saved to {{filepath}}")

    def plot_results(self) -> None:
        \"\"\"Generate diagnostic plots.\"\"\"
        if not self.history:
            raise RuntimeError("No data to plot")

        times = np.array([state.time for state in self.history])
        energies = np.array([state.energy for state in self.history])

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Energy conservation
        axes[0].plot(times, energies)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Total Energy')
        axes[0].set_title('Energy Conservation')
        axes[0].grid(True)

        # Relative energy error
        if self.initial_energy is not None:
            rel_error = (energies - self.initial_energy) / self.initial_energy
            axes[1].semilogy(times, np.abs(rel_error))
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('|ΔE/E|')
            axes[1].set_title('Relative Energy Error')
            axes[1].grid(True)
            axes[1].axhline(self.params.energy_tolerance, color='r',
                           linestyle='--', label='Tolerance')
            axes[1].legend()

        plt.tight_layout()
        plt.show()


# Unit tests
def test_{simulation_name}():
    \"\"\"Test simulation with known solution.\"\"\"
    # Create simple test case with analytical solution
    # Compare numerical vs analytical
    # Verify energy conservation
    # Check convergence with timestep refinement
    pass
```

FORBIDDEN PRACTICES:
- NO implicit units (always document)
- NO unchecked conservation violations
- NO hardcoded array sizes
- NO missing boundary conditions
- NO unvalidated initial conditions
- NO single precision without justification
- NO skipping convergence tests

OUTPUT FORMAT:
Complete scientific simulation code with:
1. Parameter and state dataclasses
2. Simulation class with documented methods
3. Energy/conservation checking
4. Checkpointing and data storage
5. Visualization methods
6. Verification tests
""",
    variables=[
        "simulation_name",
        "description",
        "model_description",
        "state_variables",
        "parameters",
        "initial_conditions",
        "time_requirements"
    ]
)


STATISTICAL_ANALYSIS_PROMPT = PromptTemplate(
    template_id="statistical_analysis",
    name="Statistical Analysis Prompt",
    template_text="""Generate a complete statistical analysis implementation.

ANALYSIS NAME: {analysis_name}
DESCRIPTION: {description}
DATA DESCRIPTION: {data_description}
STATISTICAL METHODS: {statistical_methods}
HYPOTHESES: {hypotheses}
SIGNIFICANCE LEVEL: {significance_level}

REQUIRED DEPENDENCIES:
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- statsmodels>=0.14.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

STATISTICAL ANALYSIS REQUIREMENTS:

1. DATA VALIDATION:
   - Check for missing values (NaN)
   - Detect and handle outliers
   - Verify distributional assumptions
   - Check sample size adequacy
   - Document data cleaning steps

2. DESCRIPTIVE STATISTICS:
   - Compute central tendency (mean, median, mode)
   - Compute dispersion (std, variance, IQR)
   - Compute shape (skewness, kurtosis)
   - Generate summary tables
   - Create exploratory visualizations

3. INFERENTIAL STATISTICS:
   - State null and alternative hypotheses
   - Check test assumptions (normality, homoscedasticity)
   - Compute test statistics
   - Calculate p-values
   - Interpret results with effect sizes
   - Apply multiple testing corrections

4. MODEL DIAGNOSTICS:
   - Residual analysis
   - Goodness-of-fit tests
   - Cross-validation
   - Influence diagnostics
   - Model comparison (AIC, BIC)

5. REPORTING:
   - Summary statistics tables
   - Confidence intervals
   - Visualizations with uncertainty
   - Interpretation of results
   - Limitations and assumptions

CODE STRUCTURE:

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, levene, bartlett,
    ttest_ind, ttest_rel, mannwhitneyu,
    f_oneway, kruskal, chi2_contingency
)
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings


@dataclass
class StatisticalResult:
    \"\"\"
    Container for statistical test results.

    Attributes:
        test_name: Name of statistical test
        statistic: Test statistic value
        pvalue: P-value
        effect_size: Effect size measure (Cohen's d, eta-squared, etc.)
        ci_lower: Lower confidence interval bound
        ci_upper: Upper confidence interval bound
        significant: Whether result is significant at alpha level
        interpretation: Human-readable interpretation
    \"\"\"
    test_name: str
    statistic: float
    pvalue: float
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    significant: bool = False
    interpretation: str = ""

    def __str__(self) -> str:
        \"\"\"Format results for display.\"\"\"
        result = f"{{self.test_name}}:\\n"
        result += f"  Statistic: {{self.statistic:.4f}}\\n"
        result += f"  P-value: {{self.pvalue:.4g}}\\n"

        if self.effect_size is not None:
            result += f"  Effect size: {{self.effect_size:.4f}}\\n"

        if self.ci_lower is not None and self.ci_upper is not None:
            result += f"  95% CI: [{{self.ci_lower:.4f}}, {{self.ci_upper:.4f}}]\\n"

        result += f"  Significant: {{self.significant}}\\n"
        result += f"  Interpretation: {{self.interpretation}}"

        return result


def check_normality(
    data: np.ndarray,
    method: str = 'shapiro',
    alpha: float = 0.05
) -> Tuple[bool, float]:
    \"\"\"
    Test for normality using Shapiro-Wilk or other tests.

    Args:
        data: 1D array of samples
        method: Test method ('shapiro', 'normaltest', 'anderson')
        alpha: Significance level

    Returns:
        is_normal: True if data appears normal
        pvalue: P-value from test

    Examples:
        >>> np.random.seed(42)
        >>> data = np.random.normal(0, 1, 100)
        >>> is_normal, pvalue = check_normality(data)
        >>> is_normal
        True
    \"\"\"
    data = np.asarray(data)

    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    if len(data_clean) < 3:
        warnings.warn("Too few samples for normality test")
        return False, np.nan

    if method == 'shapiro':
        if len(data_clean) > 5000:
            warnings.warn("Shapiro-Wilk not recommended for n > 5000, using normaltest")
            method = 'normaltest'
        else:
            statistic, pvalue = shapiro(data_clean)
    elif method == 'normaltest':
        statistic, pvalue = normaltest(data_clean)
    else:
        raise ValueError(f"Unknown method: {{method}}")

    is_normal = pvalue > alpha

    return is_normal, pvalue


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True
) -> float:
    \"\"\"
    Compute Cohen's d effect size.

    Args:
        group1: First group samples
        group2: Second group samples
        pooled: Use pooled standard deviation

    Returns:
        Cohen's d effect size

    Notes:
        Interpretation:
        - Small: |d| ~ 0.2
        - Medium: |d| ~ 0.5
        - Large: |d| ~ 0.8
    \"\"\"
    mean1, mean2 = np.mean(group1), np.mean(group2)

    if pooled:
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std
    else:
        std1 = np.std(group1, ddof=1)
        d = (mean1 - mean2) / std1

    return d


def {analysis_name}(
    data: pd.DataFrame,
    alpha: float = {significance_level}
) -> Dict[str, StatisticalResult]:
    \"\"\"
    {description}

    Data Description:
    {data_description}

    Statistical Methods:
    {statistical_methods}

    Hypotheses:
    {hypotheses}

    Args:
        data: DataFrame with required columns
        alpha: Significance level (default: {significance_level})

    Returns:
        Dictionary of test results

    Raises:
        ValueError: If data validation fails

    Examples:
        >>> # Create sample data
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({{
        ...     'group': ['A'] * 50 + ['B'] * 50,
        ...     'value': np.concatenate([
        ...         np.random.normal(10, 2, 50),
        ...         np.random.normal(12, 2, 50)
        ...     ])
        ... }})
        >>> results = {analysis_name}(data)
        >>> results['t_test'].significant
        True
    \"\"\"
    results = {{}}

    # 1. DATA VALIDATION
    # Check for required columns
    # Handle missing values
    # Detect outliers
    # Document data cleaning

    # 2. DESCRIPTIVE STATISTICS
    # Compute summary statistics for each group
    # Create summary DataFrame

    desc_stats = data.groupby('group')['value'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ])

    print("\\nDescriptive Statistics:")
    print(desc_stats)

    # 3. ASSUMPTION CHECKING
    # Check normality for each group
    groups = data.groupby('group')['value'].apply(np.array)

    print("\\nNormality Tests:")
    for group_name, group_data in groups.items():
        is_normal, pvalue = check_normality(group_data)
        print(f"  {{group_name}}: p={{pvalue:.4f}}, normal={{is_normal}}")

    # Check homogeneity of variance
    if len(groups) == 2:
        stat, pvalue = levene(*groups.values)
        equal_var = pvalue > alpha
        print(f"\\nLevene's test: p={{pvalue:.4f}}, equal_var={{equal_var}}")

    # 4. INFERENTIAL TESTS
    # Perform appropriate statistical tests based on assumptions

    # Example: Independent t-test or Mann-Whitney U
    if len(groups) == 2:
        group_a, group_b = groups.values

        # Check if parametric assumptions met
        all_normal = all(check_normality(g)[0] for g in groups.values)

        if all_normal and equal_var:
            # Parametric test: Independent t-test
            statistic, pvalue = ttest_ind(group_a, group_b, equal_var=True)
            test_name = "Independent t-test"

            # Compute effect size
            effect_size = compute_cohens_d(group_a, group_b)

            # Confidence interval for mean difference
            diff = np.mean(group_a) - np.mean(group_b)
            se_diff = np.sqrt(np.var(group_a, ddof=1)/len(group_a) +
                            np.var(group_b, ddof=1)/len(group_b))
            df = len(group_a) + len(group_b) - 2
            t_crit = stats.t.ppf(1 - alpha/2, df)
            ci_lower = diff - t_crit * se_diff
            ci_upper = diff + t_crit * se_diff

        else:
            # Non-parametric test: Mann-Whitney U
            statistic, pvalue = mannwhitneyu(group_a, group_b, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            effect_size = None
            ci_lower = None
            ci_upper = None

        # Store result
        significant = pvalue < alpha

        if significant:
            interpretation = f"Significant difference found (p={{pvalue:.4g}} < {{alpha}})"
        else:
            interpretation = f"No significant difference (p={{pvalue:.4g}} >= {{alpha}})"

        results['main_test'] = StatisticalResult(
            test_name=test_name,
            statistic=statistic,
            pvalue=pvalue,
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=significant,
            interpretation=interpretation
        )

    # 5. MULTIPLE TESTING CORRECTION (if applicable)
    # If performing multiple comparisons, apply correction
    # pvalues_corrected = multipletests(pvalues, method='fdr_bh')[1]

    # 6. POWER ANALYSIS
    # Compute achieved power or required sample size
    if 'effect_size' in results['main_test'].__dict__ and results['main_test'].effect_size:
        power_analysis = TTestIndPower()
        n_per_group = len(group_a)
        achieved_power = power_analysis.solve_power(
            effect_size=abs(results['main_test'].effect_size),
            nobs1=n_per_group,
            alpha=alpha,
            ratio=1.0
        )
        print(f"\\nAchieved power: {{achieved_power:.3f}}")

    return results


def visualize_results(
    data: pd.DataFrame,
    results: Dict[str, StatisticalResult]
) -> None:
    \"\"\"
    Create publication-quality visualizations.

    Args:
        data: Original DataFrame
        results: Statistical test results
    \"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot with individual points
    sns.boxplot(data=data, x='group', y='value', ax=axes[0])
    sns.stripplot(data=data, x='group', y='value', ax=axes[0],
                 color='black', alpha=0.3, size=3)
    axes[0].set_title('Distribution by Group')
    axes[0].set_ylabel('Value')

    # Violin plot with quartiles
    sns.violinplot(data=data, x='group', y='value', ax=axes[1])
    axes[1].set_title('Density Estimation by Group')
    axes[1].set_ylabel('Value')

    # Add significance annotation
    if 'main_test' in results and results['main_test'].significant:
        y_max = data['value'].max()
        y_range = data['value'].max() - data['value'].min()
        y_pos = y_max + 0.1 * y_range

        axes[0].plot([0, 1], [y_pos, y_pos], 'k-', lw=1.5)
        axes[0].text(0.5, y_pos, f"p={{results['main_test'].pvalue:.3g}}",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_per_group = 50

    data = pd.DataFrame({{
        'group': ['Control'] * n_per_group + ['Treatment'] * n_per_group,
        'value': np.concatenate([
            np.random.normal(10, 2, n_per_group),
            np.random.normal(12, 2, n_per_group)
        ])
    }})

    # Run analysis
    results = {analysis_name}(data)

    # Print results
    for test_name, result in results.items():
        print(f"\\n{{test_name}}:")
        print(result)

    # Visualize
    visualize_results(data, results)
```

FORBIDDEN PRACTICES:
- NO p-hacking or selective reporting
- NO ignoring violated assumptions
- NO missing multiple testing corrections
- NO undocumented outlier removal
- NO overstating conclusions
- NO missing effect sizes
- NO ignoring power analysis

OUTPUT FORMAT:
Complete statistical analysis with:
1. Data validation and cleaning
2. Descriptive statistics
3. Assumption checking
4. Appropriate inferential tests
5. Effect sizes and confidence intervals
6. Multiple testing corrections
7. Visualizations
8. Interpretation with limitations
""",
    variables=[
        "analysis_name",
        "description",
        "data_description",
        "statistical_methods",
        "hypotheses",
        "significance_level"
    ]
)


MACHINE_LEARNING_PIPELINE_PROMPT = PromptTemplate(
    template_id="ml_pipeline",
    name="Machine Learning Pipeline Prompt",
    template_text="""Generate a complete machine learning pipeline implementation.

PIPELINE NAME: {pipeline_name}
DESCRIPTION: {description}
ML TASK: {ml_task}
FEATURES: {features}
TARGET: {target}
EVALUATION METRICS: {evaluation_metrics}

REQUIRED DEPENDENCIES:
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

OPTIONAL DEPENDENCIES (based on task):
- xgboost>=1.7.0 (gradient boosting)
- lightgbm>=3.3.0 (gradient boosting)
- imbalanced-learn>=0.11.0 (imbalanced datasets)

ML PIPELINE REQUIREMENTS:

1. DATA PREPROCESSING:
   - Handle missing values explicitly
   - Encode categorical variables
   - Scale/normalize numerical features
   - Feature engineering
   - Train/test split with stratification

2. MODEL SELECTION:
   - Compare multiple algorithms
   - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Cross-validation (stratified k-fold)
   - Handle class imbalance if present
   - Document model choices

3. TRAINING:
   - Fit on training data only
   - Monitor training metrics
   - Save trained models
   - Record hyperparameters
   - Version control for experiments

4. EVALUATION:
   - Multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
   - Confusion matrix
   - Learning curves
   - Cross-validation scores
   - Feature importance analysis

5. PREDICTION:
   - Input validation
   - Preprocessing consistency
   - Uncertainty estimation
   - Explainability (SHAP, feature importance)

CODE STRUCTURE:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings


@dataclass
class MLConfig:
    \"\"\"Machine learning pipeline configuration.\"\"\"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1  # Use all CPUs

    # Model hyperparameters
    model_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {{}}


@dataclass
class MLResults:
    \"\"\"Container for ML results.\"\"\"
    train_score: float
    test_score: float
    cv_scores: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        \"\"\"Generate summary report.\"\"\"
        report = f"Training Score: {{self.train_score:.4f}}\\n"
        report += f"Test Score: {{self.test_score:.4f}}\\n"
        report += f"CV Score: {{self.cv_scores.mean():.4f}} (+/- {{self.cv_scores.std():.4f}})\\n"
        report += f"\\nClassification Report:\\n{{self.classification_report}}"
        return report


class {pipeline_name}:
    \"\"\"
    {description}

    Task: {ml_task}
    Features: {features}
    Target: {target}
    Evaluation: {evaluation_metrics}

    Attributes:
        config: Pipeline configuration
        preprocessor: Fitted preprocessing pipeline
        model: Fitted ML model
        results: Evaluation results

    Examples:
        >>> # Load data
        >>> data = pd.read_csv('data.csv')
        >>> X = data.drop('target', axis=1)
        >>> y = data['target']
        >>>
        >>> # Create and train pipeline
        >>> config = MLConfig(test_size=0.2)
        >>> pipeline = {pipeline_name}(config)
        >>> pipeline.fit(X, y)
        >>>
        >>> # Evaluate
        >>> pipeline.evaluate()
        >>> pipeline.plot_results()
        >>>
        >>> # Predict
        >>> predictions = pipeline.predict(X_new)
    \"\"\"

    def __init__(self, config: MLConfig):
        \"\"\"Initialize ML pipeline.\"\"\"
        self.config = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.model: Optional[Any] = None
        self.results: Optional[MLResults] = None
        self.feature_names: Optional[List[str]] = None

        # Data splits
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def create_preprocessor(
        self,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
        \"\"\"
        Create preprocessing pipeline.

        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names

        Returns:
            ColumnTransformer for preprocessing
        \"\"\"
        from sklearn.preprocessing import OneHotEncoder

        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False,
                                     handle_unknown='ignore'))
        ])

        # Combined preprocessor
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

        return preprocessor

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True
    ) -> None:
        \"\"\"
        Split data into train and test sets.

        Args:
            X: Feature DataFrame
            y: Target Series
            stratify: Use stratified split for classification
        \"\"\"
        stratify_var = y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_var
        )

        print(f"Train size: {{len(self.X_train)}}")
        print(f"Test size: {{len(self.X_test)}}")
        print(f"Class distribution:")
        print(self.y_train.value_counts(normalize=True))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> None:
        \"\"\"
        Fit preprocessing and model.

        Args:
            X: Feature DataFrame
            y: Target Series
            numerical_features: List of numerical columns (auto-detected if None)
            categorical_features: List of categorical columns (auto-detected if None)
        \"\"\"
        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.feature_names = numerical_features + categorical_features

        # Split data
        self.split_data(X, y)

        # Create and fit preprocessor
        self.preprocessor = self.create_preprocessor(numerical_features, categorical_features)
        X_train_processed = self.preprocessor.fit_transform(self.X_train)

        # Create and fit model
        # Example: Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            **self.config.model_params
        )

        print("\\nTraining model...")
        self.model.fit(X_train_processed, self.y_train)
        print("Model training complete")

    def evaluate(self) -> MLResults:
        \"\"\"
        Evaluate model performance.

        Returns:
            MLResults object with all evaluation metrics
        \"\"\"
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Transform data
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)

        # Training score
        train_score = self.model.score(X_train_processed, self.y_train)

        # Test score
        test_score = self.model.score(X_test_processed, self.y_test)

        # Cross-validation scores
        cv = StratifiedKFold(n_splits=self.config.cv_folds,
                           shuffle=True,
                           random_state=self.config.random_state)
        cv_scores = cross_val_score(
            self.model,
            X_train_processed,
            self.y_train,
            cv=cv,
            n_jobs=self.config.n_jobs
        )

        # Predictions
        y_pred = self.model.predict(X_test_processed)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Classification report
        report = classification_report(self.y_test, y_pred)

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names after transformation
            feature_names_transformed = self.preprocessor.get_feature_names_out()

            importance_df = pd.DataFrame({{
                'feature': feature_names_transformed,
                'importance': self.model.feature_importances_
            }}).sort_values('importance', ascending=False)

            feature_importance = importance_df
        else:
            feature_importance = None

        self.results = MLResults(
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores,
            confusion_matrix=cm,
            classification_report=report,
            feature_importance=feature_importance
        )

        print("\\n" + self.results.summary())

        return self.results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        \"\"\"
        Make predictions on new data.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions array
        \"\"\"
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Validate features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {{missing_features}}")

        X_processed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_processed)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        \"\"\"Get prediction probabilities.\"\"\"
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions")

        X_processed = self.preprocessor.transform(X)
        probabilities = self.model.predict_proba(X_processed)

        return probabilities

    def plot_results(self) -> None:
        \"\"\"Generate evaluation visualizations.\"\"\"
        if self.results is None:
            raise RuntimeError("No results to plot. Call evaluate() first.")

        fig = plt.figure(figsize=(15, 5))

        # Confusion matrix
        ax1 = plt.subplot(131)
        sns.heatmap(self.results.confusion_matrix, annot=True, fmt='d',
                   cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Cross-validation scores
        ax2 = plt.subplot(132)
        ax2.bar(range(len(self.results.cv_scores)), self.results.cv_scores)
        ax2.axhline(self.results.cv_scores.mean(), color='r',
                   linestyle='--', label='Mean')
        ax2.set_title('Cross-Validation Scores')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Score')
        ax2.legend()

        # Feature importance (top 15)
        if self.results.feature_importance is not None:
            ax3 = plt.subplot(133)
            top_features = self.results.feature_importance.head(15)
            ax3.barh(range(len(top_features)), top_features['importance'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_title('Top 15 Feature Importances')
            ax3.set_xlabel('Importance')

        plt.tight_layout()
        plt.show()

    def save(self, filepath: Path) -> None:
        \"\"\"Save trained pipeline.\"\"\"
        joblib.dump({{
            'preprocessor': self.preprocessor,
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names
        }}, filepath)
        print(f"Pipeline saved to {{filepath}}")

    @classmethod
    def load(cls, filepath: Path) -> '{pipeline_name}':
        \"\"\"Load trained pipeline.\"\"\"
        data = joblib.load(filepath)
        pipeline = cls(data['config'])
        pipeline.preprocessor = data['preprocessor']
        pipeline.model = data['model']
        pipeline.feature_names = data['feature_names']
        return pipeline
```

FORBIDDEN PRACTICES:
- NO data leakage (fitting on test data)
- NO missing train/test split
- NO single metric evaluation
- NO ignoring class imbalance
- NO hardcoded feature names
- NO missing cross-validation
- NO overfitting (monitor train vs test)

OUTPUT FORMAT:
Complete ML pipeline with:
1. Configuration dataclass
2. Preprocessing pipeline
3. Model training with CV
4. Comprehensive evaluation
5. Feature importance analysis
6. Prediction with validation
7. Model persistence
8. Visualization utilities
""",
    variables=[
        "pipeline_name",
        "description",
        "ml_task",
        "features",
        "target",
        "evaluation_metrics"
    ]
)


OPTIMIZATION_ALGORITHM_PROMPT = PromptTemplate(
    template_id="optimization_algorithm",
    name="Optimization Algorithm Prompt",
    template_text="""Generate a complete optimization algorithm implementation.

ALGORITHM NAME: {algorithm_name}
DESCRIPTION: {description}
OPTIMIZATION TYPE: {optimization_type}
OBJECTIVE FUNCTION: {objective_function}
CONSTRAINTS: {constraints}
ALGORITHM DETAILS: {algorithm_details}

REQUIRED DEPENDENCIES:
- numpy>=1.24.0
- scipy>=1.10.0

OPTIONAL DEPENDENCIES:
- numba>=0.57.0 (for performance)

OPTIMIZATION ALGORITHM REQUIREMENTS:

1. PROBLEM FORMULATION:
   - Define objective function clearly
   - Document constraints (equality, inequality, bounds)
   - Specify decision variables
   - Document problem type (convex, nonconvex, etc.)

2. ALGORITHM IMPLEMENTATION:
   - Implement stopping criteria (tolerance, max iterations)
   - Track convergence history
   - Handle numerical stability
   - Support warm starts
   - Provide convergence diagnostics

3. VALIDATION:
   - Test on problems with known solutions
   - Verify KKT conditions for constrained problems
   - Check gradient/Hessian accuracy (finite differences)
   - Benchmark against scipy.optimize

4. PERFORMANCE:
   - Minimize function evaluations
   - Use efficient linear algebra
   - Consider sparse matrices if applicable
   - Profile and optimize bottlenecks

CODE STRUCTURE:

```python
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict
import warnings


@dataclass
class OptimizationResult:
    \"\"\"Container for optimization results.\"\"\"
    x: np.ndarray  # Optimal solution
    fun: float  # Objective value at solution
    success: bool  # Whether optimization succeeded
    message: str  # Termination message
    nit: int  # Number of iterations
    nfev: int  # Number of function evaluations
    njev: int  # Number of Jacobian evaluations
    history: List[float] = field(default_factory=list)  # Objective history

    def __str__(self) -> str:
        result = f"Optimization Result:\\n"
        result += f"  Success: {{self.success}}\\n"
        result += f"  Message: {{self.message}}\\n"
        result += f"  Iterations: {{self.nit}}\\n"
        result += f"  Function evaluations: {{self.nfev}}\\n"
        result += f"  Final objective: {{self.fun:.6e}}\\n"
        result += f"  Solution: {{self.x}}"
        return result


def {algorithm_name}(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    constraints: Optional[List[Dict]] = None,
    bounds: Optional[Bounds] = None,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> OptimizationResult:
    \"\"\"
    {description}

    Algorithm: {algorithm_details}
    Optimization Type: {optimization_type}
    Objective: {objective_function}
    Constraints: {constraints}

    Args:
        objective: Objective function f(x) -> float
        x0: Initial guess (n,)
        gradient: Gradient function g(x) -> (n,) (auto-computed if None)
        hessian: Hessian function H(x) -> (n, n) (optional)
        constraints: List of constraint dictionaries
        bounds: Bounds on variables
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        OptimizationResult object

    Examples:
        >>> # Minimize f(x) = (x[0]-1)^2 + (x[1]-2)^2
        >>> def f(x):
        ...     return (x[0]-1)**2 + (x[1]-2)**2
        >>> def grad_f(x):
        ...     return np.array([2*(x[0]-1), 2*(x[1]-2)])
        >>> x0 = np.array([0.0, 0.0])
        >>> result = {algorithm_name}(f, x0, gradient=grad_f)
        >>> np.allclose(result.x, [1.0, 2.0])
        True
    \"\"\"
    # Validate inputs
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)

    # Auto-compute gradient if not provided
    if gradient is None:
        gradient = lambda x: finite_difference_gradient(objective, x)

    # Initialize tracking
    history = []
    x = x0.copy()
    iteration = 0

    # Main optimization loop
    while iteration < max_iter:
        # Evaluate objective
        f_val = objective(x)
        history.append(f_val)

        # Evaluate gradient
        g = gradient(x)
        grad_norm = np.linalg.norm(g)

        # Check convergence
        if grad_norm < tol:
            return OptimizationResult(
                x=x,
                fun=f_val,
                success=True,
                message="Converged: gradient norm below tolerance",
                nit=iteration,
                nfev=len(history),
                njev=iteration+1,
                history=history
            )

        # Compute search direction
        # (Algorithm-specific implementation here)
        # Example: Steepest descent
        direction = -g

        # Line search
        step_size = backtracking_line_search(objective, x, direction, g)

        # Update
        x_new = x + step_size * direction

        # Handle bounds
        if bounds is not None:
            x_new = np.clip(x_new, bounds.lb, bounds.ub)

        # Check for sufficient decrease
        if abs(objective(x_new) - f_val) < tol:
            return OptimizationResult(
                x=x_new,
                fun=objective(x_new),
                success=True,
                message="Converged: objective change below tolerance",
                nit=iteration+1,
                nfev=len(history)+1,
                njev=iteration+1,
                history=history
            )

        x = x_new
        iteration += 1

    # Max iterations reached
    return OptimizationResult(
        x=x,
        fun=objective(x),
        success=False,
        message="Maximum iterations reached",
        nit=iteration,
        nfev=len(history),
        njev=iteration,
        history=history
    )


def finite_difference_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    \"\"\"
    Compute gradient using central finite differences.

    Args:
        f: Objective function
        x: Point to evaluate gradient
        eps: Finite difference step size

    Returns:
        Gradient vector
    \"\"\"
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    gradient: np.ndarray,
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50
) -> float:
    \"\"\"
    Backtracking line search with Armijo condition.

    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        gradient: Gradient at x
        alpha_init: Initial step size
        rho: Step size reduction factor
        c: Armijo parameter
        max_iter: Maximum iterations

    Returns:
        Step size
    \"\"\"
    alpha = alpha_init
    f_x = f(x)
    slope = np.dot(gradient, direction)

    for _ in range(max_iter):
        x_new = x + alpha * direction

        # Check Armijo condition
        if f(x_new) <= f_x + c * alpha * slope:
            return alpha

        alpha *= rho

    warnings.warn("Line search failed to satisfy Armijo condition")
    return alpha


def verify_gradient(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6
) -> Tuple[bool, float]:
    \"\"\"
    Verify gradient implementation using finite differences.

    Args:
        f: Objective function
        gradient: Gradient function
        x: Point to check
        eps: Finite difference step

    Returns:
        is_correct: Whether gradient is correct
        max_error: Maximum absolute error
    \"\"\"
    grad_analytic = gradient(x)
    grad_numerical = finite_difference_gradient(f, x, eps)

    error = np.abs(grad_analytic - grad_numerical)
    max_error = np.max(error)

    is_correct = max_error < 1e-5

    if not is_correct:
        print(f"Gradient verification failed:")
        print(f"  Max error: {{max_error:.2e}}")
        print(f"  Analytic: {{grad_analytic}}")
        print(f"  Numerical: {{grad_numerical}}")

    return is_correct, max_error


# Example: Rosenbrock function (standard test problem)
def test_rosenbrock():
    \"\"\"Test optimization on Rosenbrock function.\"\"\"
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad_rosenbrock(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])

    # Verify gradient
    x_test = np.array([0.5, 0.5])
    is_correct, error = verify_gradient(rosenbrock, grad_rosenbrock, x_test)
    assert is_correct, f"Gradient incorrect: error={{error}}"

    # Optimize
    x0 = np.array([0.0, 0.0])
    result = {algorithm_name}(rosenbrock, x0, gradient=grad_rosenbrock)

    print(result)

    # Check solution (should be [1, 1])
    assert result.success
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-3)

    return result
```

FORBIDDEN PRACTICES:
- NO missing convergence criteria
- NO unchecked gradients
- NO infinite loops
- NO hardcoded dimensions
- NO ignoring constraints
- NO missing validation tests

OUTPUT FORMAT:
Complete optimization code with:
1. Result dataclass
2. Main optimization function
3. Gradient utilities
4. Line search
5. Convergence checking
6. Validation functions
7. Test problems
""",
    variables=[
        "algorithm_name",
        "description",
        "optimization_type",
        "objective_function",
        "constraints",
        "algorithm_details"
    ]
)


# Dictionary of all scientific computing templates
ALL_SCIENTIFIC_TEMPLATES = {
    "numerical_computing": NUMERICAL_COMPUTING_PROMPT,
    "scientific_simulation": SCIENTIFIC_SIMULATION_PROMPT,
    "statistical_analysis": STATISTICAL_ANALYSIS_PROMPT,
    "ml_pipeline": MACHINE_LEARNING_PIPELINE_PROMPT,
    "optimization_algorithm": OPTIMIZATION_ALGORITHM_PROMPT,
}


def get_scientific_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get scientific computing prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_SCIENTIFIC_TEMPLATES.get(template_id)


def list_scientific_templates() -> List[str]:
    """
    List all available scientific computing template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_SCIENTIFIC_TEMPLATES.keys())
