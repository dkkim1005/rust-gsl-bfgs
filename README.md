# gsl_bfgs
### Rust implementation of the BFGS algorithm built on the GNU Scientific Library(GSL)

The `gsl_bfgs` crate provides a pure Rust implementation of the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm, which is a popular method for numerical optimization, especially in unconstrained optimization problems. This implementation is derived from the GNU Scientific Library (GSL), which is written in C.

## how to use?

The main entry point for using the BFGS optimization in this crate is the function `gsl_bfgs::optimize`. To use it, you need to implement the Callback trait for your specific target function. The Callback trait defines three essential functions that will be used by the optimizer:

1.	`eval`: Evaluates the target function at a given point.
2.	`grad`: Computes the gradient (partial derivatives) of the target function at a given point.
3.	`ndim`: Returns the number of dimensions for the input vector.

Below is the example code for how to use the `gsl_bfgs::optimize` function:

``` rust
use gsl_bfgs;

/// Struct representing the Rosenbrock function:
/// f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
struct Rosenbrock;

impl gsl_bfgs::Callback for Rosenbrock {
    /// Evaluate the Rosenbrock function at a given point `x`
    fn eval(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    /// Compute the gradient (partial derivatives) of the function
    fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
        g[0] = 400.0 * x[0].powi(3) - 400.0 * x[0] * x[1] + 2.0 * x[0] - 2.0; // ∂f/∂x
        g[1] = 200.0 * (x[1] - x[0].powi(2)); // ∂f/∂y
    }

    /// Return the number of dimensions for `x`
    fn ndim(&self) -> usize {
        2
    }
}

let func = Rosenbrock;
let mut p: Vec<f64> = vec![0.0, 0.0]; // Initial guess for the minimum
let itermax = 100; // Maximum number of iterations
let toll = 1e-7; // Tolerance for convergence

/// Perform the BFGS optimization to find the local minimum.
/// The optimized result is stored in `p`.
gsl_bfgs::optimize(&func, &mut p, itermax, toll);
```

## contact
If you encounter a bug or have suggestions to improve the code, feel free to submit a pull request.
 * email: dkkim1005@gmail.com
