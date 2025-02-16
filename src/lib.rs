// Copyright (C) 2025 Dongkyu Kim (dkkim1005@gmail.com)
//
// This code is a derivative work of the GNU Scientific Library (GSL) project and its extended version.
// It is licensed under the GNU General Public License Version 3.0 (GPL-3.0) by
// Brian Gough, Fabrice Rossi, Dong-Hee Kim, and many others.
//
// This derivative work is licensed under the GNU GPL-3.0 License by Dongkyu Kim.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU GPL-3 as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU GPL-3.0 for more details.
//
// You should have received a copy of the GNU GPL-3.0
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/// Represents the status of the optimization using BFGS algorithm.
///
/// This enum is used to indicate whether BFGS algorithm has
/// successfully converged, failed, or requires more iterations.
#[derive(Debug)]
pub enum Status {
    /// Indicates that the optimization has successfully converged.
    ///
    /// # Fields
    /// - `(f64, f64)`: A tuple containing:
    ///   - **Function value (`f64`)**: The value of the objective function at convergence.
    ///   - **Gradient norm (`f64`)**: The norm of the gradient at the final point.
    ///
    /// This variant is returned when the optimization reaches a local optimum.
    Converged((f64, f64)),

    /// Represents an abnormal termination state.
    ///
    /// This occurs when the optimization process could fail to find a local
    /// optimal point due to round-off errors.
    Abnormal,

    /// Indicates that the optimizer has exceeded the allowed number of iterations.
    ///
    /// This means that the optimization algorithm needs more iterations to
    /// reach the optimal point but was stopped due to iteration limits.
    TooManyIterations,
}

/// A trait representing a callback for function optimization.
///
/// # Example
///
/// ```
/// use gsl_bfgs;
///
/// struct Test;
///
/// impl gsl_bfgs::Callback for Test {
///     /// Evaluates a quadratic function.
///     fn eval(&self, x: &Vec<f64>) -> f64 {
///         (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
///     }
///
///     /// Computes the gradient of the quadratic function.
///     fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
///         g[0] = 2.0 * (x[0] - 1.0);
///         g[1] = 2.0 * (x[1] - 2.0);
///     }
///
///     /// Returns the number of dimensions (2 in this case).
///     fn ndim(&self) -> usize {
///         2
///     }
/// }
/// ```
pub trait Callback {
    /// Evaluates the target function for a given input vector.
    ///
    /// # Parameters
    /// - `x` (**in**) - The input vector for which the function is evaluated.
    ///
    /// # Returns
    /// - The function value at `x`.
    fn eval(&self, x: &Vec<f64>) -> f64;

    /// Computes the gradient vector of the target function at a given input.
    ///
    /// # Parameters
    /// - `x` (**in**) - The input vector at which the gradient is computed.
    /// - `g` (**out**) - The output vector to store the computed gradient.
    fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>);

    /// Returns the dimensionality of the input vector.
    ///
    /// # Returns
    /// - The number of dimensions of the input vector.
    fn ndim(&self) -> usize;
}

/// Optimize the target function:
///
/// # Parameters
/// - `func` (**in**) - The target function implementing the `Callback` trait.
/// - `p` (**in/out**) - The initial guess, which will be modified during optimization.
/// - `itmax` (**in**) - The number of iterations.
/// - `toll` (**in**) - A tolerance value; lower values result in higher precision.
pub fn optimize<F: Callback>(func: &F, p: &mut Vec<f64>, itmax: i32, toll: f64) -> Status {
    let n: usize = func.ndim();
    assert_eq!(n, p.len(), "dimension not matching for the input vector!");

    let mut dg: Vec<f64> = vec![0.0; n];
    let mut g: Vec<f64> = vec![0.0; n];
    let mut hdg: Vec<f64> = vec![0.0; n];
    let mut xi: Vec<f64> = vec![0.0; n];
    let mut hessian: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut fp = func.eval(p);
    func.grad(p, &mut g);
    let mut delta_f = 0.0;
    let mut g0norm = nrm2(&g);
    xi.clone_from_slice(&g);
    xi = scal(-1.0 / g0norm, &xi);
    let mut pnorm = nrm2(&xi);
    let mut fp0 = -g0norm;
    let mut alpha;

    for _ in 0..itmax {
        if pnorm.abs() <= std::f64::EPSILON
            || g0norm.abs() <= std::f64::EPSILON
            || fp.abs() <= std::f64::EPSILON
        {
            return Status::Abnormal;
        }

        if delta_f < 0.0 {
            let del = { -1.0_f64 * delta_f }.max(10.0 * std::f64::EPSILON * fp.abs());
            alpha = 1.0_f64.min(2.0 * del / (-fp0));
        } else {
            alpha = STPMX.abs();
        }

        if !search(func, p, &mut xi, &mut alpha) {
            return Status::Abnormal;
        }

        let pnew: Vec<f64> = axay(1.0, &p, alpha, &xi);
        let fret = func.eval(&pnew);
        delta_f = fret - fp;

        fp = fret;
        xi = axay(1.0, &pnew, -1.0, &p);
        p.clone_from_slice(&pnew);

        if delta_f.abs() < TOLX {
            return Status::Converged((fret, g0norm));
        }

        let mut test = xi
            .iter()
            .zip(p.iter())
            .map(|(&xi_i, &p_i)| xi_i.abs() / 1.0_f64.max(p_i.abs()))
            .reduce(f64::max)
            .unwrap();
        if test < TOLX {
            return Status::Converged((fret, g0norm));
        }

        dg.clone_from_slice(&g);
        func.grad(&p, &mut g);

        let den = 1.0_f64.max(fret.abs());
        test = g
            .iter()
            .zip(p.iter())
            .map(|(&g_i, &p_i)| g_i.abs() * 1.0_f64.max(p_i.abs()) / den)
            .reduce(f64::max)
            .unwrap();
        if test < toll {
            return Status::Converged((fret, nrm2(&g)));
        }

        dg = axay(1.0, &g, -1.0, &dg);
        hdg.iter_mut().zip(hessian.iter()).for_each(|(hdg_i, row)| {
            *hdg_i = dot(row, &dg);
        });

        let fac = dot(&dg, &xi);
        let fae = dot(&dg, &hdg);
        let sumdg = dot(&dg, &dg);
        let sumxi = dot(&xi, &xi);

        if fac > (std::f64::EPSILON * sumdg * sumxi).sqrt() {
            let fac = 1.0 / fac;
            let fad = 1.0 / fae;
            dg = axay(fac, &xi, -fad, &hdg);
            (0..n).for_each(|i| {
                (0..n).skip(i).for_each(|j| {
                    hessian[i][j] +=
                        fac * xi[i] * xi[j] - fad * hdg[i] * hdg[j] + fae * dg[i] * dg[j];
                    hessian[j][i] = hessian[i][j];
                })
            });
            xi = hessian.iter().map(|row| -dot(&row, &g)).collect();
        }

        g0norm = nrm2(&g);
        pnorm = nrm2(&xi);
        xi = scal(1.0 / pnorm, &xi);
        pnorm = nrm2(&xi);
        fp0 = dot(&xi, &g);
    }
    Status::TooManyIterations
}

/* line search to find a stepsize alpha */
fn search<F: Callback>(
    func: &F,
    xold: &mut Vec<f64>,
    p: &mut Vec<f64>,
    alpha_new: &mut f64,
) -> bool {
    let n = func.ndim();
    let mut alpha = *alpha_new;
    let mut alpha_prev: f64 = 0.0;
    let mut x = xold.clone();
    let mut df = vec![0_f64; n];
    let f0: f64 = func.eval(&x);
    func.grad(&x, &mut df);
    let fp0: f64 = dot(&df, p);
    let mut falpha_prev = f0;
    let mut fpalpha_prev = fp0;

    /* Avoid uninitialized variables morning */
    let mut a: f64 = 0.0;
    let mut b = alpha;
    let mut fa = f0;
    let mut fb: f64 = 0.0;
    let mut fpa = fp0;
    let mut fpb: f64 = 0.0;

    /* Begin bracketing */
    for _ in 0..BRACKET_ITERS {
        x = axay(1.0, &xold, alpha, &p);
        let falpha = func.eval(&x);

        /* Fletcher's rho test */
        if falpha > (f0 + alpha * RHO * fp0) || falpha >= falpha_prev {
            a = alpha_prev;
            fa = falpha_prev;
            fpa = fpalpha_prev;
            b = alpha;
            fb = falpha;
            fpb = std::f64::NAN;
            break;
        }

        func.grad(&x, &mut df);
        let fpalpha = dot(&df, p);

        /* Fletcher's sigma test */
        if fpalpha.abs() <= -SIGMA * fp0 {
            *alpha_new = alpha;
            return true;
        }

        if fpalpha >= 0.0 {
            a = alpha;
            fa = falpha;
            fpa = fpalpha;
            b = alpha_prev;
            fb = falpha_prev;
            fpb = fpalpha_prev;
            break; /* goto sectioning */
        }

        let delta = alpha - alpha_prev;
        let lower = alpha + delta;
        let upper = alpha + TAU1 * delta;
        let alpha_next = interpolate(
            alpha_prev,
            falpha_prev,
            fpalpha_prev,
            alpha,
            falpha,
            fpalpha,
            lower,
            upper,
            ORDER,
        );

        alpha_prev = alpha;
        falpha_prev = falpha;
        fpalpha_prev = fpalpha;
        alpha = alpha_next;
    }

    /*  Sectioning of bracket [a,b] */
    for _ in 0..SECTION_ITERS {
        let delta = b - a;
        {
            let lower = a + TAU2 * delta;
            let upper = b - TAU3 * delta;
            alpha = interpolate(a, fa, fpa, b, fb, fpb, lower, upper, ORDER);
        }
        x = axay(1.0, &xold, alpha, &p);
        let falpha = func.eval(&x);
        if (a - alpha) * fpa <= std::f64::EPSILON {
            /* roundoff prevents progress */
            return false;
        }
        if falpha > f0 + RHO * alpha * fp0 || falpha >= fa {
            b = alpha;
            fb = falpha;
            fpb = std::f64::NAN;
        } else {
            func.grad(&x, &mut df);
            let fpalpha = dot(&df, p);
            if fpalpha.abs() <= -SIGMA * fp0 {
                *alpha_new = alpha;
                return true;
            }
            if ((b - a) >= 0.0 && fpalpha >= 0.0) || ((b - a) <= 0.0 && fpalpha <= 0.0) {
                b = a;
                fb = fa;
                fpb = fpa;
                a = alpha;
                fa = falpha;
                fpa = fpalpha;
            } else {
                a = alpha;
                fa = falpha;
                fpa = fpalpha;
            }
        }
    }
    *alpha_new = alpha;
    true
}

/* solve_quadratic.c - finds the real roots of a x^2 + b x + c = 0 */
fn solve_quadratic(a: f64, b: f64, c: f64) -> (Option<f64>, Option<f64>) {
    let disc = b * b - 4_f64 * a * c;
    if a.abs() <= std::f64::EPSILON {
        /* Handle linear case */
        if b.abs() <= std::f64::EPSILON {
            return (None, None);
        } else {
            let x0 = -c / b;
            return (Some(x0), None);
        }
    }
    if disc > 0_f64 {
        if b.abs() <= std::f64::EPSILON {
            let r = (0.5 * disc.sqrt() / a).abs();
            return (Some(-r), Some(r));
        } else {
            let sgnb = {
                if b > 0_f64 {
                    0_f64
                } else {
                    -1_f64
                }
            };
            let temp = -0.5 * (b + sgnb + disc.sqrt());
            let r1 = temp / a;
            let r2 = c / temp;
            if r1 < r2 {
                return (Some(r1), Some(r2));
            } else {
                return (Some(r2), Some(r1));
            }
        }
    } else if disc.abs() <= std::f64::EPSILON {
        return (Some(-0.5 * b / a), Some(-0.5 * b / a));
    } else {
        return (None, None);
    }
}

/*
 * Find a minimum in x=[0,1] of the interpolating quadratic through
 * (0,f0) (1,f1) with derivative fp0 at x=0.  The interpolating
 * polynomial is q(x) = f0 + fp0 * z + (f1-f0-fp0) * z^2
 */

fn interp_quad(f0: f64, fp0: f64, f1: f64, zl: f64, zh: f64) -> f64 {
    let fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
    let fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));
    let c = 2_f64 * (f1 - f0 - fp0); /* curvature */
    let mut zmin = zl;
    let mut _fmin = fl;
    if fh < _fmin {
        zmin = zh;
        _fmin = fh;
    }
    if c > 0_f64 {
        /* positive curvature required for a minimum */
        let z = -fp0 / c; /* location of minimum */
        if z > zl && z < zh {
            let f = f0 + z * (fp0 + z * (f1 - f0 - fp0));
            if f < _fmin {
                zmin = z;
                _fmin = f;
            }
        }
    }
    zmin
}

/* Find a minimum in x=[0,1] of the interpolating cubic through
 * (0,f0) (1,f1) with derivatives fp0 at x=0 and fp1 at x=1.
 *
 * The interpolating polynomial is:
 *
 * c(x) = f0 + fp0 * z + eta * z^2 + xi * z^3
 *
 * where eta=3*(f1-f0)-2*fp0-fp1, xi=fp0+fp1-2*(f1-f0).
 */

fn cubic(c0: f64, c1: f64, c2: f64, c3: f64, z: f64) -> f64 {
    c0 + z * (c1 + z * (c2 + z * c3))
}

fn check_extremum(c0: f64, c1: f64, c2: f64, c3: f64, z: f64, zmin: &mut f64, fmin: &mut f64) {
    /* could make an early return by testing curvature >0 for minimum */
    let y = cubic(c0, c1, c2, c3, z);
    if y < *fmin {
        *zmin = z; /* accepted new point*/
        *fmin = y;
    }
}

fn interp_cubic(f0: f64, fp0: f64, f1: f64, fp1: f64, zl: f64, zh: f64) -> f64 {
    let eta = 3_f64 * (f1 - f0) - 2_f64 * fp0 - fp1;
    let xi = fp0 + fp1 - 2_f64 * (f1 - f0);
    let c0 = f0;
    let c1 = fp0;
    let c2 = eta;
    let c3 = xi;
    let mut zmin = zl;
    let mut fmin = cubic(c0, c1, c2, c3, zl);
    check_extremum(c0, c1, c2, c3, zh, &mut zmin, &mut fmin);
    let sols = solve_quadratic(3_f64 * c3, 2_f64 * c2, c1);
    if let Some(z0) = sols.0 {
        match sols.1 {
            Some(z1) => {
                if z0 > zl && z0 < zh {
                    check_extremum(c0, c1, c2, c3, z0, &mut zmin, &mut fmin);
                }
                if z1 > zl && z1 < zh {
                    check_extremum(c0, c1, c2, c3, z1, &mut zmin, &mut fmin);
                }
            }
            None => {
                if z0 > zl && z0 < zh {
                    check_extremum(c0, c1, c2, c3, z0, &mut zmin, &mut fmin);
                }
            }
        }
    }
    zmin
}

fn interpolate(
    a: f64,
    fa: f64,
    fpa: f64,
    b: f64,
    fb: f64,
    fpb: f64,
    xmin: f64,
    xmax: f64,
    order: i32,
) -> f64 {
    /* Map [a,b] to [0,1] */
    let mut zmin = (xmin - a) / (b - a);
    let mut zmax = (xmax - a) / (b - a);
    if zmin > zmax {
        let tmp = zmin;
        zmin = zmax;
        zmax = tmp;
    }
    let z: f64;
    if order > 2 && !fpb.is_nan() {
        z = interp_cubic(fa, fpa * (b - a), fb, fpb * (b - a), zmin, zmax);
    } else {
        z = interp_quad(fa, fpa * (b - a), fb, zmin, zmax);
    }
    a + z * (b - a)
}

fn dot(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    x.iter().zip(y.iter()).map(|(x, y)| x * y).sum()
}

fn nrm2(x: &Vec<f64>) -> f64 {
    x.iter().map(|a| a * a).sum::<f64>().sqrt()
}

fn scal(fac: f64, x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|a| fac * a).collect()
}

fn axay(alpha: f64, x: &Vec<f64>, beta: f64, y: &Vec<f64>) -> Vec<f64> {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| alpha * a + beta * b)
        .collect()
}

// control parameters for BFGS
const ORDER: i32 = 3;
const RHO: f64 = 0.01;
const SIGMA: f64 = 0.1;
const TAU1: f64 = 9.0;
const TAU2: f64 = 0.05;
const TAU3: f64 = 0.5;
const STPMX: f64 = 1.0;
const TOLX: f64 = 4.0 * std::f64::EPSILON;
const BRACKET_ITERS: i32 = 100;
const SECTION_ITERS: i32 = 100;

#[cfg(test)]
mod tests {

    const PASS_LEVEL: f64 = 1e-04;
    const TOLL: f64 = 1e-15;
    const ITERMAX: i32 = 5000;

    use crate::{optimize, Callback};
    use rand::Rng;

    fn euclidean_distance(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn test_optimization_for<T: Callback>(func: T, global_minimum: Vec<f64>) {
        let mut rng = rand::thread_rng();
        let mut p: Vec<f64> = (0..func.ndim())
            .map(|_| rng.gen_range(-0.25..0.25))
            .collect();
        optimize(&func, &mut p, ITERMAX, TOLL);
        assert!(euclidean_distance(&p, &global_minimum) < PASS_LEVEL);
    }

    // Test functions for optimization (https://en.wikipedia.org/wiki/Test_functions_for_optimization)
    struct Rosenbrock;
    struct Booth;
    struct McCormick;
    struct Rastrigin;
    struct Sphere;

    impl Callback for Rosenbrock {
        fn eval(&self, x: &Vec<f64>) -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        }

        fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
            g[0] = 400.0 * x[0].powi(3) - 400.0 * x[0] * x[1] + 2.0 * x[0] - 2.0;
            g[1] = 200.0 * (x[1] - x[0].powi(2));
        }

        fn ndim(&self) -> usize {
            2
        }
    }

    impl Callback for Booth {
        fn eval(&self, x: &Vec<f64>) -> f64 {
            (x[0] + 2.0 * x[1] - 7.0).powi(2) + (2.0 * x[0] + x[1] - 5.0).powi(2)
        }

        fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
            g[0] = 2.0 * (x[0] + 2.0 * x[1] - 7.0) + 4.0 * (2.0 * x[0] + x[1] - 5.0);
            g[1] = 4.0 * (x[0] + 2.0 * x[1] - 7.0) + 2.0 * (2.0 * x[0] + x[1] - 5.0);
        }

        fn ndim(&self) -> usize {
            2
        }
    }

    impl Callback for McCormick {
        fn eval(&self, x: &Vec<f64>) -> f64 {
            (x[0] + x[1]).sin() + (x[0] - x[1]).powi(2) - 1.5 * x[0] + 2.5 * x[1] + 1.0
        }

        fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
            g[0] = (x[0] + x[1]).cos() + 2.0 * x[0] - 2.0 * x[1] - 1.5;
            g[1] = (x[0] + x[1]).cos() - 2.0 * x[0] + 2.0 * x[1] + 2.5;
        }

        fn ndim(&self) -> usize {
            2
        }
    }

    impl Callback for Rastrigin {
        fn eval(&self, x: &Vec<f64>) -> f64 {
            let fac = 2.0 * std::f64::consts::PI;
            x.iter()
                .map(|a| a.powi(2) - 10.0 * (fac * a).cos())
                .sum::<f64>()
        }

        fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
            let fac = 2.0 * std::f64::consts::PI;
            x.iter().zip(g.iter_mut()).for_each(|(a, b)| {
                *b = *a * 2.0 + 10.0 * fac * (*a * fac).sin();
            });
        }

        fn ndim(&self) -> usize {
            100
        }
    }

    impl Callback for Sphere {
        fn eval(&self, x: &Vec<f64>) -> f64 {
            x.iter().map(|a| a.powi(2)).sum::<f64>()
        }

        fn grad(&self, x: &Vec<f64>, g: &mut Vec<f64>) {
            x.iter().zip(g.iter_mut()).for_each(|(a, b)| {
                *b = *a * 2.0;
            });
        }

        fn ndim(&self) -> usize {
            100
        }
    }

    #[test]
    fn test_for_rosenbrock() {
        let global_minimum: Vec<f64> = vec![1.0, 1.0];
        test_optimization_for(Rosenbrock, global_minimum);
    }

    #[test]
    fn test_for_booth() {
        let global_minimum: Vec<f64> = vec![1.0, 3.0];
        test_optimization_for(Booth, global_minimum);
    }

    #[test]
    fn test_for_mccormick() {
        let global_minimum: Vec<f64> = vec![-0.54719, -1.54719];
        test_optimization_for(McCormick, global_minimum);
    }

    #[test]
    fn test_for_rastrigin() {
        let global_minimum: Vec<f64> = vec![0.0; 100];
        test_optimization_for(Rastrigin, global_minimum);
    }

    #[test]
    fn test_for_sphere() {
        let global_minimum: Vec<f64> = vec![0.0; 100];
        test_optimization_for(Sphere, global_minimum);
    }
}
