//!
//! # MosekModel library
//!
//! MosekModel is a crate for setting up conic optimization models to be solved with MOSEK. The
//! interface currently supports 
//! - Linear and conic variables and constraints
//! - Integer variables
//!
//! The model used is this:
//! ```math 
//! min/max   c^t x 
//! such that A x + b ∊ Kc
//!           X ∊ Kx
//! ```
//! where `Kc=Kc_0 × ... × Kc_m` and `Kx=Kx_0 × ... × Kx_n`, each `Kc_i` and `Kx_i` is a conic
//! domain from the currently supported set plus an offset:
//! - Non-positive or non-negative orthant (see [nonpositive], [nonnegative], [less_than] and
//!   [greater_than]).
//! - Unbounded values (see [unbounded]).
//! - Fixed values (see [zero] and [equal_to])
//! - Second order cone(s) (see [in_quadratic_cone], [in_quadratic_cones]): 
//!   ```math
//!   { x ∊ R^n | x_1^2 ≥ ‖ x_2^2 + ... + x_n^2 ‖^2, x_1 ≥ 0 }
//!   ```
//! - Rotated second order cone(s) (see [in_rotated_quadratic_cone], [in_rotated_quadratic_cones]): 
//!   ```math
//!   { x ∊ R^n | 1/2 x_1 x_2 ≥ ‖ x_3^2 + ... + x_n^2 ‖^2, x_1, x_2 ≥ 0 }
//!   ```
//! - Symmetric positive semidefinite cone(s) if dimension `n > 1` (see [in_psd_cone], [in_psd_cones]).
//! - Primal power cone(s) (see [in_power_cone], [in_power_cones]): 
//!   ```math
//!   { x ∊ R^n | x_1^(β_1) ··· x_k^(β_k) ≥ √(x_(k+1)^2 ··· x_n^2), x_0,..., x_k ≥ 0 }
//!   ```
//! - Dual power cone(s) (see [in_dual_power_cone], [in_dual_power_cones]): 
//!   ```math
//!   { x ∊ R^n | (x_1/β_1)^(β_1) ··· (x_k/β_k)^(β_k) ≥ √(x_(k+1)^2 ··· x_n^2), x_0,..., x_k ≥ 0 }
//!   ```
//! - Primal exponential cone(s) (see [in_exponential_cone], [in_exponential_cones]): 
//!   ```math
//!   { x ∊ R^3 | x_1 ≥ x_1 exp(x_3/x_2), x_0, x_1 ≥ 0 }
//!   ```
//! - Dual exponential cone(s) (see [in_dual_exponential_cone], [in_dual_exponential_cones]): 
//!   ```math
//!   { x ∊ R^3 | x_1 ≥ -x_3 exp(-1) exp(x_2/x_3), x_3 ≤ 0, x_1 ≥ 0 }
//!   ```
//! - Primal geometric mean cone(s) (see [in_geometric_mean_cone], [in_geometric_mean_cones]): 
//!   ```math
//!   { x ∊ R^n | (x_1 ··· x_(n-1))^{1/(n-1)} |x_n|, x_1,...,x_(n-1) ≥ 0}
//!   ```
//! - Dual geometric mean cone(s) (see [in_dual_geometric_mean_cone], [in_dual_geometric_mean_cones]): 
//!   ```math
//!   { x ∊ R^n | (n-1)(x_1 ··· x_(n-1))^{1/(n-1)} |x_n|, x_1,...,x_(n-1) ≥ 0}
//!   ```
//! - Scaled vectorized positive semidefinite cone(s) (see [in_svecpsd_cone], [in_svecpsd_cones]). For a `n` dimensional positive symmetric this
//!   is the scaled lower triangular part of the matrix in column-major format, i.e. 
//!   ```math
//!   { x ∊ R^(n(n+1)/2)} | sMat(x) ∊ S^n }
//!   ```
//!   where
//!   ```math
//!             │ x_1    x_2/√2   ···    x_n/√2         │
//!   sMat(x) = │ x_2/√2 x_n+1    ···    x_(2n-1)/√2    │
//!             │                 ···                   │
//!             │ x_n/√2 x_(2n-1)/√2 ... x_(n(n+1)/2)^2 │
//!   ```
//!
//! # Expressions and shapes
//!
//! Constraint, domains, variables and expressions have shapes, and the latter three can be either
//! dense or sparse meanning that some entries are fixed to zero. A shaped variable, expression and
//! constraint is basically a multi-dimensional array of scalar variables, expressions and
//! constraints respectively.
//!
//! ## Domains
//! 
//! A domain is a value that indicates things like cone type, cone parameters, right-hand sides,
//! shape and sparsity pattern. This is used when creating constraint or variables to define their
//! properties.
//!
//! ## Variables
//!
//! When a [Variable] is created in a model, the model adds the necessary
//! internal information to map a linear variable index to something in the underlying task, but
//! after that, a variable is essentially a list of indexes of the scalar variables t a shape and
//! sparsity. Variable objects can be stacked, indexed and sliced to obtain new variable objects.
//!
//! When a model has been optimized, the variable object is used to access the parts of the
//! solution it represents through the [Model] object.
//!
//! ## Constraints
//! A constraint is created in a [Model] from an expression (something implementing [ExprTrait])
//! and a domain. The sparsity pattern of the domain is ignored, and a constraint is always dense.
//! When a constraint has been created it can be indexed, sliced and stacked like a variable, and
//! it can be used to access the relevant parts of the solution through the [Model] object.
//!
//! ## Expression
//!
//! An expression is an n-dimensional array of scalar affine expressions. 
//!
//! # Note
//! Please note that the package is still somewhat exprimental.
//!
//! # Example
//!
//! ```
//! use mosekmodel::*;
//! use mosekmodel::expr::*;
//!
//! let a0 = vec![ 3.0, 1.0, 2.0, 0.0 ];
//! let a1 = vec![ 2.0, 1.0, 3.0, 1.0 ];
//! let a2 = vec![ 0.0, 2.0, 0.0, 3.0 ];
//! let c  = vec![ 3.0, 1.0, 5.0, 1.0 ];
//!
//! // Create a model with the name 'lo1'
//! let mut m = Model::new(Some("lo1"));
//! // Create variable 'x' of length 4
//! let x = m.variable(Some("x"), greater_than(vec![0.0,0.0,0.0,0.0]));
//!
//! // Create constraints
//! _ = m.constraint(None, &(&x).index(1), less_than(10.0));
//! _ = m.constraint(Some("c1"), &x.clone().dot(a0.as_slice()), equal_to(30.0));
//! _ = m.constraint(Some("c2"), &x.clone().dot(a1.as_slice()), greater_than(15.0));
//! _ = m.constraint(Some("c3"), &x.clone().dot(a2.as_slice()), less_than(25.0));
//!
//! // Set the objective function to (c^t * x)
//! m.objective(Some("obj"), Sense::Maximize, &x.clone().dot(c.as_slice()));
//!
//! // Solve the problem
//! m.solve();
//!
//! // Get the solution values
//! let (psta,dsta) = m.solution_status(SolutionType::Default);
//! println!("Status = {:?}/{:?}",psta,dsta);
//! let xx = m.primal_solution(SolutionType::Default,&x);
//! println!("x = {:?}", xx);
//! ```

extern crate mosek;
extern crate itertools;

mod utils;
pub mod variable;
pub mod domain;
pub mod matrix;
pub mod expr;
pub mod model;

use expr::workstack::WorkStack;

pub use model::{Sense,
                SolutionType,
                SolutionStatus,
                ModelItem,
                ModelItemIndex,
                Model,
                SolverParameterValue,
                Constraint,
                ConDomainTrait,
                VarDomainTrait};
pub use matrix::{Matrix,NDArray,};
pub use expr::{ExprTrait,
               ExprRightMultipliable,
               ModelExprIndex,
               IntoExpr,
               Expr};
pub use variable::Variable;
pub use domain::{LinearDomain,
                 ConicDomain,
                 LinearDomainType,
                 ConicDomainType,
                 LinearDomainOfsType,
                 PSDDomain,
                 unbounded,
                 less_than,
                 greater_than,
                 nonnegative,
                 nonpositive,
                 zero,
                 zeros,
                 equal_to,
                 in_quadratic_cone,
                 in_quadratic_cones,
                 in_svecpsd_cone,
                 in_svecpsd_cones,
                 in_rotated_quadratic_cone,
                 in_rotated_quadratic_cones,
                 in_geometric_mean_cone,
                 in_geometric_mean_cones,
                 in_dual_geometric_mean_cone,
                 in_dual_geometric_mean_cones,
                 in_exponential_cone,
                 in_exponential_cones,
                 in_dual_exponential_cone,
                 in_dual_exponential_cones,
                 in_power_cone,
                 in_power_cones,
                 in_dual_power_cone,
                 in_dual_power_cones,
                 in_psd_cone,
                 in_psd_cones };
