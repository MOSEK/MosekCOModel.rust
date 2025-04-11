//!
//! # MosekModel library
//!
//! MosekModel is a crate for setting up conic optimization models to be solved with
//! [MOSEK](https://mosek.com) via [mosek.rust](https://crates.io/crates/mosek). The
//! interface currently supports 
//! - Linear and conic variables and constraints
//! - Integer variables
//! - Disjunctive constraints
//!
//! The [Model] object encapsulates a model of the form
//! $$
//! \begin{array}{ll}
//! \\mathrm{min/max}     & c^t x \\\\
//! \\mathrm{such\\ that} & A x + b \\in K_c\\\\
//!                       & X \\in K_x
//! \\end{array}
//! $$
//!
//! where \\(K_c = K_{c_0}\\times\\cdots\\times K_{c_m}\\), \\(K_x = K_{x_0} \\times\\cdots\\times K_{x_n}\\), each \\(K_{c_i}\\) and \\(K_{x_i}\\) is a conic
//! domain from the currently supported set plus an offset:
//! - Non-positive or non-negative orthant (see [nonpositive], [nonnegative], [less_than] and
//!   [greater_than]).
//! - Unbounded values (see [unbounded]).
//! - Fixed values (see [zero] and [equal_to])
//! - Second order cone(s) (see [in_quadratic_cone], [in_quadratic_cones]): 
//!   $$
//!   \\left\\{ x \\in R^n | x_1^2 \\geq \\left\\Vert x_2^2 + \\cdots + x_n^2 \\right\\Vert^2, x₁ \\geq 0 \\right\\}
//!   $$
//! - Rotated second order cone(s) (see [in_rotated_quadratic_cone], [in_rotated_quadratic_cones]): 
//!   $$
//!   \\left\\{ x \in R^n | \\frac{1}{2} x_1 x_2 \geq \\left\\Vert x_3^2 + \\cdots + x_n^2 \\right\\Vert^2, x_1, x_2 \\geq 0 \\right\\}
//!   $$
//!<!-- - Symmetric positive semidefinite cone(s) if dimension `n > 1` (see [in_psd_cone], [in_psd_cones]).-->
//! - Primal power cone(s) (see [in_power_cone], [in_power_cones]): 
//!   $$
//!   \\left\\{ x \\in R^n | x_2^{\\beta_1} \\cdots x_k^{\\beta_k} \\geq \\sqrt{x_{k+1}^2 \\cdots x_n^2}, x_0,\\ldots, x_k \geq 0 \\right\\}
//!   $$
//! - Dual power cone(s) (see [in_dual_power_cone], [in_dual_power_cones]): 
//!   $$
//!   \\left\\{ x \\in R^n | (x_1/\\beta_1)^{\\beta_1} \\cdots (x_k)^{\\beta_k} \geq \\sqrt{x_{k+1}^2 \\cdots x_n^2}, x_0,\\ldots, x_k \\geq 0 \\right\\}
//!   $$
//! - Primal exponential cone(s) (see [in_exponential_cone], [in_exponential_cones]): 
//!   $$
//!   \\left\\{ x \\in R^3 | x_1 \\geq x_1 e^{x_3/x_2}, x_0, x_1 \geq 0 \\right\\}
//!   $$
//! - Dual exponential cone(s) (see [in_dual_exponential_cone], [in_dual_exponential_cones]): 
//!   $$
//!   \\left\\{ x \\in R^3 | x_1 \\geq -x_3 e^{-1} e^{x_2/x_3}, x_3 \\geq 0, x_1 \\geq 0 \\right\\}
//!   $$
//! - Primal geometric mean cone(s) (see [in_geometric_mean_cone], [in_geometric_mean_cones]): 
//!   $$
//!   \\left\\{ x \\in R^n| (x_1\\cdots x_{n-1})^{1/(n-1)} |x_n|, x_1,\\ldots,x_{n-1} \\geq 0\\right\\}
//!   $$
//! - Dual geometric mean cone(s) (see [in_dual_geometric_mean_cone], [in_dual_geometric_mean_cones]): 
//!   $$
//!   \\left\\{ x \\in R^n | (n-1)(x_1 \\cdots x_{n-1})^{1/(n-1)} |x_n|, x_1,\\ldots,x_{n-1} \\geq 0\\right\\}
//!   $$
//! - Scaled vectorized positive semidefinite cone(s) (see [in_svecpsd_cone], [in_svecpsd_cones]). For a `n` dimensional positive symmetric matrix this
//!   is the scaled lower triangular part of the matrix in column-major format, i.e. 
//!   $$
//!   \\left\\{ x \\in R^{n(n+1)/2} | \\mathrm{sMat}(x) \\in S_+^n \\right\\}
//!   $$
//!   where
//!   $$
//!   \\mathrm{sMat}(x) = \\left[ \\begin{array}{cccc} 
//!     x_1            & x_2/\\sqrt{2} & \\cdots & x_n/\\sqrt{2}      \\\\
//!     x_2/\\sqrt{2}  & x_n+1         & \\cdots & x_{2n-1}/\\sqrt{2} \\\\
//!                    &               & \\cdots &                    \\\\
//!     x_n/\\sqrt{2}  & x_{2n-1}/\\sqrt{2} & \\cdots & x_{n(n+1_/2}^2
//!   \\end{array} \\right]
//!   $$
//! - Symmetric positive semidefinite cones
//!   $$
//!   X \\in \\mathcal{S}^n_+
//!   $$
//! 
//! # Expressions and shapes
//!
//! The central traits for expressions are [ExprTrait], which all objects that must act as
//! expressions have to implement, and [IntoExpr] which is anything that can be turned into an
//! [ExprTrait]. For example, [Variable] `Vec<f64>`, `f64` and [NDArray] implement [IntoExpr] since
//! they can be turned into a expression, while the various expression structs (e.g. [Expr],
//! [expr::ExprAdd], [expr::ExprStack] etc.) implement [ExprTrait]. Note that expressions are *consumed* when
//! creating new expressions, while variables and constants can be passed by reference and will be
//! cloned. This is because the expression constructing functions accept [IntoExpr]s rather than
//! [ExprTrait]s. For example, an `add` function might look like this:
//! ```
//! use mosekcomodel::{ExprTrait,IntoExpr};
//!
//! struct ExprAdd<const N : usize,E1,E2> 
//!     where 
//!         E1 : ExprTrait<N>,
//!         E2 : ExprTrait<N>
//! {
//!     e1 : E1,
//!     e2 : E2
//! }
//! fn add<const N : usize, E1,E2>( e1 : E1, e2 : E2 ) -> ExprAdd<N,E1::Result,E2::Result> 
//!     where E1 : IntoExpr<N>,
//!           E2 : IntoExpr<N>
//! {
//!     ExprAdd{
//!         e1 : e1.into_expr(),
//!         e2 : e2.into_expr()
//!     }
//! }
//! ```
//! Now, when [IntoExpr] is implemented for all [ExprTrait], as well as for both `&Variable<N>` and
//! for `Variable<N>`, we can pass an expression, a variable or a variable reference to the
//! function.
//!
//! Constraint, domains, variables and expressions have shapes, and the latter three can be either
//! dense or sparse meanning that some entries are fixed to zero. A shaped variable, expression and
//! constraint is basically a multi-dimensional array of scalar variables, affine expressions and
//! scalar constraints respectively.
//!
//! When reshaping an object it is important to understand the order of the scalar elements in the
//! multi-dimensional array. In `MosekModel` everything is in row-major order, i.e. for a
//! two-dimensional array, where the first dimension is the height and the second is the width
//! $$
//! \\left[\\begin{array}{cc} a & b \\\\ c & d \\end{array}\\right]
//! $$
//! the elements are ordered as `[a,b,c,d]`. More generally, elements are ordered by inner
//! dimension first.
//!
//! A scalar is an n-dimensional object with `n=0`.
//!
//! In `MosekModel` the dimension of variables, constraints, expressions and domains are part of
//! the object type, so only correct combinations of dimensionality are allowed. The actual shapes
//! still have to be checked at runtime.
//!
//! ## Domains
//! 
//! A domain is an object that indicates things like type (cone type or domain type), cone
//! parameters, right-hand sides, shape and sparsity pattern. This is used when creating constraint
//! or variables to define their properties. Normally, the domain is not created directly, but
//! rather an object implementing [IntoDomain] (for variables) or [IntoShapedDomain] (for
//! constraints) is created and passed to the relevant function that will then validate it. Such an
//! object may be incomplete and even inconsistent. It is completed and validated when used to
//! create varibles and constraints.
//!
//! ## Constraints and Variables
//! 
//! Variables and constraints are created through the [Model] object. Functions creating variables
//! and constraints have two versions, a `try_` and a plain version. The former will return a
//! [Result::Err] whenever an error was encountered that left the [Model] in a consistent state.
//! The latter version will `panic!` on any error.
//!
//! ### Variables
//!
//! When a [Variable] is created in a model as [Model::variable] or [Model::ranged_variable], the
//! model adds the necessary
//! internal information to map a linear variable index to something in the underlying task, but
//! after that, a variable is essentially a list of indexes of the scalar variables t a shape and
//! sparsity. Variable objects can be stacked, indexed and sliced to obtain new variable objects.
//!
//! When a model has been optimized, the variable object is used to access the parts of the
//! solution it represents through the [Model] object.
//!
//! ### Constraints
//! A constraint is created in a [Model] from an expression (something implementing [ExprTrait])
//! and a domain using [Model::constraint] or [Model::ranged_constraint]. The sparsity pattern of
//! the domain is ignored, and a constraint is always dense.
//! When a constraint has been created it can be indexed, sliced and stacked like a variable, and
//! it can be used to access the relevant parts of the solution through the [Model] object.
//!
//! ## Expression
//!
//! An expression is an n-dimensional array of scalar affine expressions. Anything that implements
//! the trait [ExprTrait] can be used as an `N`-dimensional expression.
//!
//! # Note
//! Please note that the package is still somewhat exprimental.
//!
//! # Example: `lo1`
//!
//! Simple linear example:
//!
//! ```rust
//! // Importing everything from mosekcomodel provides all basic functionality.
//! use mosekcomodel::*;
//!
//! let a0 = vec![ 3.0, 1.0, 2.0, 0.0 ];
//! let a1 = vec![ 2.0, 1.0, 3.0, 1.0 ];
//! let a2 = vec![ 0.0, 2.0, 0.0, 3.0 ];
//! let c  = vec![ 3.0, 1.0, 5.0, 1.0 ];
//!
//! // Create a model with the name 'lo1'
//! let mut m = Model::new(Some("lo1"));
//! // Redirect log output from the solver to stdout for debugging.
//! // if uncommented.
//! m.set_log_handler(|msg| print!("{}",msg));
//! // Create variable 'x' of length 4
//! let x = m.variable(Some("x"), greater_than(vec![0.0,0.0,0.0,0.0]));
//!
//! // Create constraints
//! _ = m.constraint(None,       x.index(1),           less_than(10.0));
//! _ = m.constraint(Some("c1"), x.dot(a0.as_slice()), equal_to(30.0));
//! _ = m.constraint(Some("c2"), x.dot(a1.as_slice()), greater_than(15.0));
//! _ = m.constraint(Some("c3"), x.dot(a2.as_slice()), less_than(25.0));
//!
//! // Set the objective function to (c^t * x)
//! m.objective(Some("obj"), Sense::Maximize, x.dot(c.as_slice()));
//!
//! m.write_problem("lo1.ptf");
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
//!
//! # Example: `portfolio_1_basic`
//! 
//! Example using second order cones to model risk in a basic portfolio model.
//!
//! ```rust
//! use mosekcomodel::*;
//! 
//! // Computes the optimal portfolio for a given risk
//! //
//! // # Arguments
//! // * `n`  Number of assets
//! // * `mu` An n dimmensional vector of expected returns
//! // * `gt` A matrix with n columns so (GT')*GT  = covariance matrix
//! // * `x0` Initial holdings
//! // * `w`  Initial cash holding
//! // * `gamma` Maximum risk (=std. dev) accepted
//! fn basic_markowitz( n : usize,
//!                     mu : &[f64],
//!                     gt : &NDArray<2>,
//!                     x0 : &[f64],
//!                     w  : f64,
//!                     gamma : f64) -> f64 {
//!     let mut model = Model::new(Some("Basic Markowitz"));
//!     // Redirect log output from the solver to stdout for debugging.
//!     // if uncommented.
//!     model.set_log_handler(|msg| print!("{}",msg));
//! 
//!     // Defines the variables (holdings). Shortselling is not allowed.
//!     let x = model.variable(Some("x"), greater_than(vec![0.0;n]));
//! 
//!     //  Maximize expected return
//!     model.objective(Some("obj"), Sense::Maximize, x.dot(mu));
//! 
//!     // The amount invested  must be identical to intial wealth
//!     model.constraint(Some("budget"), x.sum(), equal_to(w+x0.iter().sum::<f64>()));
//! 
//!     // Imposes a bound on the risk
//!     model.constraint(Some("risk"), 
//!                      vstack![Expr::from(gamma).reshape(&[1]), 
//!                              gt.mul(&x)], in_quadratic_cone());
//! 
//!     model.write_problem("portfolio-1.ptf");
//!     // Solves the model.
//!     model.solve();
//! 
//!     let xlvl = model.primal_solution(SolutionType::Default, &x).unwrap(); 
//!     mu.iter().zip(xlvl.iter()).map(|(&a,&b)| a*b).sum()
//! }
//! 
//! const N : usize   = 8;
//! const W : f64     = 59.0;
//! let mu            = [0.07197349, 0.15518171, 0.17535435, 0.0898094 , 0.42895777, 0.39291844, 0.32170722, 0.18378628];
//! let x0            = [8.0, 5.0, 3.0, 5.0, 2.0, 9.0, 3.0, 6.0];
//! let gammas        = [36.0];
//! let GT            = matrix::dense([N,N],vec![
//!     0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
//!     0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
//!     0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
//!     0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
//!     0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
//!     0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
//!     0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
//!     0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ]);
//! 
//! let expret : Vec<(f64,f64)> = gammas.iter().map(|&gamma| (gamma,basic_markowitz( N, &mu, &GT, &x0, W, gamma))).collect();
//! println!("-----------------------------------------------------------------------------------");
//! println!("Basic Markowitz portfolio optimization");
//! println!("-----------------------------------------------------------------------------------");
//! for (gamma,expret) in expret.iter() {
//!   println!("Expected return: {:.4e} Std. deviation: {:.4e}", expret, gamma);
//! }
//! ```
#![doc = include_str!("../js/mathjax.tag")]

extern crate mosek;
extern crate itertools;

//pub mod solver;
pub mod constraint;
pub mod variable;
pub mod domain;
pub mod matrix;
pub mod expr;
pub mod model;
pub mod disjunction;
pub mod experimental;
pub mod utils;

use expr::workstack::WorkStack;

pub use model::{Sense,
                SolutionType,
                SolutionStatus,
                ModelItem,
                ModelItemIndex,
                VarDomainTrait,
                Model,
                SolverParameterValue,
                constraint};
pub use matrix::{Matrix,NDArray,IntoIndexes};
pub use expr::{ExprTrait,
               ExprRightMultipliable,
               ExprLeftMultipliable,
               ModelExprIndex,
               IntoExpr,
               Expr,
               RightDottable,
               stack,hstack,vstack,stackvec,sumvec};
pub use constraint::{Constraint,ConstraintDomain};
pub use variable::Variable;
pub use domain::{IntoDomain,
                 IntoShapedDomain,
                 LinearDomain,
                 ConicDomain,
                 PSDDomain,
                 LinearDomainType,
                 ConicDomainType,
                 OffsetTrait,
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
                 in_psd_cones,
                 in_range,
                 };
pub use disjunction::{ConjunctionTrait,DisjunctionTrait};
