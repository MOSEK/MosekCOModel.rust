//!
//! # MosekModel library
//!
//! MosekModel is a crate for setting up conic optimization models to be solved with MOSEK. The
//! interface currently supports 
//! - Linear and conic variables and constraints
//! - Integer variables
//!
//! The model used is this:
//! $$
//! \begin{array}{ll}
//! \\mathrm{min/max}  & c^t x \\\\
//! \mathrm{such that} & A x + b \\in K_c\\\\
//!                    & X \\in K_x
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
//! - Scaled vectorized positive semidefinite cone(s) (see [in_svecpsd_cone], [in_svecpsd_cones]). For a `n` dimensional positive symmetric this
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
//! 
//! # Expressions and shapes
//!
//! The central trait for expressions is [ExprTrait], which all objects that must act as
//! expressions have to implement. For example, [Variable] and [NDArray] implement [ExprTrait].
//! [ExprTrait] also implements most of the functionality for creating new expressions. Note that
//! im most cases, expression objects *own* their data (for example the [expr::ExprAdd] object owns its
//! operands). This means that normally it will be necessary to clone a [Variable] object that is
//! used in an expression, if it is used in more than one place.
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
//! or variables to define their
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
//! An expression is an n-dimensional array of scalar affine expressions. Anything that implements
//! the trait [ExprTrait] can be used as an `N`-dimensional expression.
//!
//! # Note
//! Please note that the package is still somewhat exprimental.
//!
//! # Example
//!
//! ```
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
#![doc = include_str!("../js/mathjax.tag")]

extern crate mosek;
extern crate itertools;

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
                Model,
                SolverParameterValue,
                Constraint,
                ConDomainTrait,
                VarDomainTrait};
pub use matrix::{Matrix,NDArray,IntoIndexes};
pub use expr::{ExprTrait,
               ExprRightMultipliable,
               ExprLeftMultipliable,
               ModelExprIndex,
               IntoExpr,
               Expr,
               Dot,
               stack,hstack,vstack,stackvec,sumvec};
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
                 in_psd_cones 
                 };
pub use disjunction::{ClauseTrait,DisjunctionTrait,term};
