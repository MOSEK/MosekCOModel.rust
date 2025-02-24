//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!  File:      logistic.rs
//!
//!  Purpose: Implements logistic regression with regulatization.
//!
//!           Demonstrates using the exponential cone and log-sum-exp in Fusion.

extern crate mosekmodel;

use expr::const_expr;
use itertools::iproduct;
use mosekmodel::*;


fn softplus<E2>(model : & mut Model, n : usize, t : &Variable<1>, u : E2) where E2 : ExprTrait<1> {
    let z1 = model.variable(None,&[n,1]);
    let z2 = model.variable(None,&[n,1]);
    model.constraint(None, &z1.clone().add(z2.clone()), equal_to(1.0).with_shape(&[n,1]));
    model.constraint(None, &hstack![z1.clone(), const_expr(&[n,1],1.0), u.reshape(&[n,1]).sub(t.clone().reshape(&[n,1]))], in_exponential_cones(&[n,3],1));
    model.constraint(None, &hstack![z2.clone(), const_expr(&[n,1],1.0), t.clone().reshape(&[n,1]).neg()], in_exponential_cones(&[n,3],1));
}

/// Model logistic regression (regularized with full 2-norm of theta)
///
/// # Arguments
///
/// - `X` n x d matrix of data points
/// - `Y` length n vector classifying training points
/// - `lamb` regularization parameter
#[allow(non_snake_case)]
fn logistic_regression(X : NDArray<2>, 
                       Y : &[bool],
                       lamb : f64) -> (Model,Variable<1>)
{
    let n = X.shape()[0];
    let d = X.shape()[1];

    let mut model = Model::new(None);

    let theta = model.variable(Some("theta"), d);
    let t     = model.variable(None,n);
    let reg   = model.variable(None,&[]);

    model.objective(None,Sense::Minimize, &t.clone().sum().add(reg.clone().mul(lamb)));
    model.constraint(None,&vstack!(reg.clone().with_shape(&[1]), theta.clone()), in_quadratic_cone(d+1));

    let signs : Vec<f64> = Y.iter().map(|&yi| if yi { -1.0 } else { 1.0 }).collect();

    softplus(& mut model, n, &t, X.mul(theta.clone()).mul_elem(signs));

    (model,theta)
}

#[allow(non_snake_case)]
fn main() {
    // Test: detect and approximate a circle using degree 2 polynomials
    let n = 30;

    let Y : Vec<bool> = iproduct!(0..n,0..n).map(|(i,j)| {
      let x = -1.0 + 2.0*i as f64/(n as f64-1.0);
      let y = -1.0 + 2.0*j as f64/(n as f64-1.0);
      x*x+y*y >= 0.69
    }).collect();

    let X = NDArray::new([n*n,6],None,
                         iproduct!(0..n,0..n).map(|(i,j)| {
                              let x = -1.0 + 2.0*i as f64/(n as f64-1.0);
                              let y = -1.0 + 2.0*j as f64/(n as f64-1.0);
                              [ 1.0, x, y, x*y, x*x, y*y]})
                             .flatten()
                             .collect::<Vec<f64>>()).unwrap();

    let (mut model,theta) = logistic_regression(X, &Y, 0.1);

    model.set_log_handler(|msg| print!("{}",msg));
    model.solve();
    
    let xx = model.primal_solution(SolutionType::Default, &theta).unwrap();
    println!("theta = {:?}",&xx);
}

#[test]
fn test() { main() }
