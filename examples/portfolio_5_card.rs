//
// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:     portfolio_5_card.rs 
//
//  Description :  Implements a basic portfolio optimization model
//                 with cardinality constraints on number of assets traded.
//

extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;


/// # Description
/// 
/// Extends the basic Markowitz model with cardinality constraints.
///
/// # Arguments
/// * `n` Number of assets
/// * `mu` An n dimensional vector of expected returns
/// * `GT` A matrix with n columns so (GT')*GT  = covariance matrix
/// * `x0` Initial holdings 
/// * `w` Initial cash holding
/// * `gamma` Maximum risk (=std. dev) accepted
/// * `k` Maximum number of assets on which we allow to change position.
/// 
/// # Returns
/// Optimal expected return and the optimal portfolio.
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn markowitz_with_cardinality(mu : &[f64],
                              GT : &NDArray<2>,
                              x0 : &[f64],
                              w  : f64,
                              gamma : f64,
                              K : usize) -> Vec<f64> {
    let n = GT.width();
    let m = GT.height();
    // Upper bound on the traded amount
    let w0 : f64 = w+x0.iter().sum::<f64>();
    let u = vec![w0;n];

    let mut model = Model::new(Some("Markowitz portfolio with cardinality bound"));
    // Defines the variables. No shortselling is allowed.
    let x = model.variable(Some("x"), greater_than(vec![0.0;n]));

    // Additional "helper" variables 
    let z = model.variable(Some("z"), unbounded().with_shape(&[n]));
    // Binary variables  - do we change position in assets
    let y = model.variable(Some("y"), greater_than(vec![0.0; n]).integer());
    _ = model.constraint(None, &y, less_than(vec![1.0; n]));

    //  Maximize expected return
    model.objective(Some("obj"), Sense::Maximize, &mu.dot(x.clone()));

    // The amount invested  must be identical to initial wealth
    _ = model.constraint(Some("budget"), &x.clone().sum(), equal_to(w+x0.iter().sum::<f64>()));

    // Imposes a bound on the risk
    _ = model.constraint(Some("risk"), &Expr::from(gamma).reshape(&[1]).vstack(GT.clone().mul(x.clone())), in_quadratic_cone(m+1));

    // z >= |x-x0| 
    _ = model.constraint(Some("buy"), &z.clone().sub(x.clone().sub(Expr::from(x0))), greater_than(vec![0.0; n]));
    _ = model.constraint(Some("sell"), &z.clone().sub(Expr::from(x0).sub(x.clone())), greater_than(vec![0.0; n]));

    // Constraints for turning y off and on. z-diag(u)*y<=0 i.e. z_j <= u_j*y_j
    _ = model.constraint(Some("y_on_off"), &z.clone().sub(Expr::from(y.clone()).mul_elem(u)), less_than(vec![0.0;n])); 

    // At most K assets change position
    _ = model.constraint(Some("cardinality"), &Expr::from(y.clone()).sum().sub(Expr::from(K as f64)), less_than(0.0));

    // Integer optimization problems can be very hard to solve so limiting the 
    // maximum amount of time is a valuable safe guard
    model.set_parameter("MSK_DPAR_MIO_MAX_TIME",180.0);

    // Solve multiple instances by varying the parameter K
    model.solve();

    model.primal_solution(SolutionType::Integer,&x).unwrap()
}

#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn main() {
    const n : usize = 8;
    const m : usize = 8;

    let w = 1.0;
    let mu = &[0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379];
    let x0 = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let GT = &[
        0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
        0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
        0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
        0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
        0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ] ;
    let gamma  = 0.25;

    let mut xsols : Vec<Vec<f64>> = Vec::new();
    for K in 1..n+1 {
        xsols.push(markowitz_with_cardinality(mu,&matrix::dense([n,m],GT.to_vec()),x0,w,gamma,K));
    }
    println!("\n-----------------------------------------------------------------------------------");
    println!("Markowitz portfolio optimization with cardinality constraints");
    println!("-----------------------------------------------------------------------------------\n");
    for (K,xsol) in xsols.iter().enumerate() {
        println!("Bound: {}   Expected return: {:.4}  Solution {:?}", 
                 K+1, 
                 mu.iter().zip(xsol.iter()).map(|(a,b)| a*b).sum::<f64>(), 
                 xsol.as_slice());
    }
}
