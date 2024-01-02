//
// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:      portfolio_3_impact.rs
//
// Purpose :   Implements a basic portfolio optimization model
//             with x^(3/2) market impact costs.
//

extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;

/// Extends the basic Markowitz model with a market cost term.
///
/// // Arguments
///
/// * `n` Number of assets
/// * `mu` An n dimensional vector of expected returns
/// * `gt` A matrix with n columns so (GT')*GT  = covariance matrix
/// * `x0` Initial holdings 
/// * `w` Initial cash holding
/// * `gamma` Maximum risk (=std. dev) accepted
/// * `m` It is assumed that  market impact cost for the j'th asset is `|m_j|x_j-x0_j|^3/2`
///
/// // Returns 
/// Optimal expected return and the optimal portfolio     
fn markowitz_impact(n : usize,
                    mu : &[f64],
                    gt : &DenseMatrix,
                    x0 : &[f64],
                    w : f64,
                    gamma : f64,
                    m : &[f64]) -> (Vec<f64>,Vec<f64>) {
    let mut model = Model::new(Some("Markowitz portfolio with market impact"));
    // Redirect log output from the solver to stdout for debugging.
    // if uncommented.
    model.set_log_handler(|msg| print!("{}",msg));
    
    // Defines the variables. No shortselling is allowed.
    let x = model.variable(Some("x"), greater_than(vec![0.0; n]));
    
    // Variables computing market impact 
    let t = model.variable(Some("t"), n);

    // Maximize expected return
    model.objective(Some("obj"), Sense::Maximize, &mu.dot(x.clone()));

    // Invested amount + slippage cost = initial wealth
    model.constraint(Some("budget"), &x.clone().sum().add(m.dot(t.clone())), equal_to(w+x0.iter().sum::<f64>()));

    // Imposes a bound on the risk
    model.constraint(Some("risk"), 
                     &vstack![gamma.into_expr().reshape(&[1]), 
                              gt.clone().mul(x.clone())], in_quadratic_cone(n+1));

    // t >= |x-x0|^1.5 using a power cone
    model.constraint(Some("tz"), 
                     &hstack![ t.clone().into_expr().reshape(&[n,1]),
                               vec![1.0;n].into_expr().reshape(&[n,1]),
                               x.clone().sub(x0).reshape(&[n,1]) ],
                     in_power_cones(&[n,3],1,&[2.0/3.0,1.0/3.0]));

    model.solve();

    (model.primal_solution(SolutionType::Default,&x).unwrap(), 
     model.primal_solution(SolutionType::Default,&t).unwrap())
}

#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn main() {
    const n : usize = 8;
    let w = 1.0;
    let mu = [0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379];
    let x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let GT = matrix::dense(n,n,vec![
        0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
        0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
        0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
        0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
        0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ]);
                  
    // Somewhat arbitrary choice of m
    let gamma = 0.36;
    let m = [0.01; n];
    let (xsol, tsol) = markowitz_impact(n,&mu,&GT,&x0,w,gamma,&m);
    println!("\n-----------------------------------------------------------------------------------");
    println!("Markowitz portfolio optimization with market impact cost");
    println!("-----------------------------------------------------------------------------------\n");
    println!("Expected return: {:.4e} Std. deviation: {:.4e} Market impact cost: {:.4e}", 
             mu.iter().zip(xsol.iter()).map(|(&m,&z)| m*z).sum::<f64>(),
             gamma,
             m.iter().zip(tsol.iter()).map(|(&m,&t)| m*t).sum::<f64>());
    println!("Optimal portfolio: {:?}", xsol);
}
