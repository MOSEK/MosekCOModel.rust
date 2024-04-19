extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;

//
// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:      portfolio_6_factor.rs
//
//  Description :  Implements a basic portfolio optimization model
//                 with factor structured covariance matrix.
//


// Description:
//     Extends the basic Markowitz model with factor structure in the covariance matrix.
//
// Input:
//     n: Number of securities
//     mu: An n dimensional vector of expected returns
//     G_factor_T: The factor (dense) part of the factorized risk
//     theta: specific risk vector
//     x0: Initial holdings 
//     w: Initial cash holding
//     gamma: Maximum risk (=std. dev) accepted
//
// Output:
//    Optimal expected return and the optimal portfolio     
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn factor_model_markowitz(n : usize, 
                          mu : &[f64],
                          G_factor_T : &DenseMatrix, 
                          S_theta : &[f64], 
                          x0 : &[f64], 
                          w : f64,
                          gamma : &[f64]) -> (f64,Vec<f64>)  {
    let mut model = Model::new(Some("Basic Markowitz"));

    // Variables 
    // The variable x is the fraction of holdings in each security. 
    // It is restricted to be positive, which imposes the constraint of no short-selling.   
    x = M.variable("x", n, Domain.greaterThan(0.0))

    // Objective (quadratic utility version)
    M.objective('obj', ObjectiveSense.Maximize, Expr.dot(mu, x))
   
    // Budget constraint
    M.constraint('budget', Expr.sum(x), Domain.equalsTo(w + sum(x0)))

    // Conic constraint for the portfolio std. dev
    M.constraint('risk', Expr.vstack([Expr.constTerm(gamma),
                                      Expr.mul(G_factor_T, x), 
                                      Expr.mulElm(np.sqrt(theta), x)]), Domain.inQCone())

    // Solve optimization
    M.solve()
    
    return mu @ x.level(), x.level()
}
  
fn cholesky(m : &DenseMatrix) -> DenseMatrix {
    let n = m.width();
    if n != m.height() {
        panic!("Cholesky requires a square matrix");
    }
    let mt = m.transpose();
    let mut mt_data = mt.data().to_vec();

    mosek::potrf(mosek::Uplo::LO, n.try_into().unwrap(), mt_data.as_mut_slice()).unwrap();

    double[] vecs = mat_to_vec_c(m);
    LinAlg.potrf(mosek.uplo.lo, n, vecs);
    double[][] s = vec_to_mat_c(vecs, n, n);

    // Zero out upper triangular part (LinAlg.potrf does not use it, original matrix values remain there)
    for (int i = 0; i < n; ++i) 
    {
      for (int j = i+1; j < n; ++j) 
      {
        s[i][j] = 0.0;
      }
    }
    return s;
}


#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn main() {
    const n : usize = 8;
    let w = 1.0;
    let mu = &[0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379];
    let x0 = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let B = matrix::dense(8,2,vec![ 0.4256,  0.1869,
                                 0.2413,  0.3877,
                                 0.2235,  0.3697,
                                 0.1503,  0.4612,
                                 1.5325, -0.2633,
                                 1.2741, -0.2613,
                                 0.6939,  0.2372,
                                 0.5425,  0.2116 ]);
    let S_F = matrix::dense(2,2,vec![0.0620, 0.0577, 0.0577, 0.0908]);
    let theta = &[0.0720, 0.0508, 0.0377, 0.0394, 0.0663, 0.0224, 0.0417, 0.0459];
    let P = cholesky(S_F);
    let G_factor = matrix_mul(B, P);  
    let G_factor_T = transpose(G_factor);

    let gammas = &[0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48] ;

    println!("\n-----------------------------------------------------------------------------------")
    println!("Markowitz portfolio optimization with factor model")
    println!("-----------------------------------------------------------------------------------\n")

    for gamma in gammas.iter() {
        let (er, x) = factor_model_markowitz(n, mu, G_factor_T, theta, x0, w, gamma);
        println!("Expected return: {.4e} Std. deviation: {:.4e}" % (er, gamma));
        println!("Optimal portfolio: {:?}",x);
    }
}
