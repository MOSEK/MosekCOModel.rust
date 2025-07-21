//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      `portfolio_4_transcost.rs`
//!
//!  Purpose :   Implements a basic portfolio optimization model
//!              with fixed setup costs and transaction costs
//!              as a mixed-integer problem.
//!

extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel_mosek::Model;

/// Extends the basic Markowitz model with a market cost term.
///
/// # Arguments
/// * `n` Number of assets
/// * `mu` An n dimensional vector of expected returns
/// * `GT` A matrix with n columns so (GT')*GT  = covariance matrix
/// * `x0` Initial holdings 
/// * `w` Initial cash holding
/// * `gamma` Maximum risk (=std. dev) accepted
/// * `f` If asset j is traded then a fixed cost f_j must be paid
/// * `g` If asset j is traded then a cost g_j must be paid for each unit traded
/// Output:
///    `(xsol,ysol,zsol)`, where 
///    * `xsol` is the amounts traded for each asset
///    * `ysol` are the binary variables indicating whether an asset is traded
///    * `zsol` are the transaction costs imposed on the trade
///    Optimal expected return and the optimal portfolio     
#[allow(non_snake_case)]
fn markowitz_with_transactions_cost( mu : &[f64],
                                     GT : &NDArray<2>,
                                     x0 : &[f64],
                                     w  : f64,
                                     gamma : f64,
                                     f : &[f64],
                                     g : &[f64]) -> (Vec<f64>,Vec<f64>,Vec<f64>) {
    let mut model = Model::new(Some("Markowitz portfolio with transaction costs"));
    let n = GT.width();
    // Upper bound on the traded amount
    let w0 = w+x0.iter().sum::<f64>();
    let u = vec![w0; n];
    let m = GT.height();

    // Defines the variables. No shortselling is allowed.
    let x = model.variable(Some("x"), greater_than(vec![0.0; n]));

    // Additional Some("helper") variables 
    let z = model.variable(Some("z"), unbounded().with_shape(&[n]));   
    // Binary variables
    let y = model.variable(Some("y"), greater_than(vec![0.0; n]).integer());
    _ = model.constraint(None, &y, less_than(vec![1.0; n]));

    //  Maximize expected return
    model.objective(Some("obj"), Sense::Maximize, mu.dot(&x));

    // Invest amount + transactions costs = initial wealth
    _ = model.constraint(Some("budget"), 
                        x.sum().add(f.dot(&y)).add(g.dot(&z)),
                        equal_to(w0));

    // Imposes a bound on the risk
    _ = model.constraint(Some("risk"), 
                         Expr::from(gamma).reshape(&[1])
                            .vstack( GT.mul(&x) ),
                            in_quadratic_cone());

    // z >= |x-x0| 
    _ = model.constraint(Some("buy"), z.sub(&x).sub(Expr::from(x0)), greater_than(vec![0.0;n]));
    _ = model.constraint(Some("sell"), z.sub(Expr::from(x0).sub(&x)), greater_than(vec![0.0; n]));
    // Alternatively, formulate the two constraints as
    //model.constraint(Some("trade"), Expr.hstack(z,Expr.sub(x,x0)), Domain.inQcone())

    // Constraints for turning y off and on. z-diag(u)*y<=0 i.e. z_j <= u_j*y_j
    _ = model.constraint(Some("y_on_off"), z.sub(y.mul_elem(u)), less_than(vec![0.0;n]));

    // Integer optimization problems can be very hard to solve so limiting the 
    // maximum amount of time is a valuable safe guard
    model.set_parameter("MSK_DPAR_MIO_MAX_TIME", 180.0); 
    model.solve();

    ( model.primal_solution(SolutionType::Integer, &x).unwrap(),
      model.primal_solution(SolutionType::Integer, &y).unwrap(),
      model.primal_solution(SolutionType::Integer, &z).unwrap() )
}

#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
fn main() {
    const n : usize = 8;
    const m : usize = 8;
    let w = 1.0;
    let mu = &[0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379];
    let x0 = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let GT_data : &[f64]= &[
        0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
        0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
        0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
        0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
        0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ];
    let GT = matrix::dense([n, m], GT_data.to_vec());

    let f = &[0.01; n];
    let g = &[0.001; n];
    let gamma = 0.36;
    let (xsol, _, zsol) = markowitz_with_transactions_cost(mu,&GT,x0,w,gamma,f,g);
    println!("\n-----------------------------------------------------------------------------------");
    println!("Markowitz portfolio optimization with transactions cost");
    println!("-----------------------------------------------------------------------------------\n");
    println!("Expected return: {:.4e} Std. deviation: {:.4e} Transactions cost: {:.4e}",
             mu.iter().zip(xsol.iter()).map(|(&a,&b)| a*b).sum::<f64>(),
             gamma,
             f.iter().zip(zsol.iter()).map(|(&a,&b)| a*b).sum::<f64>()+g.iter().zip(zsol.iter()).map(|(&a,&b)| a*b).sum::<f64>());
}
#[test]
fn test() { main() }
