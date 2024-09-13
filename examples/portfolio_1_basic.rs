extern crate mosekmodel;

use mosekmodel::*;

/// Computes the optimal portfolio for a given risk
///
/// # Arguments
/// * `n`  Number of assets
/// * `mu` An n dimmensional vector of expected returns
/// * `gt` A matrix with n columns so (GT')*GT  = covariance matrix
/// * `x0` Initial holdings
/// * `w`  Initial cash holding
/// * `gamma` Maximum risk (=std. dev) accepted
fn basic_markowitz( n : usize,
                    mu : &[f64],
                    gt : &NDArray<2>,
                    x0 : &[f64],
                    w  : f64,
                    gamma : f64) -> f64 {
    let mut model = Model::new(Some("Basic Markowitz"));
    // Redirect log output from the solver to stdout for debugging.
    // if uncommented.
    model.set_log_handler(|msg| print!("{}",msg));

    // Defines the variables (holdings). Shortselling is not allowed.
    let x = model.variable(Some("x"), greater_than(vec![0.0;n]));

    //  Maximize expected return
    model.objective(Some("obj"), Sense::Maximize, &mu.dot(x.clone()));

    // The amount invested  must be identical to intial wealth
    model.constraint(Some("budget"), &x.clone().sum(), equal_to(w+x0.iter().sum::<f64>()));

    // Imposes a bound on the risk
    model.constraint(Some("risk"), 
                     &vstack![Expr::from(gamma).reshape(&[1]), 
                              gt.clone().mul(x.clone())], in_quadratic_cone(n+1));

    model.write_problem("portfolio-1.ptf");
    // Solves the model.
    model.solve();

    let xlvl = model.primal_solution(SolutionType::Default, &x).unwrap(); 
    mu.iter().zip(xlvl.iter()).map(|(&a,&b)| a*b).sum()
}

///  The example. Reads in data and solves the portfolio models.
#[test]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
fn main() {
    const n : usize   = 8;
    const w : f64     = 59.0;
    let mu            = [0.07197349, 0.15518171, 0.17535435, 0.0898094 , 0.42895777, 0.39291844, 0.32170722, 0.18378628];
    let x0            = [8.0, 5.0, 3.0, 5.0, 2.0, 9.0, 3.0, 6.0];
    let gammas        = [36.0];
    let GT            = matrix::dense([n,n],vec![
        0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
        0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
        0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
        0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
        0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
        0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ]);

    let expret : Vec<(f64,f64)> = gammas.iter().map(|&gamma| (gamma,basic_markowitz( n, &mu, &GT, &x0, w, gamma))).collect();
    println!("-----------------------------------------------------------------------------------");
    println!("Basic Markowitz portfolio optimization");
    println!("-----------------------------------------------------------------------------------");
    for (gamma,expret) in expret.iter() {
      println!("Expected return: {:.4e} Std. deviation: {:.4e}", expret, gamma);
    }
}

