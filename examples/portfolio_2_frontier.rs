extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;


/// Computes several portfolios on the optimal portfolios by
///
/// ```
/// for alpha in alphas:
///     maximize   expected return - alpha * variance
///     subject to the constraints
/// ```
/// 
/// # Arguments
/// - `n`: Number of assets
/// - `mu`: An n dimmensional vector of expected returns
/// - `GT`: A matrix with n columns so (GT')*GT  = covariance matrix
/// - `x0`: Initial holdings
/// - `w`: Initial cash holding
/// - `alphas`: List of the alphas
/// # Returns
/// The efficient frontier as list of tuples (alpha, expected return, variance)
#[allow(non_snake_case)]
fn efficient_frontier( n : usize,
                       mu : &[f64],
                       GT : &NDArray<2>,
                       x0 : &[f64],
                       w  : f64,
                       alphas : &[f64]) -> Vec<(f64,f64,f64)> {
    let mut model = Model::new(Some("Efficient frontier"));

    // Defines the variables (holdings). Shortselling is not allowed.
    let x = model.variable(Some("x"), greater_than(vec![0.0; n])); // Portfolio variables
    let s = model.variable(Some("s"), &[]);  // Variance variable

    model.constraint(Some("budget"), &x.clone().sum(), equal_to(w + x0.iter().sum::<f64>()));

    // Computes the risk
    model.constraint(Some("variance"), 
                     &vstack![s.clone().flatten(), 
                              (0.5).into_expr().flatten(), 
                              GT.clone().mul(x.clone())], in_rotated_quadratic_cone(n+2));

    // Solve the problem for many values of parameter alpha

    alphas.iter().map(|&alpha| {
        //  Define objective as a weighted combination of return and variance
        model.objective(Some("obj"), Sense::Maximize, & mu.dot(x.clone()).sub(s.clone().mul(alpha)));
        model.solve();

        (alpha,
         model.primal_solution(SolutionType::Default, &x).unwrap().iter().zip(mu.iter()).map(|(&a,&b)| a*b).sum::<f64>(),
         model.primal_solution(SolutionType::Default, &s).unwrap()[0])
    }).collect()
}

/// The example. Reads in data and solves the portfolio models.
#[allow(non_snake_case)]
fn main() {
    let n : usize = 8;
    let w  = 1.0;
    let mu = [0.07197, 0.15518, 0.17535, 0.08981, 0.42896, 0.39292, 0.32171, 0.18379];
    let x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let GT = matrix::dense([n,n],vec![
      0.30758, 0.12146, 0.11341, 0.11327, 0.17625, 0.11973, 0.10435, 0.10638,
      0.     , 0.25042, 0.09946, 0.09164, 0.06692, 0.08706, 0.09173, 0.08506,
      0.     , 0.     , 0.19914, 0.05867, 0.06453, 0.07367, 0.06468, 0.01914,
      0.     , 0.     , 0.     , 0.20876, 0.04933, 0.03651, 0.09381, 0.07742,
      0.     , 0.     , 0.     , 0.     , 0.36096, 0.12574, 0.10157, 0.0571 ,
      0.     , 0.     , 0.     , 0.     , 0.     , 0.21552, 0.05663, 0.06187,
      0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.22514, 0.03327,
      0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.2202 ]);

    // Some predefined alphas are chosen
    let alphas = [ 0.0, 0.01, 0.1, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0 ];

    let res = efficient_frontier(n, &mu, &GT, &x0, w, &alphas);
    println!("\n-----------------------------------------------------------------------------------");
    println!("Efficient frontier") ;
    println!("-------------------------------------------------------------------------------------\n");
    println!("{:-12}  {:-12}  {:-12}", "alpha", "return", "std. dev.");
   
    for (alpha,fmux,s) in res.iter() {
      println!("\t{:-12.4}  {:-12.4e}  {:-12.4e}", alpha, fmux, s.sqrt());
    }
}
