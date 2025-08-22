//!
//! Copyright: © Mosek ApS, 2024
//!
//! Purpose: Demonstrates how to solve the problem
//! ```
//! minimize x1 + x2
//! such that
//!          x1 + x2 + x3  = 1.0
//!              x1,x2    >= 0.0
//! and      x1 >= x2 * exp(x3/x2)
//! ```
extern crate mosekcomodel;
use itertools::Itertools;
use mosekcomodel::*;
use mosekcomodel_optserver::{Model,SolverAddress};

fn ceo1(address : &str) -> (SolutionStatus,SolutionStatus,Result<Vec<f64>,String>) 
{
    let mut m = Model::new(Some("ceo1"));
    m.set_parameter((), SolverAddress(address.to_string()));

    let x = m.variable(Some("x"), unbounded().with_shape(&[3]));

    // Create the constraint
    //      x[0] + x[1] + x[2] = 1.0
    _ = m.constraint(Some("lc"), x.sum(), equal_to(1.0));

    // Create the exponential conic constraint
    let expc = m.constraint(Some("expc"), &x, in_exponential_cone());

    // Set the objective function to (x[0] + x[1])
    m.objective(Some("obj"), Sense::Minimize, x.index(0..2).sum());

    m.write_problem("ceo1.jtask");
    // Solve the problem
    m.solve();

    let (psta,dsta) = m.solution_status(SolutionType::Default);

    // Get the linear solution values
    let solx = m.primal_solution(SolutionType::Default, &x).unwrap();
    println!("x1,x2,x3 = {}, {}, {}", solx[0], solx[1], solx[2]);

    // Get conic solution of expc
    let  expclvl = m.primal_solution(SolutionType::Default, &expc).unwrap();
    let  expcsn  = m.dual_solution(SolutionType::Default, &expc).unwrap();
    
    println!("expc levels = {:?}", expclvl);

    println!("expc dual conic var levels = {:?}", expcsn);

    (psta,dsta,Ok(solx))
}
//TAG:end-ceo1

fn main() {
    let mut args = std::env::args().get(1..);
    let address = args.next().unwrap_or("http://solve.mosek.com:30080".to_string());

    let (psta,dsta,xx) = ceo1(address.as_str());
    println!("Status = {:?}/{:?}",psta,dsta);
    println!("x = {:?}", xx);
}

#[test]
fn test() { 
    let (psta,dsta,sol) = ceo1("http://solve.mosek.com:30080"); 
    let xx = sol.unwrap();
    assert!(matches!(psta, SolutionStatus::Optimal));
    assert!(matches!(dsta, SolutionStatus::Optimal));
    assert!((xx[0]+xx[1]+xx[2]-1.0).abs() < 1e-7);

}
