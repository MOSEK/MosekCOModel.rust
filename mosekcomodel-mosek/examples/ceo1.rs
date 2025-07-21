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
use mosekcomodel::*;
use mosekcomodel_mosek::Model;

fn main() {
    let mut m = Model::new(Some("ceo1"));

    let x = m.variable(Some("x"), unbounded().with_shape(&[3]));

    // Create the constraint
    //      x[0] + x[1] + x[2] = 1.0
    _ = m.constraint(Some("lc"), x.sum(), equal_to(1.0));

    // Create the exponential conic constraint
    let expc = m.constraint(Some("expc"), &x, in_exponential_cone());

    // Set the objective function to (x[0] + x[1])
    m.objective(Some("obj"), Sense::Minimize, x.index(0..2).sum());

    // Solve the problem
    m.solve();

    // Get the linear solution values
    let solx = m.primal_solution(SolutionType::Default, &x).unwrap();
    println!("x1,x2,x3 = {}, {}, {}", solx[0], solx[1], solx[2]);

    // Get conic solution of expc
    let  expclvl = m.primal_solution(SolutionType::Default, &expc).unwrap();
    let  expcsn  = m.dual_solution(SolutionType::Default, &expc).unwrap();
    
    println!("expc levels = {:?}", expclvl);

    println!("expc dual conic var levels = {:?}", expcsn);
}
//TAG:end-ceo1


#[test]
fn test() { main() }
