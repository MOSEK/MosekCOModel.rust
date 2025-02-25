///
///   Copyright: © Mosek ApS, 2024
///
///   Purpose: Demonstrates how to solve the problem
///
///   minimize x1 + x2
///   such that
///            x1 + x2 + x3  = 1.0
///                x1,x2    >= 0.0
///   and      x1 >= x2 * exp(x3/x2)
///
///TAG:begin-ceo1
extern crate mosekcomodel;
use mosekcomodel::*;

fn main() {
    //TAG:begin-create-model
    let mut m = Model::new(Some("ceo1"));
    //TAG:end-create-model

    //TAG:begin-create-variable
    let x = m.variable(Some("x"), unbounded().with_shape(&[3]));
    //TAG:end-create-variable

    //TAG:begin-create-lincon
    // Create the constraint
    //      x[0] + x[1] + x[2] = 1.0
    _ = m.constraint(Some("lc"), &x.clone().sum(), equal_to(1.0));
    //TAG:end-create-lincon

    //TAG:begin-create-concon
    // Create the exponential conic constraint
    let expc = m.constraint(Some("expc"), &x.clone(), in_exponential_cone());
    //TAG:end-create-concon

    //TAG:begin-set-objective
    // Set the objective function to (x[0] + x[1])
    m.objective(Some("obj"), Sense::Minimize, &x.clone().index(0..2).sum());
    //TAG:end-set-objective

    // Solve the problem
    //TAG:begin-solve
    m.solve();
    //TAG:end-solve

    //TAG:begin-get-solution
    // Get the linear solution values
    let solx = m.primal_solution(SolutionType::Default, &x).unwrap();
    //TAG:end-get-solution
    println!("x1,x2,x3 = {}, {}, {}", solx[0], solx[1], solx[2]);

    //TAG:begin-get-con-sol
    // Get conic solution of expc
    let  expclvl = m.primal_solution(SolutionType::Default, &expc).unwrap();
    let  expcsn  = m.dual_solution(SolutionType::Default, &expc).unwrap();
    //TAG:end-get-con-sol
    
    println!("expc levels = {:?}", expclvl);

    println!("expc dual conic var levels = {:?}", expcsn);
}
//TAG:end-ceo1


#[test]
fn test() { main() }
