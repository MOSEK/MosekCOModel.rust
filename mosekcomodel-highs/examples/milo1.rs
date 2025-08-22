//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!  File:     milo1.rs
//!
//!  Purpose:  Demonstrates how to solve a small mixed
//!            integer linear optimization problem.
//!
//!  The problem:
//!
//!  max       x1 + 0.64 x2
//!  such that 50 x1 + 31 x2 < 250
//!            3 x1 - 2 x2   > -4 
//!            x1,x2 ∈ N₊²
//!
extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel_highs::Model;

fn milo1() -> (SolutionStatus,Result<Vec<f64>,String>) {
    let a0 : &[f64] = &[ 50.0, 31.0 ];
    let a1 : &[f64] = &[ 3.0,  -2.0 ];

    let c : &[f64] = &[ 1.0, 0.64 ];
   
    let mut m = Model::new(Some("milo1"));
    
    let x = m.variable(Some("x"), greater_than(0.0).with_shape(&[2]).integer());

    // Create the constraints
    //      50.0 x[0] + 31.0 x[1] <= 250.0
    //       3.0 x[0] -  2.0 x[1] >= -4.0
    m.constraint(Some("c1"), a0.dot(&x), less_than(250.0));
    m.constraint(Some("c2"), a1.dot(&x), greater_than(-4.0));

    // Set the objective function to (c^T * x)
    m.objective(Some("obj"), Sense::Maximize, c.dot(&x));

    // Solve the problem
    m.solve();

    // Get the solution values
    let (psta,_) = m.solution_status(SolutionType::Integer);

    (psta,m.primal_solution(SolutionType::Integer, &x))
}

fn main() {
    let (psta,xx) = milo1();
    let xx = xx.unwrap();
    println!("Status = {:?}",psta);
    println!("x = {:?}", xx);
    let x = xx;

    assert!(50.0 * x[0] + 32.0 * x[1] < 250.0 + 1e-7);
    assert!( 3.0 * x[0] -  2.0 * x[1] > -4.0 - 1e-7);
}

#[test]
#[cfg(test)]
fn test() { main() }

