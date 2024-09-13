//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//  File:      sdo1.rs
//
//  Purpose:
//  Solves the mixed semidefinite and conic quadratic optimization problem
//
//                   [2, 1, 0]
//    minimize    Tr [1, 2, 1] * X + x0
//                   [0, 1, 2]
//
//                   [1, 0, 0]
//    subject to  Tr [0, 1, 0] * X + x0           = 1.0
//                   [0, 0, 1]
//
//                   [1, 1, 1]
//                Tr [1, 1, 1] * X      + x1 + x2 = 0.5
//                   [1, 1, 1]
//
//                   X >> 0,  x0 >= (x1^2 + x2^2) ^ (1/2)
//
extern crate mosekmodel;
use mosekmodel::*;

#[test]
fn main() {
    let mut m = Model::new(Some("sdo1"));
    // Setting up the variables
    let barx  = m.variable(Some("X"),  in_psd_cone(3));
    let x     = m.variable(Some("x"),  in_quadratic_cone(3));

    // Setting up constant coefficient matrices
    let barc  = matrix::dense([3, 3], vec![2., 1., 0., 
                                           1., 2., 1., 
                                           0., 1., 2.]);
    let bara1 = matrix::diag(&[1.0;3]);
    let bara2 = matrix::ones([3,3]);

    // Objective
    m.objective(None, Sense::Minimize, &barx.clone().dot(barc).add(x.clone().index(0)));

    // Constraints
    _ = m.constraint(Some("c1"), &barx.clone().dot(bara1).add((&x).index(0)), equal_to(1.0));
    _ = m.constraint(Some("c2"), &barx.clone().dot(bara2).add((&x).index([1..3]).sum()), equal_to(0.5));


    m.solve();

    let barx_sol = m.primal_solution(SolutionType::Default,&barx);
    let x_sol = m.primal_solution(SolutionType::Default, &x);

    println!("X = {:?}",barx_sol);
    println!("x = {:?}",x_sol);
}
