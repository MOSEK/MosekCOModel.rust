// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:      duality.rs
//
// Purpose: Show how to read the dual value of a constraint.
extern crate mosekcomodel;

use mosekcomodel::*;

#[allow(non_snake_case)]
fn main() {
    let A = [ [ -0.5, 1.0 ] ];
    let b = [ 1.0 ];
    let c =  [ 1.0, 1.0 ];

    let mut model = Model::new(Some("duality"));

    let x = model.variable(Some("x"), greater_than(vec![0.0,0.0]));

    let con = model.constraint(None, &NDArray::from(&A).mul(x.clone()).sub(b.to_vec()), equal_to(vec![0.0]));

    model.objective(Some("obj"), Sense::Minimize, &x.clone().dot(c.to_vec()));

    model.solve();
    let xsol = model.primal_solution(SolutionType::Default, &x).unwrap();
    let ssol = model.dual_solution(SolutionType::Default, &x).unwrap();
    let ysol = model.dual_solution(SolutionType::Default, &con).unwrap();

    println!("x1,x2,s1,s2,y = {}, {}, {}, {}, {}", xsol[0], xsol[1], ssol[0], ssol[1], ysol[0]);
}

#[test]
fn test() { main() }
