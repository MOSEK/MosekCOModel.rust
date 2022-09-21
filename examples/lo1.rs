////
//  Copyright: $$copyright
//
//  File:      $${file}
//
//  Purpose: Demonstrates how to solve the problem
//
//  maximize 3*x0 + 1*x1 + 5*x2 + x3
//  such that
//           3*x0 + 1*x1 + 2*x2        = 30,
//           2*x0 + 1*x1 + 3*x2 + 1*x3 > 15,
//                  2*x1 +      + 3*x3 < 25
//  and
//           x0,x1,x2,x3 > 0,
//           0 < x1 < 10
////
//TAG:begin-lo1
//TAG:begin-import
extern crate mosekmodel;
//TAG:end-import

use mosekmodel::{SolutionType,Model,Sense,greater_than,less_than,equal_to,ModelItemIndex};
use mosekmodel::expr::Expr;

fn main() {
//TAG:begin-define-data

    let a0 = vec![ 3.0, 1.0, 2.0, 0.0 ];
    let a1 = vec![ 2.0, 1.0, 3.0, 1.0 ];
    let a2 = vec![ 0.0, 2.0, 0.0, 3.0 ];
    let c  = vec![ 3.0, 1.0, 5.0, 1.0 ];
//TAG:end-define-data

    // Create a model with the name 'lo1'
//TAG:begin-create-model
    let mut m = Model::new(Some("lo1"));
//TAG:end-create-model
    // Create variable 'x' of length 4
//TAG:begin-create-variable
    let x = m.variable(Some("x"), greater_than(vec![0.0,0.0,0.0,0.0]));
//TAG:end-create-variable

    // Create constraints
//TAG:begin-create-bound
    let _ = m.constraint(None, &x.index(1), less_than(10.0));
//TAG:end-create-bound
//TAG:begin-create-constraints
    let _ = m.constraint(Some("c1"), &Expr::dot(x.clone(),a0), equal_to(30.0));
    let _ = m.constraint(Some("c2"), &Expr::dot(x.clone(),a1), greater_than(15.0));
    let _ = m.constraint(Some("c3"), &Expr::dot(x.clone(),a2), less_than(25.0));
//TAG:end-create-constraints

    // Set the objective function to (c^t * x)
//TAG:begin-set-objective
    m.objective(Some("obj"), Sense::Maximize, &Expr::dot(x.clone(),c));
//TAG:end-set-objective

    // Solve the problem
//TAG:begin-solve
    m.solve();
//TAG:end-solve

    m.write_problem("lo1.ptf");

    // Get the solution values
//TAG:begin-get-solution
    let xx = m.primal_solution(SolutionType::Default,&x);
    println!("x = {:?}", xx);
//TAG:end-get-solution
}

//TAG:end-lo1
