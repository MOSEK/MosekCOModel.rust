//!
//! Copyright : Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File :      `sdo3.rs`
//!
//! Purpose :   Solves the semidefinite problem:
//! ``` 
//! min   tr(X_1) + ... + tr(X_n)
//! st.   <A_11,X_1> + ... + <A_1n,X_n> >= b_1
//!       ...
//!       <A_k1,X_1> + ... + <A_kn,X_n> >= b_k
//! ```               
//! where `X_i` are symmetric positive semidefinite of dimension d,
//!
//! `A_ji` are constant symmetric matrices and b_i are constant.
//!
//! This example is to demonstrate creating and using 
//! many matrix variables of the same dimension.

extern crate mosekcomodel;
extern crate rand;
use expr::nil;
use mosekcomodel::*;


fn rand_matrix(shape : [usize;2]) -> NDArray<2> {
    matrix::dense(shape, (0..shape[0]*shape[1]).map(|_| rand::random()).collect::<Vec<f64>>())
}


#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
fn main() {
    // Sample input data
    const n : usize = 100;
    const d : usize = 4;
    const b : [f64;3] = [9.0,10.0,11.0];
    const k : usize = b.len();

    let A : Vec<NDArray<2>> = (0..k*n).map(|_| rand_matrix([d,d])).map(|m| (m.transpose()+m).mul_scalar(2.5)).collect();

    let mut m = Model::new(Some("sdo3"));
    m.set_log_handler(|msg| print!("{}",msg));
    // Create a model with n semidefinite variables of dimension d x d
    let X = m.variable(Some("X"),in_psd_cones(&[n,d,d],1,2));

    // Pick indexes of diagonal entries for the objective

    m.objective(None,Sense::Minimize, 
                X.index([0..n,0..1,0..1])
                    .add(X.index([0..n,1..2,1..2]))
                    .add(X.index([0..n,2..3,2..3])).sum());

    // Each constraint is a sum of inner products
    // Each semidefinite variable is a slice of X
    for (&bi,As) in b.iter().zip(A.chunks(n)) {
        m.constraint(None,
                     As.iter().enumerate()
                        .map(|(j,A)| X.index([j..j+1,0..d,0..d]).reshape(&[d,d]).dot(A.clone()))
                        .fold(nil(&[]).dynamic(),|c,e| c.add(e).dynamic()),
                     greater_than(bi));
    }

    // Solve
    m.solve();

    // Get results. Each variable is a slice of X
    println!("Contributing variables:");
    for j in 0..n {
        let Xj = m.primal_solution(SolutionType::Default, &X.index([j..j+1, 0..d,0..d])).unwrap();
        if Xj.iter().any(|&s| s > 1e-6) {
            println!("X{} = {:?}",j, Xj);
        }
    }
}

#[test]
fn test() { main() }
