//!
//!  Copyright : Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!  File :      sdo2.rs
//!
//!  Purpose :   Solves the semidefinite problem with two symmetric variables:
//!
//!                 min   <C1,X1> + <C2,X2>
//!                 st.   <A1,X1> + <A2,X2> = b
//!                             (X2)_{1,2} <= k
//!                
//!                 where X1, X2 are symmetric positive semidefinite,
//!
//!                 C1, C2, A1, A2 are assumed to be constant symmetric matrices,/
//!                 and b, k are constants./

extern crate mosekcomodel;
use mosekcomodel::*;


#[allow(non_snake_case)]
fn main() {
    // Since the value of infinity is ignored, we define it solely
    // for symbolic purposes

    // Sample data in sparse lower-triangular triplet form
    let b = 23.0;
    let k = -3.0;

    // Convert input data into Fusion sparse matrices
    let C1 = matrix::sparse([3, 3], vec![[0,0],[2,2]],            &[1.0, 6.0]);
    let C2 = matrix::sparse([4, 4], vec![[0,0],[1,0],[1,1],[2,2]],&[1.0,-3.0,2.0,1.0]);
    let A1 = matrix::sparse([3, 3], vec![[0,0],[2,0],[2,2]],      &[1.0,1.0,2.0]);
    let A2 = matrix::sparse([4, 4], vec![[1,0],[1,1],[3,3]],      &[1.0,-1.0,-3.0]);

    // Define the model
    let mut m = Model::new(Some("sdo2"));
    m.set_log_handler(|msg| print!("{}",msg));
    // Two semidefinite variables
    let X1 = m.variable(Some("X1"),in_psd_cone(3));
    let X2 = m.variable(Some("X2"),in_psd_cone(4));

    // Objective
    m.objective(None,Sense::Minimize, &C1.dot(X1.clone()).add(C2.dot(X2.clone())));

    // Equality constraint
    m.constraint(None, & A1.dot(X1.clone()).add(A2.dot(X2.clone())), equal_to(b));

    // Inequality constraint
    m.constraint(None, &X2.clone().index([0,1]),less_than(k));

    // Solve
    m.solve();

    // Retrieve result
    {
        let X1 : Vec<[f64;3]> = m.primal_solution(SolutionType::Default, &X1).unwrap()
            .chunks(3)
            .map(|c| [c[0],c[1],c[2]])
            .collect();
        let X2 : Vec<[f64; 4]> = m.primal_solution(SolutionType::Default, &X2).unwrap()
            .chunks(4)
            .map(|c| [c[0],c[1],c[2],c[3]])
            .collect();

        println!("X1: {:?}",X1);
        println!("X2: {:?}",X2);
    }
}
#[test]
fn test() { main() }
