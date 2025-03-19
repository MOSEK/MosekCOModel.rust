//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      `total_variation.rs`
//!
//! Purpose:   Demonstrates how to solve a total 
//!            variation problem using the Fusion API.
//!

extern crate mosekcomodel;
extern crate rand;
extern crate rand_distr;
use itertools::iproduct;
use mosekcomodel::*;
use rand::*;
use rand_distr::*;


#[allow(non_snake_case)]
fn total_var(sigma : f64, f : &NDArray<2>) -> (Model,Variable<2>,Constraint<0>) {
    let mut M = Model::new(Some("TV"));
    let n = f.height();
    let m = f.width();

    let u = M.variable(Some("u"), nonnegative().with_shape(&[n+1,m+1]));
    _ = M.constraint(None, &u, less_than(1.0).with_shape(&[n+1,m+1]));
    let t = M.variable(Some("t"), unbounded().with_shape(&[n,m]));

    // In this example we define sigma and the input image f as parameters
    // to demonstrate how to solve the same model with many data variants.
    // Of course they could simply be passed as ordinary arrays if that is not needed.

    let ucore  = (&u).index([0..n,0..m]);
    let deltax = (&u).index([1..n+1,0..m]).sub(ucore.clone()).reshape(&[n,m,1]);
    let deltay = (&u).index([0..n,1..m+1]).sub(ucore.clone()).reshape(&[n,m,1]);

    M.constraint( Some("Delta"), &stack![2; (t.clone()).reshape(&[n,m,1]), deltax, deltay], in_quadratic_cones(&[n,m,3], 2));

    let c = M.constraint(Some("TotalVar"), 
                         &sigma.reshape(&[1,1])
                            .vstack(f.to_expr().sub(ucore).reshape(&[n*m,1]))
                            .flatten(),
                         in_quadratic_cone(n*m+1));

    M.objective( None, Sense::Minimize, &t.sum());

    (M,(&u).index([0..n,0..m]),c.index([0..1]).reshape(&[]))
}


#[allow(non_snake_case)]
fn main() {
    let n : usize = 10;
    let m : usize = 20;

    let R = rand::rngs::StdRng::from_seed([0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]);
    // Create a parametrized model with given shape

    let signal = matrix::dense([m,n],
                               iproduct!(0..m,0..n)
                                   .map(|(i,j)| ((i+j) as f64)/((n+m) as f64))
                                   .collect::<Vec<f64>>());
    let d = rand_distr::Normal::new(0.0, 0.08).unwrap();
    let noise  = matrix::dense([m,n],d.sample_iter(R).take(m*n).collect::<Vec<f64>>());
    let f = signal.add(noise);

    let (mut M,ucore,sigma_con) = total_var(0.0, &f);

    for sigma in [0.0004, 0.0005, 0.0006] {
        let sigma_val = sigma * (m*n) as f64;

        M.update(&sigma_con, &sigma_val);
        // Example: Linear signal with Gaussian noise    
       

        M.solve();
        M.write_problem("total_variation.ptf");

        let _sol = M.primal_solution(SolutionType::Default, &ucore).unwrap();
        // Now use the solution
        // ...

        // Uncomment to get graphics:
        // show(n, m, np.reshape(ucore.level(), (n,m)))

        println!("rel_sigma = {}  total_var = {}",
                 sigma,
                 M.primal_objective(SolutionType::Default).unwrap());

    }
}
#[test]
fn test() { main() }
