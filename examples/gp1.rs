//!
//!   Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!   File:      gp1.rs
//!
//!   Purpose:   Demonstrates how to solve a simple Geometric Program (GP)
//!              cast into conic form with exponential cones and log-sum-exp.
//!
//!   Example from: 
//!                <https://gpkit.readthedocs.io/en/latest/examples.html//maximizing-the-volume-of-a-box>
//!
extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel::expr::ones;

/// Models log(sum(exp(Ax+b))) <= 0.
/// Each row of [A b] describes one of the exp-terms
#[allow(non_snake_case)]
fn logsumexp(model : & mut Model, 
             A : &NDArray<2>,
             x : &Variable<1>,
             b : &[f64])
{
    let k = A.shape()[0];
    let u = model.variable(None,&[k,1]);
    model.constraint(None,u.sum(), equal_to(1.0));
    model.constraint(None, 
                     hstack![u.to_expr(),
                             ones(&[k,1]),
                             A.mul(x).add(b).reshape(&[k,1])],
                     in_exponential_cones(&[k,3],1));
}

/// maximize     h*w*d
/// subjecto to  2*(h*w + h*d) <= Awall
///              w*d <= Afloor
///              alpha <= h/w <= beta
///              gamma <= d/w <= delta
///
/// Variable substitutions:  h = exp(x), w = exp(y), d = exp(z).
///
/// maximize     x+y+z
/// subject      log( exp(x+y+log(2/Awall)) + exp(x+z+log(2/Awall)) ) <= 0
///                              y+z <= log(Afloor)
///              log( alpha ) <= x-y <= log( beta )
///              log( gamma ) <= z-y <= log( delta )
#[allow(non_snake_case)]
fn max_volume_box(Aw    : f64, 
                  Af    : f64, 
                  alpha : f64, 
                  beta  : f64,  
                  gamma : f64, 
                  delta : f64) -> Vec<f64>
{
    let mut model = Model::new(Some("max_vol_box"));

    let xyz = model.variable(None, 3);
    model.objective(Some("Objective"), Sense::Maximize, xyz.sum());
  
    logsumexp(&mut model, 
              &NDArray::from(&[[1.0,1.0,0.0], [1.0,0.0,1.0]]),
              &xyz,
              &[(2.0/Aw).ln(), (2.0/Aw).ln()]);
  
    model.constraint(None,xyz.dot(vec![0.0, 1.0,1.0]), less_than(Af.ln()));
    model.constraint(None,xyz.dot(vec![1.0,-1.0,0.0]), greater_than(alpha.ln()));
    model.constraint(None,xyz.dot(vec![1.0,-1.0,0.0]), less_than(beta.ln()));
    model.constraint(None,xyz.dot(vec![0.0,-1.0,1.0]), greater_than(gamma.ln()));
    model.constraint(None,xyz.dot(vec![0.0,-1.0,1.0]), less_than(delta.ln()));
  
    //model.setLogHandler(new java.io.PrintWriter(System.out));
    model.solve();
  
    let xyz_val = model.primal_solution(SolutionType::Default,&xyz).unwrap();
    let hwd_val = xyz_val.iter().map(|v| v.exp()).collect();

    hwd_val
}

#[allow(non_snake_case)]
fn main() {
    let Aw    = 200.0;
    let Af    = 50.0;
    let alpha = 2.0;
    let beta  = 10.0;
    let gamma = 2.0;
    let delta = 10.0;
    
    let hwd = max_volume_box(Aw, Af, alpha, beta, gamma, delta);

    println!("h={:.4} w={:.4} d={:.4}\n", hwd[0], hwd[1], hwd[2]);
}

#[test]
fn test() { main() }
