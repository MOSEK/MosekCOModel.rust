///
///  Copyright
///     Copyright (c) MOSEK ApS, Denmark. All rights reserved.
///
///  File
///     sospoly.rs
///
///  # Purpose
///
///  Models the cone of nonnegative polynomials and nonnegative trigonometric
///  polynomials using Nesterov's framework  [1].
///
///  Given a set of coefficients `(x0, x1, ..., xn)`, the functions model the
///  cone of nonnegative polynomials
///
///  ```math
///  P_m = { x ∊ R^{n+1} | x0 + x1*t + ... xn*t^n ≥ 0, ∀ t ∊ I }
///  ```
///
///  where I can be the entire real axis, the semi-infinite interval `[0,inf)`, or
///  a finite interval `I = [a, b]`, respectively.
///
///  # References
///
///  [1] "Squared Functional Systems and Optimization Problems",
///      Y. Nesterov, in High Performance Optimization,
///      Kluwer Academic Publishers, 2000.
extern crate mosekmodel;

use itertools::iproduct;
use mosekmodel::*;
use mosekmodel::matrix;


/// Creates a Hankel matrix of dimension n+1, where
/// ```math
///         / a if l+k=i
/// H_lk = (
///         \ 0 otherwise
/// ```
fn hankel(n : usize, i_ : isize, a : f64) -> NDArray<2> {
    if i_ < 0 || i_ as usize > 2*n {
        matrix::zeros([n+1,n+1])
    } 
    else {
        let i = i_ as usize;
        if i < n + 1 {
            matrix::sparse([n+1, n+1],
                           (0..i+1).rev().zip(0..i+1).map(|(i,j)| [i,j]).collect().as_slice(),
                           vec![a; i+1].as_slice())
        } 
        else {
            matrix::sparse([n+1, n+1], 
                           (i-n..n+1).rev().zip(i-n..n+1).map(|(i,j)| [i,j]).collect().as_slice(),
                           vec![a; 2*n+1].as_slice())
        }
    }
}

/// Models the cone of nonnegative polynomials on the real axis
#[allow(unused)]
fn nn_inf(model : & mut Model, x : & Variable<1>) {
    let m = x.shape()[0] - 1;
    let n = m / 2; 
    // Setup variables
    let barx = model.variable(None, in_psd_cone(n+1));

    // x_i = Tr H(n, i) * X  i=0,...,m
    for i in 0..m+1 {
        _ = model.constraint(None, & x.clone().index(i).sub(hankel(n, i as isize, 1.0).dot(barx.clone())), equal_to(0.0));
    }
}

/// Models the cone of nonnegative polynomials on the semi-infinite interval `[0,∞)`
#[allow(unused)]
fn nn_semiinf(m : & mut Model, x : & Variable<1>) {
    let n = x.shape()[0] - 1;
    let n1 = n / 2;
    let n2 = (n - 1) / 2;  

    // Setup variables
    let barx1 = m.variable(None, in_psd_cone(n1+1));
    let barx2 = m.variable(None, in_psd_cone(n2+1));

    // x_i = Tr H(n1, i) * X1 + Tr H(n2,i-1) * X2, i=0,...,n
    
    for i in 0..n+1 {
        m.constraint(None, &x.clone().index(i).sub(hankel(n1,i as isize,1.0).dot(barx1.clone()).add(hankel(n2,i as isize -1,1.0).dot(barx2.clone()))), equal_to(0.0));
    }
    for i in 0..n+1 {
        m.constraint(None, 
                     &x.clone().index(i).sub(
                        hankel(n1,i as isize,1.0).dot(barx1.clone()).add(
                            hankel(n2, i as isize -1, 1.0).dot(barx2.clone()))),
                     equal_to(0.0));
    }
}

/// Models the cone of nonnegative polynomials on the finite interval `[a,b]`
fn nn_finite(model : & mut Model, x : & Variable<1>, a : f64, b : f64) {
    let m = x.shape()[0]-1;
    let n = m / 2;

    if m == 2 * n {
        let barx1 = model.variable(None,in_psd_cone(n+1));
        let barx2 = model.variable(None,in_psd_cone(n));

        // x_i = Tr H(n,i)*X1 + (a+b)*Tr H(n-1,i-1) * X2 - a*b*Tr H(n-1,i)*X2 - Tr H(n-1,i-2)*X2, i=0,...,m
    
        for i in 1..m+1 {
            _ = model.constraint(
                None,
                &hankel(n,i as isize,1.0).dot(barx1.clone())
                    .sub(hankel(n-1,i as isize,a*b).dot(barx2.clone()))
                    .add(hankel(n-1,i as isize-1,a+b).dot(barx2.clone())
                         .sub(hankel(n-1,i as isize -2,1.0).dot(barx2.clone()))),
                equal_to(0.0));
        }
    } else {
        let barx1 = model.variable(None, in_psd_cone(n+1));
        let barx2 = model.variable(None, in_psd_cone(n+1));

        // x_i = Tr H(n,i-1)*X1 - a*Tr H(n,i)*X1 + b*Tr H(n,i)*X2 - Tr H(n,i-1)*X2, i=0,...,m
        for i in 1..m+1 {
            _ = model.constraint( 
                None,
                &x.index(i)
                    .sub(
                        hankel(n,i as isize -1,1.0).dot(barx1.clone()).sub(hankel(n,i as isize,a).dot(barx1.clone()))
                            .add(hankel(n,i as isize,b).dot(barx2.clone()).sub(hankel(n,i as isize-1,1.0).dot(barx2.clone())))),
                equal_to(0.0));
        }
    }
}

  // returns variables u representing the derivative of
  //  x(0) + x(1)*t + ... + x(n)*t^n,
  // with u(0) = x(1), u(1) = 2*x(2), ..., u(n-1) = n*x(n).
fn diff(model : & mut Model, x : & Variable<1>) -> Variable<1> {
    let n = x.shape()[0]-1;
    let u = model.variable(None, n);
    _ = model.constraint(None,
                         &u.clone().reshape(&[n,1]).sub(x.clone().index(1..n+1).reshape(&[n,1]).mul_elem(matrix::dense([n,1],(1..n+1).map(|v| v as f64).collect::<Vec<f64>>().as_slice()))), 
                         equal_to(vec![0.0;n].as_slice()).with_shape(&[n,1]));
    u
}

fn fitpoly(data : & NDArray<2>, n : usize) -> Vec<f64> {
    let mut model = Model::new(Some("smooth poly"));

    let datadim = data.shape();
    let m = datadim[0];
    let datacof = data.data();
    let adata : Vec<f64> = iproduct!(datacof[0..datadim[0]].iter(), 0..n+1).map(|(c,i)| c.powf(i as f64)).collect();

    let a = matrix::dense([m,n+1],adata);

    let b = &datacof[datadim[0]..datadim[0]*2];

    let x = model.variable(Some("x"), n + 1);
    let z = model.variable(Some("z"), 1);
    let dx = diff(& mut model, &x);

    _ = model.constraint(None,&a.mul(x.clone()),equal_to(b));

    // z - f'(t) >= 0, for all t \in [a, b]
    let ub = model.variable(None,n);
    _ = model.constraint(None,
                         &ub.clone().sub(vstack![z.clone().sub(dx.clone().index(0..1)), dx.clone().index(1..n)]),
                         equal_to(vec![0.0; n]));

    nn_finite(&mut model, &ub, datacof[0], datacof[datacof.len()-datadim[1]]); 

    // f'(t) + z >= 0, for all t \in [a, b]
    let lb = model.variable(None,n);
    _ = model.constraint(None,
                         &lb.clone().sub(vstack![z.clone().add(dx.clone().index(0..1)), dx.clone().index(1..n)]),
                         equal_to(vec![0.0; n]));

    nn_finite(&mut model, &lb, datacof[0], datacof[datacof.len() - datadim[1]]);

    model.objective(None, Sense::Minimize, &z.index(0));
    model.solve();
    model.primal_solution(SolutionType::Interior, &x).unwrap()
}


fn main() {
    let data = matrix::dense([3,2],
                             vec![ -1.0, 1.0,
                                    0.0, 0.0,
                                    1.0, 1.0 ]);
    
    let x2 = fitpoly(&data, 2);
    let x4 = fitpoly(&data, 4);
    let x8 = fitpoly(&data, 8);

    println!("fitpoly(data,2) -> {:?}",x2);
    println!("fitpoly(data,4) -> {:?}",x4);
    println!("fitpoly(data,8) -> {:?}",x8);
}
