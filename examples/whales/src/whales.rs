//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//! 
//!  File: whales.rs
//!
//!  # Purpose
//!  Computes the minimal ellipsoid containing a set of ellipsoids
//!  References:
//!    [1] "Lectures on Modern Optimization", Ben-Tal and Nemirovski, 2000.    
//!    [2] "MOSEK modeling manual", 2013
//!    [3] "Convex Optimization" Boyd and Vandenberghe, 2004
//!
//!
//!  # The Model
//! 
//!  We wish to find the minimal (by volume) ellipsoid containing a set of other ellipsoids.
//!
//!  
//!
//!
//!  The containing ellipsoid is parameterized as 
//!  ```math 
//!  Eb = { x: ‖Ax+b‖₂ ≤ 1 }, A ∊ S^n_+
//!  ```
//!  and each of the `m` ellipsoids to be surrounded are parameteized as 
//!  ```math
//!  Ei = { x'A_i x + b_i'x + c_i ≤ 0 }, x ∊ R^n
//!  ```
//!  
//!  From [3] we can formulate the model as 
//!  ```math
//!  min log det A^{-1}
//!  such that 
//!     t_i ∊ n R^m_+
//!     A ∊ S^n_+
//!     b ∊ R^n
//!     | A²-t_i A_i     Ab-t_i b_i  0     |
//!     | (Ab-t_i b_i)'  -1-t_i c_i  (Ab)' | ∊ S^{2n+1}_-
//!     | 0              Ab          -A²   |
//!  ```
//!  Since `A²` is PSD, we can safely replace `A²` and `Ab` by a variables `P` and `q` to get
//!  ```math
//!  min log det A^{-1}
//!  such that 
//!     t_i ∊ R^m_+
//!     P ∊ S^n_+
//!     q ∊ R^n
//!     | P-t_i A_i      q-t_i b_i  0   |
//!     | (q-t_i b_i)'  -1-t_i c_i  q'  | ∊ S^{2n+1}_-
//!     | 0              q          -P  |
//!  ```


use mosekmodel::*;
use ellipsoids::*;
use glam::{DVec2,DMat2};


#[allow(non_snake_case)]
pub fn minimal_bounding_ellipsoid<const N : usize>(data : &[Ellipsoid<N>]) -> Result<([[f64;N];N],[f64;N]),String> {
    let n = N;

    let mut M = Model::new(Some("lowner-john-outer"));
    M.set_log_handler(|msg| print!("{}",msg));

    let m = data.len();

    // Maximize log(det(P))
    let t = M.variable(Some("t"), unbounded());
    
    //let P = M.variable(Some("P"),in_psd_cone(n));
    let P = det_rootn(Some("P"), & mut M, t.clone(), n);
    let q = M.variable(Some("q"), unbounded().with_shape(&[n]));
    M.objective(None, Sense::Maximize, &t);


    for e in data.iter() {
       ellipsoid_contains(&mut M,&P,&q,e);
    }

    M.solve();

    M.write_problem("lj-outer-ellipsoid.ptf");

    let Psol = M.primal_solution(SolutionType::Default, &P)?;
    let qsol = M.primal_solution(SolutionType::Default, &q)?;

    let mut rP = [[0.0;N];N]; rP.iter_mut().flat_map(|row| row.iter_mut()).zip(Psol.iter()).for_each(|(t,&s)| *t = s);
    let mut rq = [0.0;N]; rq.clone_from_slice(qsol.as_slice());

    Ok( (rP,rq) )
}

#[allow(non_snake_case)]
pub fn maximal_contained_ellipsoid<const N : usize>(data : &[Ellipsoid<N>]) -> Result<([[f64;N];N],[f64;N]),String> 
{
    let mut M = Model::new(Some("lowner-john-inner"));

    let t = M.variable(Some("t"), unbounded());
    
    //let P = M.variable(Some("P"),in_psd_cone(n));
    let P = det_rootn(Some("P"), & mut M, t.clone(), N);
    let q = M.variable(Some("q"), unbounded().with_shape(&[N]));
    M.objective(None, Sense::Maximize, &t);

    for e in data.iter() {
       ellipsoid_contained(&mut M,&P,&q,e);
    }

    M.solve();

    M.write_problem("lj-inner-ellipsoid.ptf");

    let Psol = M.primal_solution(SolutionType::Default, &P)?;
    let qsol = M.primal_solution(SolutionType::Default, &q)?;

    let mut rP = [[0.0;N];N]; rP.iter_mut().flat_map(|row| row.iter_mut()).zip(Psol.iter()).for_each(|(t,&s)| *t = s);
    let mut rq = [0.0;N]; rq.clone_from_slice(qsol.as_slice());

    Ok( (rP,rq) )
}

#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use super::*;
    use ellipsoids::*;

    fn ellipse_from_param(dx : f64, dy : f64, sx : f64, sy : f64, theta : f64) -> Ellipsoid<2> {
        let A = [ [ sx*theta.cos(), sy*theta.sin()], [-sx*theta.sin(), sy*theta.cos() ] ];
        let b = [ dx, dy ];

        Ellipsoid::new(&A,&b)
    }

    //xA'Ax + 2Abx + b'b-1 = 0

    #[test]
    fn test() {
        let ellipses : &[Ellipsoid<2>] = &[
            Ellipsoid::new(&[[2.3246108597249653, -0.5023002219069703],  [-0.5023002219069703, 1.143239126869145]], &[ 1.6262243355626635,   2.3572628105100137]),
            Ellipsoid::new(&[[0.7493116238095875, -0.4519632176898713],  [-0.4519632176898713, 1.8627720007697564]],&[-0.7741774083306816,  -0.19900209213742667]),
            Ellipsoid::new(&[[0.8582679328352112,  0.21661818880463501], [ 0.21661818880463501,0.8721042500171801]],&[-0.30881539733571506,  0.6395981801584628]),
            Ellipsoid::new(&[[2.9440718650596165, -1.0778871720849044],  [-1.0778871720849044, 3.797461570034363]], &[ 2.2609091903635528,   5.387366401851428]),
            Ellipsoid::new(&[[0.6104500401008942, -0.06447520306755025], [-0.06447520306755025,0.6137552998087111]],&[-0.36695528785675785, -0.7214779986292089]),
            Ellipsoid::new(&[[1.1044036516422946, -0.18480500741119338], [-0.18480500741119338,4.271283557279645]], &[-5.123066175367,       2.1838724317503617]),
       ];

        let (P,q) = minimal_bounding_ellipsoid(ellipses).uwrap();

        println!("P = {:?}",P);
        println!("q = {:?}",q);

        // Now, bonud ellipsoid is defined by A,b, with
        // A² = P => P = sqrt(A)
        // Ab = q => A\q
    
        
    }
}
