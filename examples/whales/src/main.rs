extern crate mosekmodel;
mod whales;
mod utils2d;

use whales::{Ellipsoid,outer_ellipsoid};
use utils2d::{det,matscale,matmul,matadd,trace,inv};

#[allow(non_snake_case)]
fn main() {
    let ellipses : &[Ellipsoid<2>] = &[
        Ellipsoid::new(&[[1.2576, -0.3873], [-0.3873,0.3467]], &[ 0.2722,  0.1969], 0.1831),
        Ellipsoid::new(&[[1.4125, -2.1777], [-2.1777,6.7775]], &[-1.228,  -0.0521], 0.3295),
        Ellipsoid::new(&[[1.7018,  0.8141], [ 0.8141,1.7538]], &[-0.4049,  1.5713], 0.2077),
        Ellipsoid::new(&[[0.9742, -0.7202], [-0.7202,1.5444]], &[ 0.0265,  0.5623], 0.2362),
        Ellipsoid::new(&[[0.6798, -0.1424], [-0.1424,0.6871]], &[-0.4301, -1.0157], 0.3284),
        Ellipsoid::new(&[[0.1796, -0.1423], [-0.1423,2.6181]], &[-0.3286,  0.557 ], 0.4931) 
    ];

    let (Psq,Pq) = outer_ellipsoid(ellipses);

    println!("P² = {:?}",Psq);
    println!("Pq = {:?}",Pq);
    
    let s = det(&Psq).sqrt();
    let P = matscale(&matadd(&Psq,&[[s,0.0],[0.0,s]]), 1.0/(trace(&Psq) + 2.0*s).sqrt());
    let q = matmul(&inv(&P),&Pq);

    println!("P = {:?}",P);
    println!("q = {:?}",q);
}

