//! # Comparable times
//! 
//! Comparing runtimes of simple expressions as implemented in MosekCOModel (--release mode) and in
//! MOSEK Java Fusion.
//! 
//! Date: March 13, 2025
//!|                         | Rust  | Java  |
//!| Stacking, mixed         |  0.19 |  0.55 |
//!| Stacking, dense         |  0.09 |  0.20 |
//!| Stacking, sparse        |  0.02 |  0.06 |
//!| Mul dense X * dense M   |  0.14 |  0.26 |
//!| Mul sparse X * dense M  |  0.17 |  0.07 |
//!| Mul dense X * sparse M  |  0.12 |  0.03 |
//!| Mul sparse X * sparse M |  0.10 |  0.06 |
//!| Mul dense M * dense X   |  0.14 |  0.33 |
//!| Mul dense M * sparse X  |  0.24 |  0.09 |
//!| Mul sparse M * dense X  |  0.14 |  0.04 |
//!| Mul sparse M * sparse X |  0.28 |  0.06 |
//!
use std::{time, collections::HashMap};

use mosekcomodel::{*, expr::workstack::WorkStack};

const REP : usize = 10;
pub fn stacking1() -> f64 {
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[100,100,100]));
    let y = model.variable(None,unbounded().with_shape(&[100,100,100]).with_sparsity_indexes((0..1000000).step_by(11).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        for d in 0..3 {
            rs.clear();
            x.clone().stack(d, y.clone()).stack(d,x.clone()).stack(d,y.clone()).eval(& mut rs, & mut ws, & mut xs).unwrap();
        }
    }

    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn stacking2() -> f64 {
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[100,100,100]));

   let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        for d in 0..3 {
            rs.clear();
            x.clone().stack(d, x.clone()).stack(d,x.clone()).stack(d,x.clone()).eval(& mut rs, & mut ws, & mut xs).unwrap();
        }
    }

    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn stacking3() -> f64 {
    let mut model = Model::new(None);

    let y = model.variable(None,unbounded().with_shape(&[100,100,100]).with_sparsity_indexes((0..1000000).step_by(11).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        for d in 0..3 {
            rs.clear();
            y.clone().stack(d, y.clone()).stack(d,y.clone()).stack(d,y.clone()).eval(& mut rs, & mut ws, & mut xs).unwrap();
        }
    }

    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul1() -> f64 {
    const N : usize = 400;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));
    let m = matrix::dense([N,N], vec![1.1; N*N]);

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul2() -> f64 {
    const N : usize = 400;
    let mut model = Model::new(None);

    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::dense([N,N], vec![1.1; N*N]);
    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        y.clone().add(y.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul3() -> f64 {
    const N : usize = 800;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));

    let m = matrix::sparse([N,N], (0..N*N).step_by(11).map(|i| [i/N,i%N]).collect::<Vec<[usize;2]>>(), (0..N*N).step_by(11).map(|i| (i % 100) as f64 / 50.0).collect::<Vec<f64>>());

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul4() -> f64 {
    const N : usize = 800;
    let mut model = Model::new(None);

    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::sparse([N,N], (0..N*N).step_by(11).map(|i| [i/N,i%N]).collect::<Vec<[usize;2]>>(), (0..N*N).step_by(11).map(|i| (i % 100) as f64 / 50.0).collect::<Vec<f64>>());

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        y.clone().add(y.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul5() -> f64 {
    const N : usize = 400;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));
    let m = matrix::dense([N,N], vec![1.1; N*N]);

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        m.clone().mul(x.clone().add(x.clone().transpose())).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul6() -> f64 {
    const N : usize = 800;
    let mut model = Model::new(None);

    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::dense([N,N], vec![1.1; N*N]);
    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        m.clone().mul(y.clone().add(y.clone().transpose())).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul7() -> f64 {
    const N : usize = 800;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));

    let m = matrix::sparse([N,N], (0..N*N).step_by(11).map(|i| [i/N,i%N]).collect::<Vec<[usize;2]>>(), (0..N*N).step_by(11).map(|i| (i % 100) as f64 / 50.0).collect::<Vec<f64>>());

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        m.clone().mul(x.clone().add(x.clone().transpose())).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn mul8() -> f64 {
    const N : usize = 800;
    let mut model = Model::new(None);

    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::sparse([N,N], (0..N*N).step_by(11).map(|i| [i/N,i%N]).collect::<Vec<[usize;2]>>(), (0..N*N).step_by(11).map(|i| (i % 100) as f64 / 50.0).collect::<Vec<f64>>());

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        m.clone().mul(y.clone().add(y.clone().transpose())).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}


pub fn sumon1() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().axispermute(&[1,2,0])).sum_on(&[1,2]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon2() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().axispermute(&[1,2,0])).sum_on(&[0,2]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon3() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().axispermute(&[1,2,0])).sum_on(&[0,1]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon4() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().axispermute(&[1,2,0])).sum_on(&[1]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}


pub fn sumon1s() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        s.clone().add(s.clone().axispermute(&[1,2,0])).sum_on(&[1,2]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon2s() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        s.clone().add(s.clone().axispermute(&[1,2,0])).sum_on(&[0,2]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon3s() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        s.clone().add(s.clone().axispermute(&[1,2,0])).sum_on(&[0,1]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}

pub fn sumon4s() -> f64 {
    const N : usize = 300;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N,N]));
    let s = model.variable(None,unbounded().with_shape(&[N,N,N]).with_sparsity_indexes((0..N*N*N).step_by(7).collect()));

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        s.clone().add(s.clone().axispermute(&[1,2,0])).sum_on(&[1]).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    t0.elapsed().as_secs_f64()/REP as f64
}
