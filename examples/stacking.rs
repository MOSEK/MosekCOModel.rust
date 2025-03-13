extern crate mosekcomodel;
use std::time;

use mosekcomodel::{*, expr::workstack::WorkStack};

const REP : usize = 10;
fn stacking1() {
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

    println!("{:<30}: Avg time: {:.2} sec", "Stacking, mixed",t0.elapsed().as_secs_f64()/REP as f64);
}

fn stacking2() {
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

    println!("{:<30}: Avg time: {:.2} sec", "Stacking, dense",t0.elapsed().as_secs_f64()/REP as f64);
}

fn stacking3() {
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

    println!("{:<30}: Avg time: {:.2} sec", "Stacking,sparse",t0.elapsed().as_secs_f64()/REP as f64);
}

fn mul1() {
    const N : usize = 200;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));
    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));
    let m = matrix::dense([N,N], vec![1.1; N*N]);

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    println!("{:<30}: Avg time: {:.2} sec", "Mul dense X * dense M",t0.elapsed().as_secs_f64()/REP as f64);
}

fn mul2() {
    const N : usize = 400;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));
    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::dense([N,N], vec![1.1; N*N]);
    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        y.clone().add(y.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    println!("{:<30}: Avg time: {:.2} sec", "Mul sparse X * dense M",t0.elapsed().as_secs_f64()/REP as f64);
}

fn mul3() {
    const N : usize = 400;
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[N,N]));
    let y = model.variable(None,unbounded().with_shape(&[N,N]).with_sparsity_indexes((0..N*N).step_by(7).collect()));

    let m = matrix::sparse([N,N], (0..N*N).step_by(11).map(|i| [i/N,i%N]).collect::<Vec<[usize;2]>>(), (0..N*N).step_by(11).map(|i| (i % 100) as f64 / 50.0).collect::<Vec<f64>>());

    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        y.clone().add(y.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    println!("{:<30}: Avg time: {:.2} sec", "Mul sparse X * dense M",t0.elapsed().as_secs_f64()/REP as f64);
}



pub fn main() {
    //stacking1();
    //stacking2();
    //stacking3();
    //mul1();
    mul2();
    mul3();
}



