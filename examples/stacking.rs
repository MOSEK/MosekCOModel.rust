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
    let mut model = Model::new(None);

    let x = model.variable(None,unbounded().with_shape(&[200,200]));
    let y = model.variable(None,unbounded().with_shape(&[200,200]).with_sparsity_indexes((0..40000).step_by(7).collect()));

    let m = matrix::dense([200,200], vec![1.1; 200*200]);
    let mut ws = WorkStack::new(1024);
    let mut rs = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let t0 = time::Instant::now();

    for _ in 0..REP {
        x.clone().add(x.clone().transpose()).mul(m.clone()).eval(&mut rs, &mut ws, &mut xs).unwrap();
    }
    
    println!("{:<30}: Avg time: {:.2} sec", "Mul X * M",t0.elapsed().as_secs_f64()/REP as f64);
}





pub fn main() {
    //stacking1();
    //stacking2();
    //stacking3();
    mul1();
}



