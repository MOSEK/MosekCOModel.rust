extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};
extern crate mosekmodel;

use mosekmodel::*;

// Operation specific benchmarks

fn make_6d_variable(m : & mut Model, sp : bool, n : usize) -> Variable<6> {
    let shape = [n,n,n,n,n,n];
    if ! sp {
        m.variable(None,&shape)
    }
    else {
        let nelm = shape.iter().product();
        let mut strides = [0usize;6]; strides.iter_mut().zip(shape.iter()).rev().fold(1usize,|c,(s,&d)| { *s = c; c*d });
        //println!("shape = {:?}, strides = {:?}", shape,strides);
        let sparsity = (0..nelm).step_by(13).map(|i| { let mut r = [0usize;6]; r.iter_mut().zip(strides.iter()).fold(i,|i,(r,&s)| {*r = i/s; i%s}); r }).collect::<Vec<[usize;6]>>();
        //println!("sparsiy = {:?}", &sparsity[..100]);
        m.variable(None,unbounded().with_shape_and_sparsity(&shape, &sparsity))
    }
}


// ExprTrait::axispermute
fn axis_permute(sp : bool, n : usize) {
    let mut m = Model::new(None);
    let shape = [n,n,n,n,n,n];
    let v = make_6d_variable(&mut m, sp, n);
    _ = m.constraint(None, &v.axispermute(&[3,2,1,0,4,5]).axispermute(&[2,0,1,4,3,5]).axispermute(&[5,4,3,2,1,0]),unbounded().with_shape(&shape));
}

fn axis_permute_dense_8( c : &mut Criterion) { c.bench_function("axis-permute-dense-8", |b| b.iter(|| axis_permute(false,8))); }
fn axis_permute_sparse_8(c : &mut Criterion) { c.bench_function("axis-permute-dense-8", |b| b.iter(|| axis_permute(true,8))); }

// ExprTrait::sum_on 
fn sum_on(sp : bool, n : usize, dims : [usize;3]) {
    let mut m = Model::new(None);
    let v = make_6d_variable(&mut m, sp, n);
    let e = v.clone().add(v.clone().axispermute(&[3,2,1,0,4,5])).add(v.clone().axispermute(&[2,0,1,4,3,5]));

    _ = m.constraint(None, &e.sum_on(&dims),unbounded().with_shape(&[n,n,n]));
}

fn sum_on_dense_8_012( c : &mut Criterion) { c.bench_function("sum-on-dense-8-012", |b| b.iter(|| sum_on(false,8,[0,1,2]))); }
fn sum_on_dense_8_351( c : &mut Criterion) { c.bench_function("sum-on-dense-8-351", |b| b.iter(|| sum_on(false,8,[1,3,5]))); }
fn sum_on_dense_8_345( c : &mut Criterion) { c.bench_function("sum-on-dense-8-345", |b| b.iter(|| sum_on(false,8,[3,4,5]))); }

fn sum_on_sparse_8_012( c : &mut Criterion) { c.bench_function("sum-on-sparse-8-012", |b| b.iter(|| sum_on(true,8,[0,1,2]))); }
fn sum_on_sparse_8_351( c : &mut Criterion) { c.bench_function("sum-on-sparse-8-351", |b| b.iter(|| sum_on(true,8,[1,3,5]))); }
fn sum_on_sparse_8_345( c : &mut Criterion) { c.bench_function("sum-on-sparse-8-345", |b| b.iter(|| sum_on(true,8,[3,4,5]))); }


criterion_group!(benches,
                 axis_permute_dense_8,
                 axis_permute_sparse_8,
                 sum_on_dense_8_012,
                 sum_on_dense_8_351,
                 sum_on_dense_8_345,
                 sum_on_sparse_8_012,
                 sum_on_sparse_8_351,
                 sum_on_sparse_8_345,
                );
criterion_main!(benches);


