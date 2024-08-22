extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};
extern crate mosekmodel;

use mosekmodel::{*, expr::workstack::WorkStack};
use utils::ShapeToStridesEx;

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

fn bench_axis_permute( c : &mut Criterion, sp : bool, n : usize) { c.bench_function(format!("axis-permute-{}-{}",if sp {"sparse"} else {"dense"}, n).as_str(), |b| b.iter(|| axis_permute(sp,n))); }

// ExprTrait::sum_on 
fn sum_on(sp : bool, n : usize, dims : [usize;3]) {
    let mut m = Model::new(None);
    let v = make_6d_variable(&mut m, sp, n);
    let e = v.clone().add(v.clone().axispermute(&[3,2,1,0,4,5])).add(v.clone().axispermute(&[2,0,1,4,3,5]));

    _ = m.constraint(None, &e.sum_on(&dims),unbounded().with_shape(&[n,n,n]));
}

fn bench_sum_on( c : &mut Criterion,sp : bool, n : usize, axes : [usize;3]) { c.bench_function(format!("sum-on-{}-{}-{}{}{}",if sp {"sparse"} else {"dense"},n,axes[0],axes[1],axes[2]).as_str(), |b| b.iter(|| sum_on(sp,n,axes))); }


fn bench_stack(c : & mut Criterion, d : usize, sp : bool, n : usize) {
    c.bench_function(
        format!("stack-{}-{}-{}",if sp {"sparse"} else {"dense"},d,n).as_str(), 
        |b| {
            let mut m = Model::new(None);
            let shape = [n,n,n];
            let v = 
                if sp {
                    let st = shape.to_strides();
                    let sp = (0..shape.iter().product()).step_by(10).map(|i| st.to_index(i)).collect::<Vec<[usize;3]>>();
                    m.variable(None, unbounded().with_shape_and_sparsity(&shape, sp.as_slice()))
                }
                else {
                    m.variable(None,&shape)
                };
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            b.iter(|| {
                rs.clear(); ws.clear(); xs.clear();
                v.clone().stack(d,v.clone()).stack(d,v.clone()).eval_finalize(& mut rs,& mut ws, & mut xs).unwrap();
            });
        });
}





















const N : usize = 10;

fn bench_axis_permute_dense_10( c : & mut Criterion) { bench_axis_permute(c,false,N)}
fn bench_axis_permute_sparse_10(c : & mut Criterion) { bench_axis_permute(c,true,N) }
fn bench_sum_on_dense_10_012(   c : & mut Criterion) { bench_sum_on(c,false,N,[0,1,2]) }
fn bench_sum_on_dense_10_345(   c : & mut Criterion) { bench_sum_on(c,false,N,[3,4,5]) }
fn bench_sum_on_dense_10_135(   c : & mut Criterion) { bench_sum_on(c,false,N,[1,3,5]) }
fn bench_sum_on_sparse_10_012(  c : & mut Criterion) { bench_sum_on(c,true,N,[0,1,2]) }
fn bench_sum_on_sparse_10_345(  c : & mut Criterion) { bench_sum_on(c,true,N,[3,4,5]) }
fn bench_sum_on_sparse_10_135(c : & mut Criterion) { bench_sum_on(c,true,N,[1,3,5]) }

fn bench_stack_dense_0_256(c : & mut Criterion) { bench_stack(c,0,false,256) }
fn bench_stack_dense_1_256(c : & mut Criterion) { bench_stack(c,1,false,256) }
fn bench_stack_dense_2_256(c : & mut Criterion) { bench_stack(c,2,false,256) }
fn bench_stack_sparse_0_768(c : & mut Criterion) { bench_stack(c,0,true,768) }
fn bench_stack_sparse_1_768(c : & mut Criterion) { bench_stack(c,1,true,768) }
fn bench_stack_sparse_2_768(c : & mut Criterion) { bench_stack(c,2,true,768) }


criterion_group!(benches,
    bench_axis_permute_dense_10,
    bench_axis_permute_sparse_10,
    bench_sum_on_dense_10_012,
    bench_sum_on_dense_10_345,
    bench_sum_on_dense_10_135,
    bench_sum_on_sparse_10_012,
    bench_sum_on_sparse_10_345,
    bench_sum_on_sparse_10_135,

    bench_stack_dense_0_256,
    bench_stack_dense_1_256,
    bench_stack_dense_2_256,
    bench_stack_sparse_0_768,
    bench_stack_sparse_1_768,
    bench_stack_sparse_2_768,
    );
criterion_main!(benches);
