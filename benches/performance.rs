extern crate criterion;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::workstack::WorkStack;
use mosekmodel::expr::*;


const N1 : usize = 100;
const N2 : usize = 500;

fn mul_dense_matrix_x_dense_expr(c: &mut Criterion) {
    c.bench_function("mul_dense_matrix_x_dense_expr",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N1,N1], // dim
                              None, // sparsity
                              (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                              (0..N1*N1*2).collect(), // subj
                              vec![1.0; N1*N1*2]); // cof

            let m = matrix::dense(N1,N1, vec![1.0; N1*N1]);
            m.mul(e).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_dense_matrix_x_sparse_expr(c: &mut Criterion) {
    c.bench_function("mul_dense_matrix_x_sparse_expr",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N2,N2], // dim
                              Some((0..N2*N2).step_by(7).collect()), // sparsity
                              (0..((N2*N2)/7+2)*2).step_by(2).collect(), // ptr
                              (0..((N2*N2)/7+1)*2).collect(), // subj
                              vec![1.0; (N2*N2/7+1)*2]); // cof

            let m = matrix::dense(N2,N2, vec![1.0; N2*N2]);
            m.mul(e).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_dense_expr_x_dense_matrix(c: &mut Criterion) {
    c.bench_function("mul_dense_expr_x_dense_matrix",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N1,N1], // dim
                              None, // sparsity
                              (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                              (0..N1*N1*2).collect(), // subj
                              vec![1.0; N1*N1*2]); // cof
            let m = matrix::dense(N1,N1, vec![1.0; N1*N1]);
            e.mul(m).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_sparse_expr_x_dense_matrix(c: &mut Criterion) {
    c.bench_function("mul_sparse_expr_x_dense_matrix",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N2,N2], // dim
                              Some((0..N2*N2).step_by(7).collect()), // sparsity
                              (0..((N2*N2)/7+2)*2).step_by(2).collect(), // ptr
                              (0..((N2*N2)/7+1)*2).collect(), // subj
                              vec![1.0; (N2*N2/7+1)*2]); // cof

            let m = matrix::dense(N2,N2, vec![1.0; N2*N2]);
            e.mul(m).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_sparse_matrix_x_dense_expr(c: &mut Criterion) {
    c.bench_function("mul_sparse_matrix_x_dense_expr",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N1,N1], // dim
                              None, // sparsity
                              (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                              (0..N1*N1*2).collect(), // subj
                              vec![1.0; N1*N1*2]); // cof

            let m = matrix::sparse(N1,N1,
                                   (1..2*N2+1).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   (0..2*N2).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   vec![1.0; 2*N2].as_slice());
            m.mul(e).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_sparse_matrix_x_sparse_expr(c: &mut Criterion) {
    c.bench_function("mul_sparse_matrix_x_sparse_expr",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N2,N2], // dim
                              Some((0..N2*N2).step_by(7).collect::<Vec<usize>>()), // sparsity
                              (0..((N2*N2)/7+1)*2).step_by(2).collect::<Vec<usize>>(), // ptr
                              (0..((N2*N2)/7)*2).collect(), // subj
                              vec![1.0; (N2*N2/7)*2]); // cof

            let m = matrix::sparse(N1,N1,
                                   (1..2*N2+1).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   (0..2*N2).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   vec![1.0; 2*N2].as_slice());
            m.mul(e).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_dense_expr_x_sparse_matrix(c: &mut Criterion) {
    c.bench_function("mul_dense_expr_x_sparse_matrix",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N1,N1], // dim
                              None, // sparsity
                              (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                              (0..N1*N1*2).collect(), // subj
                              vec![1.0; N1*N1*2]); // cof

            let m = matrix::sparse(N1,N1,
                                   (1..2*N2+1).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   (0..2*N2).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   vec![1.0; 2*N2].as_slice());
            e.mul(m).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}

fn mul_sparse_expr_x_sparse_matrix(c: &mut Criterion) {
    c.bench_function("mul_sparse_expr_x_sparse_matrix",|b|
        b.iter(|| {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            let e = Expr::new(vec![N2,N2], // dim
                              Some((0..N2*N2).step_by(7).collect::<Vec<usize>>()), // sparsity
                              (0..((N2*N2)/7+1)*2).step_by(2).collect::<Vec<usize>>(), // ptr
                              (0..((N2*N2)/7)*2).collect::<Vec<usize>>(), // subj
                              vec![1.0; (N2*N2/7)*2]); // cof

            let m = matrix::sparse(N1,N1,
                                   (1..2*N2+1).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   (0..2*N2).map(|v| v / 2).collect::<Vec<usize>>().as_slice(),
                                   vec![1.0; 2*N2].as_slice());
            e.mul(m).eval(& mut rs, & mut ws, & mut xs);
        })
    );
}



fn bigmul(c: &mut Criterion) {
    c.bench_function("bigmul",|b|
        b.iter(|| {
            const n : usize = 4096; 
            
            let mut model = Model::new(None);
            let v = model.variable(None,n);
            //let w = model.variable(None,in_psd_cone(n));
            let mx = matrix::dense(n,n,vec![1.0; n*n]);

            let _ = model.constraint(None, &mx.clone().mul(v.clone()).reshape(vec![n]),equal_to(vec![100.0;n]));
            let _ = model.constraint(None, &v.mul(mx).reshape(vec![n]),equal_to(vec![100.0;n]));
        })
    );
}




fn mul_left(c: &mut Criterion) {
    const n : usize = 256;
    c.bench_function("mul_left",|b|
        b.iter(|| {
            let mut model = Model::new(None);
            let mx = matrix::dense(n,n,vec![1.0;n*n]);

            let x = model.variable(None,vec![n,n]);
            let y = model.variable(None,vec![n,n]);
          
            let _ = model.constraint(None,&mx.mul(x.add(y)),equal_to(vec![100.0; n*n]).with_shape(vec![n,n]));
        }));
}

fn mul_right(c: &mut Criterion) {
    const n : usize = 256;
    c.bench_function("mul_right",|b|
        b.iter(|| {
            let mut model = Model::new(None);
            let mx = matrix::dense(n,n,vec![1.0;n*n]);
            
            let x = model.variable(None,vec![n,n]);
            let y = model.variable(None,vec![n,n]);

            let _ = model.constraint(None, &x.add(y).mul(mx),equal_to(vec![100.0;n*n]).with_shape(vec![n,n]));
        }));
}



criterion_group!(benches,
                 bigmul,
                 mul_left,
                 mul_right,
                 mul_dense_matrix_x_dense_expr,
                 mul_dense_matrix_x_sparse_expr,
                 mul_dense_expr_x_dense_matrix,
                 mul_sparse_expr_x_dense_matrix);
criterion_main!(benches);
