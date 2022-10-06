extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;


#[test]
fn dense_left_mul() {
    let mut rs = WorkSpace::new(1024);
    let mut ws = WorkSpace::new(1024);
    let mut xs = WorkSpace::new(1024);

    let v = Variable::new((0..n).collect(), None, vec![n]);
    let w = Variable::new((0..n).collect(), None, vec![n]);

    let mx = matrix::dense(n,n,vec![1.0; n*n]);

    mx.mul(v).eval_finalize(& mut rs, & mut ws, & mut xs);
}

#[test]
fn dense_mul_right() {
    let mut rs = WorkSpace::new(1024);
    let mut ws = WorkSpace::new(1024);
    let mut xs = WorkSpace::new(1024);

    let v = Variable::new((0..n).collect(), None, vec![n]);
    let w = Variable::new((0..n).collect(), None, vec![n]);

    let mx = matrix::dense(n,n,vec![1.0; n*n]);

    v.mul(mx).eval_finalize(& mut rs, & mut ws, & mut xs);
}


#[test]
fn sparse_left_mul() {
    let mut rs = WorkSpace::new(1024);
    let mut ws = WorkSpace::new(1024);
    let mut xs = WorkSpace::new(1024);

    let v = Variable::new((0..n).collect(), None, vec![n]);
    let w = Variable::new((0..n).collect(), None, vec![n]);

    let ms = matrix::sparse(n,n+1,
                            (0..n));

    mx.mul(v).eval_finalize(& mut rs, & mut ws, & mut xs);
}

#[test]
fn sparse_mul_right() {
    let mut rs = WorkSpace::new(1024);
    let mut ws = WorkSpace::new(1024);
    let mut xs = WorkSpace::new(1024);

    let v = Variable::new((0..n).collect(), None, vec![n]);
    let w = Variable::new((0..n).collect(), None, vec![n]);

    let mx = matrix::dense(n,n,vec![1.0; n*n]);
    let ms = matrix::sparse(mx);

    v.mul(mx).eval_finalize(& mut rs, & mut ws, & mut xs);
}

