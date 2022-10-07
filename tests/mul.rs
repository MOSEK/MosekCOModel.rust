extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::expr::workstack::WorkStack;
use mosekmodel::expr::*;
use mosekmodel::matrix::*;

#[test]
fn dense_left_mul() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | x0+x1    x2+x3 |
    // | x4+x5    x6+x7 |
    // | x8+x9  x10+x11 |
    let ed = Expr::new(vec![3,2],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]);
    // | x0+x1      0 |
    // | x2+x3  x4+x5 |
    // |        x6+x7 |
    let es = Expr::new(vec![3,2],
                       Some(vec![0,2,3,5]),
                       vec![0,2,4,6,8,10],
                       vec![0,1,2,3,4,5,6,7,8,9],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]);

    let m = matrix::dense(3,3,vec![1.1,1.2,1.3,
                                   2.1,2.2,2.3,
                                   3.1,3.2,3.3]);
    m.mul(ed).eval_finalize(& mut rs, & mut ws, & mut xs);
    {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);
        assert!(shape == &[3,2]);
        assert!(ptr   == &[0,6,12,18,24,30,36]);
        assert!(sp.is_none());
        assert!(subj  == &[0,1,4,5,8,9 , 2,3,6,7,10,11,
                           0,1,4,5,8,9 , 2,3,6,7,10,11,
                           0,1,4,5,8,9 , 2,3,6,7,10,11 ]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());


    m.mul(es).eval_finalize(& mut rs, & mut ws, & mut xs);
    {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);
        assert!(shape == &[3,2]);
        assert!(ptr   == &[0,4,8,12,16,20,24]);
        assert!(sp.is_none());
        assert!(subj  == &[0,1,4,5,8,9 , 2,3,6,7,10,11,
                           0,1,4,5,8,9 , 2,3,6,7,10,11,
                           0,1,4,5,8,9 , 2,3,6,7,10,11 ]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());
}

//#[test]
//fn dense_mul_right() {
//    let mut rs = WorkStack::new(1024);
//    let mut ws = WorkStack::new(1024);
//    let mut xs = WorkStack::new(1024);
//
//    let v = Variable::new((0..n).collect(), None, vec![n]);
//    let w = Variable::new((0..n).collect(), None, vec![n]);
//
//    let mx = matrix::dense(n,n,vec![1.0; n*n]);
//
//    v.mul(mx).eval_finalize(& mut rs, & mut ws, & mut xs);
//}
//
//
//#[test]
//fn sparse_left_mul() {
//    let mut rs = WorkStack::new(1024);
//    let mut ws = WorkStack::new(1024);
//    let mut xs = WorkStack::new(1024);
//
//    let v = Variable::new((0..n).collect(), None, vec![n]);
//    let w = Variable::new((0..n).collect(), None, vec![n]);
//
//    let ms = matrix::sparse(n,n+1,
//                            (0..n));
//
//    mx.mul(v).eval_finalize(& mut rs, & mut ws, & mut xs);
//}
//
//#[test]
//fn sparse_mul_right() {
//    let mut rs = WorkStack::new(1024);
//    let mut ws = WorkStack::new(1024);
//    let mut xs = WorkStack::new(1024);
//
//    let v = Variable::new((0..n).collect(), None, vec![n]);
//    let w = Variable::new((0..n).collect(), None, vec![n]);
//
//    let mx = matrix::dense(n,n,vec![1.0; n*n]);
//    let ms = matrix::sparse(mx);
//
//    v.mul(mx).eval_finalize(& mut rs, & mut ws, & mut xs);
//}
//
