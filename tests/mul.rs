extern crate mosekcomodel;

use matrix::NDArray;
use mosekcomodel::*;
use mosekcomodel::expr::workstack::WorkStack;
use crate::dummy::Model;

const N1 : usize = 100;
const N2 : usize = 100;

#[test]
fn mul_dense_matrix_x_dense_expr() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let e = Expr::new(&[N1,N1], // dim
                      None, // sparsity
                      (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                      (0..N1*N1*2).collect(), // subj
                      vec![1.0; N1*N1*2]); // cof

    let m = matrix::dense([N1,N1], vec![1.0; N1*N1]);
    m.mul(e).eval(& mut rs, & mut ws, & mut xs).unwrap();
}

#[test]
fn mul_dense_matrix_x_sparse_expr() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let e = Expr::new(&[N2,N2], // dim
                      Some((0..N2*N2).step_by(7).collect()), // sparsity
                      (0..((N2*N2)/7+2)*2).step_by(2).collect(), // ptr
                      (0..((N2*N2)/7+1)*2).collect(), // subj
                      vec![1.0; (N2*N2/7+1)*2]); // cof

    let m = matrix::dense([N2,N2], vec![1.0; N2*N2]);
    m.mul(e).eval(& mut rs, & mut ws, & mut xs).unwrap();
}

#[test]
fn mul_dense_expr_x_dense_matrix() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let e = Expr::new(&[N1,N1], // dim
                      None, // sparsity
                      (0..(N1*N1*2+2)).step_by(2).collect(), // ptr
                      (0..N1*N1*2).collect(), // subj
                      vec![1.0; N1*N1*2]); // cof

    let m = matrix::dense([N1,N1], vec![1.0; N1*N1]);
    e.mul(m).eval(& mut rs, & mut ws, & mut xs).unwrap();
}

#[test]
fn mul_sparse_expr_x_dense_matrix() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    let e = Expr::new(&[N2,N2], // dim
                      Some((0..N2*N2).step_by(7).collect()), // sparsity
                      (0..((N2*N2)/7+2)*2).step_by(2).collect(), // ptr
                      (0..((N2*N2)/7+1)*2).collect(), // subj
                      vec![1.0; (N2*N2/7+1)*2]); // cof

    let m = matrix::dense([N2,N2], vec![1.0; N2*N2]);
    e.mul(m).eval(& mut rs, & mut ws, & mut xs).unwrap();
}


#[test]
fn dense_left_mul() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | x0+x1    x2+x3 |
    // | x4+x5    x6+x7 |
    // | x8+x9  x10+x11 |
    let ed = Expr::new(&[3,2],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0; 12]);
    // | x0+x1      0 |
    // | x2+x3  x4+x5 |
    // |        x6+x7 |
    let es = Expr::new(&[3,2],
                       Some(vec![0,2,3,5]),
                       vec![0,2,4,6,8],
                       vec![0,1,2,3,4,5,6,7],
                       vec![1.0; 8]);

    let m = matrix::dense([3,3],vec![1.1,1.2,1.3,
                                   2.1,2.2,2.3,
                                   3.1,3.2,3.3]);

    m.clone().mul(ed).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        assert!(shape == [3,2]);
        assert!(ptr   == [0,6,12,18,24,30,36]);
        assert!(sp.is_none());
        assert!(subj  == [0,1,4,5,8,9 , 2,3,6,7,10,11,
                          0,1,4,5,8,9 , 2,3,6,7,10,11,
                          0,1,4,5,8,9 , 2,3,6,7,10,11 ]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());

    m.mul(es).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        assert!(shape == [3,2]);
        assert!(ptr   == [0,4,8,12,16,20,24]);
        assert!(sp.is_none());
        assert!(subj  == [0,1,2,3, 4,5,6,7,
                          0,1,2,3, 4,5,6,7,
                          0,1,2,3, 4,5,6,7]);
        //assert!(cof == [1.1*1.0,1.1*1.0,1.2*2.0,1.2*2.0, 1.2*3.0,1.2*3.0,1.3*4.0,1.3*4.0,
        //                2.1*1.0,2.1*1.0,2.2*2.0,2.2*2.0, 2.2*3.0,2.2*3.0,2.3*4.0,2.3*4.0,
        //                3.1*1.0,3.1*1.0,3.2*2.0,3.2*2.0, 3.2*3.0,3.2*3.0,3.3*4.0,3.3*4.0]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());
}


#[test]
fn basic_expr() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | x0+x1  x2+x3    x4+x5 |
    // | x6+x7  x8+x9  x10+x11 |
    let ed = Expr::new(&[2,3],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]);
    // | x0+x1  x2+x3        |
    // |        x4+x5  x6+x7 |
    let _es = Expr::new(&[2,3],
                       Some(vec![0,1,4,5]),
                       vec![0,2,4,6,8],
                       vec![0,1,2,3,4,5,6,7],
                       vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0]);

    let _m = matrix::dense([3,3],vec![1.1,1.2,1.3,
                                   2.1,2.2,2.3,
                                   3.1,3.2,3.3]);
    ed.eval(& mut rs, & mut ws, & mut xs).unwrap();
    println!("Workstack = {}",rs);
    _ = rs.pop_expr();
    //es.eval(& mut rs, & mut ws, & mut xs);
    //_ = rs.pop_expr();
}
#[test]
fn dense_right_mul() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | x0+x1  x2+x3    x4+x5 |
    // | x6+x7  x8+x9  x10+x11 |
    let ed = Expr::new(&[2,3],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]);
    // | 1(x0+x1)  2(x2+x3)           |
    // |           3(x4+x5)  4(x6+x7) |
    let es = Expr::new(&[2,3],
                       Some(vec![0,1,4,5]),
                       vec![0,2,4,6,8],
                       vec![0,1,2,3,4,5,6,7],
                       vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0]);

    let m = matrix::dense([3,3],vec![1.1,1.2,1.3,
                                   2.1,2.2,2.3,
                                   3.1,3.2,3.3]);
    ed.mul(m.clone()).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);

        assert!(shape == [2,3]);
        assert!(ptr   == [0,6,12,18,24,30,36]);
        assert!(sp.is_none());
        assert!(subj  == [0,1,2,3,4,5,    0,1,2,3,4,5,    0,1,2,3,4,5,
                          6,7,8,9,10,11,  6,7,8,9,10,11,  6,7,8,9,10,11]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());


    // | 1.1 2.1 3.1 | |1(x0+x1)  2(x2+x3)           |
    // | 1.2 2.2 3.2 | |          3(x4+x5)  4(x6+x7) |
    // | 1.3 2.3 3.3 |
    //
    // 1.1(x0+x1)+2.1*2(x2+x3)  2.1*3(x4+x5)+3.2*4(x6+x7) |
    // 1.2(x0+x1)+2.2*2(x2+x3)  2.2*3(x4+x5)+3.2*4(x6+x7) |
    // 1.3(x0+x1)+2.3*2(x2+x3)  2.3*3(x4+x5)+3.3*4(x6+x7) |
    es.mul(m).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);
        assert_eq!(shape, &[2,3]);
        assert_eq!(ptr,   &[0,4,8,12,16,20,24]);
        assert!(sp.is_none());
        assert_eq!(subj,&[0,1,2,3, 0,1,2,3, 0,1,2,3,
                          4,5,6,7, 4,5,6,7, 4,5,6,7]);
        assert_eq!(cof,&[1.1*1.0,1.1*1.0,2.1*2.0,2.1*2.0, 1.2*1.0,1.2*1.0,2.2*2.0,2.2*2.0, 1.3*1.0,1.3*1.0,2.3*2.0,2.3*2.0, 
                        2.1*3.0,2.1*3.0,3.1*4.0,3.1*4.0, 2.2*3.0,2.2*3.0,3.2*4.0,3.2*4.0, 2.3*3.0,2.3*3.0,3.3*4.0,3.3*4.0 ]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());
}






#[test]
fn sparse_left_mul() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | 1 x0 + 2 x1   3 x2 + 4x3   |
    // | 5 x4 + 6 x5   7 x6 + 8x7   |
    // | 9 x8 +10 x9  11 x10+12 x11 |
    let ed = Expr::new(&[3,2],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]);
    // | x0+x1      0 |
    // | x2+x3  x4+x5 |
    // |        x6+x7 |
    let es = Expr::new(&[3,2],
                       Some(vec![0,2,3,5]),
                       vec![0,2,4,6,8],
                       vec![0,1,2,3,4,5,6,7],
                       vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0]);

    // | 1.1     1.3 |
    // |         2.3 |
    // | 3.1 3.2     |
    let m = NDArray::from_tuples([3,3],
                           &[[0,0],[0,2],[1,2],[2,0],[2,1]],
                           &[1.1,1.3,2.3,3.1,3.2]).unwrap();
    m.clone().mul(ed).eval(& mut rs, & mut ws, & mut xs).unwrap();
    // | 1.1(x0+x1)
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        assert!(shape == [3,2]);
        assert!(ptr   == [0,4,8, 10,12, 16,20]);
        assert!(sp.is_none());
        assert!(subj  == [0,1,8,9, 2,3,10,11,
                          8,9,     10,11,
                          0,1,4,5, 2,3,6,7]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());

    m.mul(es).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();

        assert!(shape == [3,2]);
        assert!(ptr   == [0,2,4,6,10,12]);
        assert!(sp    == Some(&[0,1,3,4,5]));
        assert!(subj  == [0,1,     6,7,
                                   6,7,
                          0,1,2,3, 4,5]);

        println!("cof = {:?}",cof);
        println!("   != {:?}",&[1.1*1.0,1.1*1.0, 3.1*4.0,3.1*4.0, 
                                         2.3*4.0,2.3*4.0,   
                        3.1*1.0,3.1*1.0,3.2*2.0,3.2*2.0, 3.2*3.0,3.2*3.0]);
        assert!(cof == [1.1*1.0,1.1*1.0, 1.3*4.0,1.3*4.0, 
                                         2.3*4.0,2.3*4.0,   
                        3.1*1.0,3.1*1.0,3.2*2.0,3.2*2.0, 3.2*3.0,3.2*3.0]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());
}

#[test]
fn sparse_right_mul() {
    let mut rs = WorkStack::new(1024);
    let mut ws = WorkStack::new(1024);
    let mut xs = WorkStack::new(1024);

    // | x0+x1  x2+x3    x4+x5 |
    // | x6+x7  x8+x9  x10+x11 |
    let ed = Expr::new(&[2,3],
                       None,
                       vec![0,2,4,6,8,10,12],
                       vec![0,1,2,3,4,5,6,7,8,9,10,11],
                       vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]);
    // | x0+x1  x2+x3        |
    // |        x4+x5  x6+x7 |
    let es = Expr::new(&[2,3],
                       Some(vec![0,1,4,5]),
                       vec![0,2,4,6,8],
                       vec![0,1,2,3,4,5,6,7],
                       vec![1.0,1.0,2.0,2.0,3.0,3.0,4.0,4.0]);

    // | 1.1     1.3 |
    // |         2.3 |
    // | 3.1 3.2     |
    let m = NDArray::from_tuples([3,3],
                                 &[[0,0],[0,2],[1,2],[2,0],[2,1]],
                                 &[1.1,1.3,2.3,3.1,3.2]).unwrap();

    ed.mul(m.clone()).eval(& mut rs, & mut ws, & mut xs).unwrap();
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("sp    = {:?}",sp);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);

        assert!(shape == [2,3]);
        assert!(ptr   == [0,4,6,10, 14,16,20]);
        assert!(sp.is_none());
        assert!(subj  == [0,1,4,5,   4,5,   0,1,2,3,
                          6,7,10,11, 10,11, 6,7,8,9]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());


    es.mul(m).eval(& mut rs, & mut ws, & mut xs).unwrap();
    // | 1.1(x0+x1)            3.2(x2+x3) |
    // | 3.1(x6+x7) 3.2(x6+x7) 2.3(x4+x6) |
    {
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();

        println!("shape = {:?}",shape);
        println!("sp    = {:?}",sp);
        println!("ptr   = {:?}",ptr);
        println!("subj  = {:?}",subj);
        assert!(shape == [2,3]);
        assert!(ptr   == [0,2,6,8,10,12]);
        assert!(sp == Some(&[0,2,3,4,5]));
        assert!(subj  == [0,1,      0,1,2,3,
                          6,7, 6,7, 4,5]);
    }
    assert!(rs.is_empty());
    assert!(ws.is_empty());
}


#[test]
fn bigmul() {
    const N : usize = 128; 
    
    let mut model = Model::new(None);
    let v = model.variable(None,&[N,1]);
    let mx = matrix::dense([N,N],vec![1.0; N*N]);

    let _ = model.constraint(None, mx.clone().mul(&v).reshape(&[N]),equal_to(vec![100.0;N]));
    let _ = model.constraint(None, v.reshape(&[1,N]).mul(mx), equal_to(vec![100.0;N]).with_shape(&[1,N]));
}


