use mosekmodel::{*, expr::workstack::WorkStack,expr::*};

fn dense_expr() -> Expr<2> {
    Expr::new(&[3,3],
              None,
              vec![0,1,2,3,4,5,6,7,8,9],
              vec![0,1,2,0,1,2,0,1,2],
              vec![1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3])
}

fn sparse_expr() -> Expr<2> {
    Expr::new(&[3,3],
              Some(vec![0,4,5,6,7]),
              vec![0,1,2,3,4,5],
              vec![0,1,2,3,4],
              vec![1.1,2.2,3.3,4.4,5.5])
}

#[test]
fn add_test() {
    let mut m = Model::new(Some("M"));

    //      | 1 2 3 |       | 10 11 12 |
    // vd = | 4 5 6 |  wd = | 13 14 15 |
    //      | 7 8 9 |       | 16 17 18 |
    let dv = m.variable(Some("vd"), unbounded().with_shape(&[3,3]));
    let dw = m.variable(Some("wd"), unbounded().with_shape(&[3,3]));
    //      | 19  .  . |      | 23  .  . |
    // vs = |  . 20 21 | ws = |  . 24  . |
    //      |  .  . 22 |      |  . 25 26 | 
    let sv = m.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[1,2],[2,2]]));
    let sw = m.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[2,1],[2,2]]));

    let mut rs = WorkStack::new(512);
    let mut ws = WorkStack::new(512);
    let mut xs = WorkStack::new(512);
    {
        rs.clear(); ws.clear(); xs.clear();
            
        dv.clone().add(dw.clone()).eval(&mut rs,&mut ws,&mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();        

        assert_eq!(shape,&[3,3]); 
        assert_eq!(sp,None);
        assert_eq!(ptr,&[0usize,2,4,6,8,10,12,14,16,18]);
        assert_eq!(subj,&[ 1,10, 2,11, 3,12, 4,13, 5,14, 6,15, 7,16, 8,17, 9,18]);
    }
    
    {
        rs.clear(); ws.clear(); xs.clear();
        
        dv.clone().add(sw.clone()).eval(&mut rs,&mut ws,&mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();        

        assert_eq!(shape,&[3,3]); 
        assert_eq!(sp,None);
        assert_eq!(ptr,&[0usize,2,3,4,5,7,8,9,11,13]);
        assert_eq!(subj,&[ 1,23,2,3,4,5,24,6,7,8,25,9,26]);
    }

    {
        rs.clear(); ws.clear(); xs.clear();
        
        sv.clone().add(sw.clone()).eval(&mut rs,&mut ws,&mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();        

        assert_eq!(shape,&[3,3]); 
        if let Some(sp) = sp {
            assert_eq!(sp,&[0usize,4,5,7,8]);
        } else {
            panic!("sp is not None");
        }
        assert_eq!(ptr,&[0,2,4,5,6,7]);
        assert_eq!(subj,&[ 19,23,20,24,21,25,22,26]);
    }
}



#[test]
fn mul_left() {
    let mut M = Model::new(Some("M"));

    //      | 1 2 3 |       | 10 11 12 |
    // vd = | 4 5 6 |  wd = | 13 14 15 |
    //      | 7 8 9 |       | 16 17 18 |
    let dv = M.variable(Some("vd"), unbounded().with_shape(&[3,3]));
    let dw = M.variable(Some("wd"), unbounded().with_shape(&[3,3]));
    let sv = M.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[1,2],[2,2]]));
    let sw = M.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[1,2],[2,2]]));

    let mut rs = WorkStack::new(512);
    let mut ws = WorkStack::new(512);
    let mut xs = WorkStack::new(512);

    let e0 = dense_expr();
    let e1 = sparse_expr();

    let m1 = matrix::dense(3,2,vec![1.0,2.0,3.0,4.0,5.0,6.0]);
    let m2 = matrix::dense(2,3,vec![1.0,2.0,3.0,4.0,5.0,6.0]);

    let e0_1 = m2.clone().mul(e0.clone());
    let e0_2 = e0.clone().mul(2.0);

    let e1_1 = m2.clone().mul(e1.clone());
    let e1_2 = e1.clone().mul(2.0);

    e0.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    e1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    e0_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    e0_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    e1_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    e1_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
}
