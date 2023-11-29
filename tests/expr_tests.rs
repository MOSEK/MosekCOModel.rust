use mosekmodel::*;

#[test]
fn mul_left() {
    let mut M = Model::new(Some("M"));

    //      | 1 2 3 |       | 10 11 12 |
    // vd = | 4 5 6 |  wd = | 13 14 15 |
    //      | 7 8 9 |       | 16 17 18 |
    let vd = M.variable(Some("vd"), unbounded().with_shape(&[3,3]));
    let wd = M.variable(Some("wd"), unbounded().with_shape(&[3,3]));
    let vs = M.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[1,2],[2,2]]));
    let ws = M.variable(Some("vs"), unbounded().with_shape_and_sparsity(&[3,3],&[[0,0],[1,1],[1,2],[2,2]]));

    {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        

    }

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
