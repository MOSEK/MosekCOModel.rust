extern crate mosekcomodel;

use mosekcomodel::*;
use mosekcomodel_mosek::Model;

fn main() -> Result<(),String> {
    let mut m = Model::new(Some("SuperModel"));

    let _x1 = m.variable(Some("x1"), [4,5]);
    let _x2 = m.variable(Some("x2"), [4]);
    let _y1 = m.variable(Some("y1"), greater_than(vec![1.0,2.0,3.0,4.0]));
    let _y2 = m.variable(Some("y2"), greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(&[2,2]));
    let _z  = m.variable(Some("z"),  greater_than(vec![1.0,4.0]).with_shape_and_sparsity(&[2,2],&[[0,0],[1,1]]));
    Ok(())
}
#[test]
fn test() { main().unwrap(); }
