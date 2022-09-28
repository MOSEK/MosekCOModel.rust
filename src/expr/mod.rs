extern crate itertools;

mod eval;
pub mod workstack;

use itertools::{iproduct,izip};
use super::utils::*;
use super::Variable;
use workstack::WorkStack;


pub trait ExprTrait : Sized {
    /// Evaluate the expression and put the result on the `rs` stack,
    /// using the `ws` to evaluate sub-expressions and `xs` for
    /// general storage.
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack);
    /// Evaluate the expression, then clean it up and put
    /// it on the `rs` stack. The result will guarantee that
    /// - non-zeros in each row are sorted by `subj`
    /// - expression contains no zeros or duplicate nonzeros.
    /// - the expression is dense
    fn eval_finalize(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.eval(ws,rs,xs);
        eval::eval_finalize(rs,ws,xs);
    }

    // fn into_diag(self) -> ExprIntoDiag<Self> { ExprIntoDiag{ item : self } }
    // fn reshape(self, shape : &[usize]) -> ExprReshape<Self>  { ExprReshape{  item : self, shape : shape.to_vec() } }
    // fn mul_scalar(self, c : f64) -> ExprMulScalar<Self> { ExprMulScalar{ item:self, c : c } }
    // fn mul_vec_left(self, v : Vec<f64>) -> ExprMulVec<Self>
    // fn mul_matrix_left(self, matrix : Matrix) -> ExprMulMatrixLeft<Self>
    // fn mul_matrix_right(self, matrix : Matrix) -> ExprMulMatrixRight<Self>
    // fn transpose(self) -> ExprPermuteAxes<Self>
    // fn axispermute(self) -> ExprPermuteAxes<Self>

    fn add<R:ExprTrait>(self,rhs : R) -> ExprAdd<Self,R> {
        ExprAdd{lhs:self,rhs}
    }
}


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// Expression objects


/// Expr defines a literal expression with no sub-expressions
#[derive(Clone)]
pub struct Expr {
    aptr  : Vec<usize>,
    asubj : Vec<usize>,
    acof  : Vec<f64>,
    shape : Vec<usize>,
    sparsity : Option<Vec<usize>>
}

/// The Expr implementation d
impl Expr {
    pub fn new(aptr  : Vec<usize>,
               asubj : Vec<usize>,
               acof  : Vec<f64>) -> Expr {
        if aptr.len() == 0 { panic!("Invalid aptr"); }
        if ! aptr[0..aptr.len()-1].iter().zip(aptr[1..].iter()).all(|(a,b)| a <= b) {
            panic!("Invalid aptr: Not sorted");
        }
        let & sz = aptr.last().unwrap();
        if sz != asubj.len() || asubj.len() != acof.len() {
            panic!("Mismatching aptr, asubj and acof");
        }

        Expr{
            aptr,
            asubj,
            acof,
            shape : (0..sz).collect(),
            sparsity : None
        }
    }

    pub fn from_variable(variable : &Variable) -> Expr {
        let sz = variable.shape.iter().product();

        match variable.sparsity {
            None =>
                Expr{
                    aptr  : (0..sz+1).collect(),
                    asubj : variable.idxs.clone(),
                    acof  : vec![1.0; sz],
                    shape : variable.shape.clone(),
                    sparsity : None
                },
            Some(ref sp) => {
                Expr{
                    aptr  : (0..sp.len()+1).collect(),
                    asubj : variable.idxs.clone(),
                    acof  : vec![1.0; sp.len()],
                    shape : variable.shape.clone(),
                    sparsity : Some(sp.clone())
                }
            }
        }
    }

    pub fn into_diag(self) -> Expr {
        if self.shape.len() != 1 {
            panic!("Diagonals can only be made from vector expressions");
        }

        let d = self.shape[0];
        Expr{
            aptr : self.aptr,
            asubj : self.asubj,
            acof : self.acof,
            shape : vec![d,d],
            sparsity : Some((0..d*d).step_by(d+1).collect())
        }
    }

    pub fn reshape(self,shape:&[usize]) -> Expr {
        if self.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            panic!("Invalid shape for this expression");
        }

        Expr{
            aptr : self.aptr,
            asubj : self.asubj,
            acof : self.acof,
            shape : shape.to_vec(),
            sparsity : self.sparsity
        }
    }
}

impl ExprTrait for Expr {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let nnz  = self.asubj.len();
        let nelm = self.aptr.len()-1;

        let (aptr,sp,asubj,acof) = rs.alloc_expr(self.shape.as_slice(),nnz,nelm);

        match (&self.sparsity,sp) {
            (Some(ref ssp),Some(dsp)) => dsp.clone_from_slice(ssp.as_slice()),
            _ => {}
        }

        aptr.clone_from_slice(self.aptr.as_slice());
        asubj.clone_from_slice(self.asubj.as_slice());
        acof.clone_from_slice(self.acof.as_slice());
    }
}


impl expr::ExprTrait for super::Variable {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),
                                                  self.idxs.len(),
                                                  self.idxs.len());
        rptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        rsubj.clone_from_slice(self.idxs.as_slice());
        rcof.fill(1.0);
        match (rsp,&self.sparsity) {
            (Some(rsp),Some(sp)) => rsp.clone_from_slice(sp.as_slice()),
            _ => {}
        }
    }
}


////////////////////////////////////////////////////////////
// Multiply

/// Trait that indicates that the `v:T` implementing it supports
/// v.mul(expr)
pub trait ExprMultiplyableLeft {
    type O : ExprTrait;
    fn mul<T:ExprTrait>(self,rhs : E) -> O;
}

/// Trait that indicates that the type implementing T it supports
/// expr.mul(t.mul(expr)
pub trait ExprMultiplyableRight {
    type O : ExprTrait;
    fn mul<T:ExprTrait>(self,lhs : E) -> O;
}

pub trait Matrix {
    fn size(&self) -> (usize,usize);
    fn numnz(&self) -> usize;
    fn issparse(&self) -> bool;
    fn sparsity(&self,& mut [usize]);
    fn values<'a>(&self) -> &'a[f64];

    pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        DenseMatrix{
            dim  : (height,width),
            data : data
        }
    }
    pub fn ones(height : usize, width : usize) -> DenseMatrix {
        DenseMatrix{
            dim : (height,width),
            data : vec![1.0; height*width]
        }
    }
    pub fn diag(data : &[f64]) -> SparseMatrix {
        SparseMatrix{
            dim : (data.len(),data.len()),
            sp  : (0..data.len()*data.len()).step_by(data.len()+1).collect(),
            data : data.to_vec()
        }
    }
}

#[drive(Clone)]
pub struct DenseMatrix {
    dim  : (usize,usize),
    data : Vec<f64>
}

#[drive(Clone)]
pub struct SparseMatrix {
    dim  : (usize,usize),
    sp   : Vec<usize>,
    data : Vec<f64>,
}

impl DenseMatrix {
    pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        Matrix{
            dim : (height,width),
            rows : true,
            data : data
        }
    }
    pub fn ones(height : usize, width : usize) -> DenseMatrix {
        Matrix{
            dim : (height,width),
            rows : true,
            data : vec![1.0; height*width]
        }
    }
    pub fn diag(data : &[f64]) -> Matrix {
        Matrix{
            dim : (data.len(),data.len()),
            rows : true,
            data : iproduct!((0..data.len()).zip(data.iter()),0..data.len()).map(|((i,&c),j)| if i == j {c} else { 0.0 }).collect()
        }
    }
}

pub struct ExprMulLeft<E:ExprTrait,M:MatrixTrait> {
    item : E,
    lhs  : M
}

struct ExprMulRight<E:ExprTrait,M:MatrixTrait> {
    item : E,
    rhs  : M
}

pub struct ExprMulScalar<E:ExprTrait> {
    item : E,
    lhs  : f64
}

impl<E:ExprTrait> ExprTrait for ExprMulLeft<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::mul_left(&self.lhs,rs,ws,xs);
    }
}

impl<E:ExprTrait> ExprTrait for ExprMulRight<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::mul_right(&self.lhs,rs,ws,xs);
    }
}

// inplace evaluation
impl<E:ExprTrait> ExprTrait for ExprMulScalar<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        let (_shape,_ptr,_sp,_subj,cof) = rs.peek_expr_mut();
        cof.iter_mut().for_each(|c| *c *= self.lhs)
    }
}

pub struct ExprDotVec<E:ExprTrait> {
    data : Vec<f64>,
    expr : E
}

impl<E:ExprTrait> ExprTrait for ExprDot<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::dot_slice(self.data.as_slice(),rs,ws,xs);
    }
}



////////////////////////////////////////////////////////////
//
// ExprAdd is constructed for `e,d : ExprTrait` by
// ```
//   e.add(d)
// ```
// The following construction is meant to turn a chain of adds like this
// ```
//   e.add(e1).add(e2).add(e3)
// ```
// which would end up as a structure
// ```
//   ExprAdd(ExprAdd(ExprAdd(e,e1),e2),e3)
// ```
//
// which would by default be evaluated one expression at a time, into
// a construction that is aware of the recursion:
// ```
//   ExprAddRec(ExprAddRec(ExprAdd(e,e1),e2),e3)
// ```
// ExprAddRec will have a specialized `eval` function that first
// evaluates the whole chain of terms, then adds them
//
// For this purpose we use a private trait implemented only by ExprAdd
// and ExprAddRec providing a recursive evaluation function.

pub trait ExprAddRecTrait {
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize;
}

pub struct ExprAdd<L:ExprTrait+Sized,R:ExprTrait> {
    lhs : L,
    rhs : R
}
pub struct ExprAddRec<L:ExprAddRecTrait,R:ExprTrait> {
    lhs : L,
    rhs : R
}

// ExprAdd implementation
impl<L:ExprTrait,R:ExprTrait> ExprAdd<L,R> {
    pub fn add<T:ExprTrait>(self,rhs : T) -> ExprAddRec<ExprAdd<L,R>,T> {
        ExprAddRec{lhs: self, rhs}
    }
}

impl<L:ExprTrait,R:ExprTrait> ExprTrait for ExprAdd<L,R> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.lhs.eval(ws,rs,xs);
        self.rhs.eval(ws,rs,xs);

        eval::add(2,rs,ws,xs);
    }
}
impl<L:ExprTrait,R:ExprTrait> ExprAddRecTrait for ExprAdd<L,R> {
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.rhs.eval(rs,ws,xs);
        self.lhs.eval(rs,ws,xs);
        2
    }
}
// ExprAddRec implementation
impl<L:ExprAddRecTrait,R:ExprTrait>  ExprAddRec<L,R> {
    fn add<T:ExprTrait>(self,rhs : T) -> ExprAddRec<Self,T> {
        ExprAddRec{lhs: self, rhs}
    }
}

impl<L:ExprAddRecTrait,R:ExprTrait> ExprAddRecTrait for ExprAddRec<L,R> {
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.rhs.eval(rs,ws,xs);
        1+self.lhs.eval_rec(rs,ws,xs)
    }
}

impl<L:ExprAddRecTrait,R:ExprTrait> ExprTrait for ExprAddRec<L,R> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(ws,rs,xs);

        eval::add(n,rs,ws,xs);
    }
}






#[cfg(test)]
mod test {
    #[test]
    fn test_exprs() {

    }
}
