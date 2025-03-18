use super::{eval, ExprEvalError, ExprPermuteAxes, ExprReshapeOneRow, ExprTrait};
use super::workstack::WorkStack;
use super::matrix::Matrix;


pub struct ExprMulScalar<const N : usize, E:ExprTrait<N>> {
    pub(super) item : E,
    pub(super) lhs  : f64
}

/// Represents a multiplication of the form
/// $$
/// M\\in\\mathbb{R}^{(m,n)},\\ E(x) \\rightarrow \\mathbb{R}^{(p,n)}:\\ M\\times E(x)^T
/// \\rightarrow \mathbb{R}^{(m,p)}
/// $$
///
/// This combined with transpose can be used to implement both left and right multiplication, but
/// it comes at a cost. For \\(E \\times M\\) computed as \\((M^T\\times E^T)^T\\), we transpose
/// the result, which will often (except for very sparse M) be significantly larger than computing
/// the straight forward product.
pub struct ExprMulMEt<E:ExprTrait<2>> {
    pub(super) item : E,
    pub(super) shape : [usize;2],
    pub(super) data  : Vec<f64>,
    pub(super) sp    : Option<Vec<usize>>
}

pub struct ExprMulLeft<E:ExprTrait<2>> {
    pub(super) item : E,
    
    pub(super) shape : [usize;2],
    pub(super) data  : Vec<f64>,
    pub(super) sp    : Option<Vec<usize>>
}

pub struct ExprMulRight<E:ExprTrait<2>> {
    pub(super) item : E,
    pub(super) shape : [usize;2],
    pub(super) data  : Vec<f64>,
    pub(super) sp    : Option<Vec<usize>>
}

pub struct ExprMulElm<const N : usize,E> where E : ExprTrait<N>+Sized {
    pub(super) expr : E,
    pub(super)datashape : [usize; N],
    pub(super)datasparsity : Option<Vec<usize>>,
    pub(super)data : Vec<f64>
}


// multiply a scalar expression by a shaped value
pub struct ExprScalarMul<const N : usize, E> 
    where E : ExprTrait<0> 
{
    expr : E,
    datashape : [usize; N],
    datasparsity : Option<Vec<usize>>,
    data : Vec<f64>
}

///////////////////////////////////////////////////////////////////////////////
// Left multiplication
//
// SOMETHING.mul(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

/// Trait defining something that can be left-multiplied on an
/// expression.
pub trait ExprLeftMultipliable<const N : usize,E> 
    where E:ExprTrait<N>
{
    type Result;
    fn mul(self,other : E) -> Self::Result;
}

// Multiply matrix and 2D expression
impl<E, M> ExprLeftMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulMEt<ExprPermuteAxes<2,E>>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,sp,data) = self.dissolve();
        ExprMulMEt{ 
            item : ExprPermuteAxes{ item:rhs, perm: [1,0] },
            shape,
            sp,
            data
        }
        //ExprMulLeft{
        //    item : rhs,
        //    shape,
        //    data,
        //    sp}
    }
}

// Multiply matrix and vector expression
impl<E, M> ExprLeftMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<ExprReshapeOneRow<1,2,E>>>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,sp,data) = self.dissolve();
        ExprReshapeOneRow{
            item : ExprMulMEt{
                item : ExprReshapeOneRow{ item: rhs, dim : 1 },
                shape,
                data,
                sp},
            dim : 0
        }
    }
}

// multiply vector and 2D Expression
impl<E> ExprLeftMultipliable<2,E> for Vec<f64>
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<ExprPermuteAxes<2,E>>>;
    fn mul(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulMEt{
                item : ExprPermuteAxes{ item : rhs, perm : [1,0] },
                shape,
                data,
                sp : None},
            dim : 0 }
    }
}

// multiply vector and 2D Expression
impl<E> ExprLeftMultipliable<2,E> for &[f64]
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<ExprPermuteAxes<2,E>>>;
    fn mul(self,rhs : E) -> Self::Result {
        self.to_vec().mul(rhs)
    }
}

impl<const N : usize, E> ExprLeftMultipliable<N,E> for f64
    where E : ExprTrait<N>
{
    type Result = ExprMulScalar<N,E>;
    fn mul(self, rhs : E) -> Self::Result {
        ExprMulScalar{
            item : rhs,
            lhs : self
        }
    }
}




///////////////////////////////////////////////////////////////////////////////
// Right multiplication
//
// E.mul(SOMETHING) where E :ExprTrait
//
// It is used like this: ExprTrait<N> implements
// ``` 
// fn mul(self,rhs:ExprRightMultipliable<N,Self>) -> rhs::Result { rhs.mul_right(self) }
// ```
///////////////////////////////////////////////////////////////////////////////

/// Trait defining something that can be right-multiplied on an
/// expression of dimension N, producing an expression of .
pub trait ExprRightMultipliable<const N : usize,E> 
    where E:ExprTrait<N>
{
    type Result;
    fn mul_right(self,other : E) -> Self::Result;
}

impl<E, M> ExprRightMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    //type Result = ExprPermuteAxes<2,ExprMulMEt<E>>;
    type Result = ExprMulRight<E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        //let (shape,sp,data) = self.transpose().dissolve();
        let (shape,sp,data) = self.dissolve();

        // for f(M,E) = M * E'
        // E * M = (M' * E')' = f(M',E)'

        ExprMulRight{
            item : rhs,
            shape,
            data,
            sp
        }
        //ExprPermuteAxes{
        //    item : ExprMulMEt{
        //        item:rhs,
        //        shape,
        //        data,
        //        sp},
        //    perm : [1,0] }
    }
}

// R = E x M = (M' x E')'
impl<E, M> ExprRightMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<ExprReshapeOneRow<1,2,E>>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let (shape,sp,data) = self.transpose().dissolve();
        ExprReshapeOneRow{
            item : ExprMulMEt{
                item : ExprReshapeOneRow{ item: rhs, dim : 1 },
                shape,
                data,
                sp},
            dim : 0
        }
    }
}

impl<E> ExprRightMultipliable<2,E> for Vec<f64>
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulMEt{
                item : rhs,
                shape,
                data,
                sp : None},
            dim : 0 }
    }
}

impl<E> ExprRightMultipliable<2,E> for &[f64]
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulMEt<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        self.to_vec().mul_right(rhs)
    }
}

impl<E> ExprRightMultipliable<0,E> for &[f64] 
    where
        E : ExprTrait<0>
{
    type Result = ExprScalarMul<1,E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        ExprScalarMul{
            expr : rhs,
            data : self.to_vec(),
            datasparsity : None,
            datashape : [self.len()]
        }
    }
}

impl<E> ExprRightMultipliable<0,E> for Vec<f64> 
    where
        E : ExprTrait<0>
{
    type Result = ExprScalarMul<1,E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let n = self.len();
        ExprScalarMul{
            expr : rhs,
            data : self,
            datasparsity : None,
            datashape : [n]
        }
    }
}

impl<E,M> ExprRightMultipliable<0,E> for &M 
    where
        E : ExprTrait<0>,
        M : Matrix
{
    type Result = ExprScalarMul<2,E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        ExprScalarMul{
            expr         : rhs,
            data         : self.data().to_vec(),
            datasparsity : self.sparsity().map(|v| v.to_vec()),
            datashape    : self.shape()
        }
    }
}

impl<const N : usize, E> ExprRightMultipliable<N,E> for f64
    where E : ExprTrait<N>
{
    type Result = ExprMulScalar<N,E>;
    fn mul_right(self, rhs : E) -> Self::Result {
        ExprMulScalar{
            item : rhs,
            lhs : self
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
// Left element-wise multiplication
//
// SOMETHING.mul_elem(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

pub trait ExprLeftElmMultipliable<const N: usize, E> 
    where E : ExprTrait<N>
{
    type Result;
    fn mul_elem(self, other : E) -> Self::Result;
}

impl<E,M> ExprLeftElmMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulElm<2,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        let (shape,sp,data) = self.dissolve();
        ExprMulElm{
            expr : rhs,
            datashape : shape,
            datasparsity : sp,
            data
        }
    }
}

impl<E> ExprLeftElmMultipliable<1,E> for Vec<f64>
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self
        }
    }
}

impl<E> ExprLeftElmMultipliable<1,E> for &[f64] 
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result { 
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self.to_vec()
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Right element-wise multiplication
//
// SOMETHING.mul_elem(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

pub trait ExprRightElmMultipliable<const N: usize, E> 
    where E : ExprTrait<N>
{
    type Result;
    fn mul_elem(self, other : E) -> Self::Result;
}

impl<E,M> ExprRightElmMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulElm<2,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        let (shape,sp,data) = self.dissolve();
        ExprMulElm{
            expr : rhs,
            datashape : shape,
            datasparsity : sp,
            data
        }
    }
}

impl<E> ExprRightElmMultipliable<1,E> for Vec<f64>
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self
        }
    }
}

impl<E> ExprRightElmMultipliable<1,E> for &[f64] 
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result { 
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self.to_vec()
        }
    }
}



///////////////////////////////////////////////////////////////////////////////
// Trait ExprTrait<N> implementations for
//
// ExprMulLeft
// ExprMulRight
// ExprMulScalar
// ExprMulElm
///////////////////////////////////////////////////////////////////////////////

impl<E> ExprTrait<2> for ExprMulMEt<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.item.eval(ws,rs,xs)?;
        super::eval::mul_matrix_expr_transpose(
            (self.shape[0],self.shape[1]),
            self.sp.as_deref(),
            self.data.as_slice(),
            rs,ws,xs)
    }
}

impl<E> ExprTrait<2> for ExprMulLeft<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.item.eval(ws,rs,xs)?;
        if let Some(ref sp) = self.sp {
            super::eval::mul_left_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs)
        } else {
            super::eval::mul_left_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs)
        }
    }
}

impl<E> ExprTrait<2> for ExprMulRight<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.item.eval(ws,rs,xs)?;
        if let Some(ref sp) = self.sp {
            super::eval::mul_right_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs)
        } else {
            super::eval::mul_right_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs)
        }
    }
}


impl<const N : usize, E:ExprTrait<N>> ExprTrait<N> for ExprMulScalar<N,E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.item.eval(rs,ws,xs)?;
        let (_shape,_ptr,_sp,_subj,cof) = rs.peek_expr_mut();
        cof.iter_mut().for_each(|c| *c *= self.lhs);
        Ok(())
    }
}

impl<const N : usize, E : ExprTrait<N>> ExprTrait<N> for ExprMulElm<N,E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.expr.eval(ws,rs,xs)?;
        super::eval::mul_elem(&self.datashape,
                              if let Some(ref v) = &self.datasparsity { Some(v.as_slice()) } else { None },
                              self.data.as_slice(),
                              rs,ws,xs)
    }
}

impl<const N : usize,E> ExprTrait<N> for ExprScalarMul<N,E> where E : ExprTrait<0> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.expr.eval(ws,rs,xs)?;
        if let Some(ref sp) = self.datasparsity {
            eval::scalar_expr_mul(self.datashape.as_slice(), Some(sp.as_slice()), self.data.as_slice(), rs, ws, xs)
        }
        else {
            eval::scalar_expr_mul(self.datashape.as_slice(), None, self.data.as_slice(), rs, ws, xs)
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
// MulDiag
//
///////////////////////////////////////////////////////////////////////////////

pub trait ExprDiagMultipliable<E> where E : ExprTrait<2> {
    type Result : ExprTrait<1>;
    fn mul_internal(self, other : E) -> Self::Result;
}

#[cfg(test)]
mod test {
    use crate::matrix::*;
    use crate::expr::*;

    #[test]
    fn mul() {
        let m = dense([2, 5], vec![1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0]);
        let e = Expr{ shape: [2,2],
                                    aptr : vec![0,1,2,3,4],
                                    asubj: vec![5,6,7,8],
                                    acof: vec![1.0,1.0,1.0,1.0],
                                    sparsity : None};
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);
        e.mul(m).eval(& mut rs, & mut ws, & mut xs).unwrap();
    }
}

