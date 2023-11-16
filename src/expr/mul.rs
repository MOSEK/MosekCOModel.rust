use super::{ExprTrait, ExprReshapeOneRow};
use super::workstack::WorkStack;
use super::matrix::Matrix;

pub struct ExprMulScalar<const N : usize, E:ExprTrait<N>> {
    item : E,
    lhs  : f64
}

pub struct ExprMulLeft<E:ExprTrait<2>> {
    item : E,
    
    shape : [usize;2],
    data  : Vec<f64>,
    sp    : Option<Vec<usize>>
}

pub struct ExprMulRight<E:ExprTrait<2>> {
    item : E,
    shape : [usize;2],
    data  : Vec<f64>,
    sp    : Option<Vec<usize>>
}

pub struct ExprMulElm<const N : usize,E> where E : ExprTrait<N> {
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

impl<E, M> ExprLeftMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulLeft<E>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulLeft{
            item : rhs,
            shape,
            data,
            sp}
    }
}

impl<E, M> ExprLeftMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<ExprReshapeOneRow<1,2,E>>>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : ExprReshapeOneRow{ item: rhs, dim : 0 },
                shape,
                data,
                sp},
            dim : 0
        }
    }
}

impl<E> ExprLeftMultipliable<2,E> for Vec<f64>
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<E>>;
    fn mul(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : rhs,
                shape,
                data,
                sp : None},
            dim : 0 }
    }
}

impl<E> ExprLeftMultipliable<2,E> for &[f64]
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<E>>;
    fn mul(self,rhs : E) -> Self::Result {
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : rhs,
                shape : [1,self.len()],
                data : self.to_vec(),
                sp : None},
            dim : 0 }
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
    type Result = ExprMulRight<E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulRight{
            item : rhs,
            shape,
            data,
            sp}
    }
}

impl<E, M> ExprRightMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<ExprReshapeOneRow<1,2,E>>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : ExprReshapeOneRow{ item: rhs, dim : 0 },
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
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulRight{
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
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : rhs,
                shape : [1,self.len()],
                data : self.to_vec(),
                sp : None},
            dim : 0 }
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
        let (shape,data,sp) = self.extract();
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
        let (shape,data,sp) = self.extract();
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


impl<E> ExprTrait<2> for ExprMulLeft<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        if let Some(ref sp) = self.sp {
            super::eval::mul_left_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs);
        } else {
            super::eval::mul_left_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs);
        }
    }
}

impl<E> ExprTrait<2> for ExprMulRight<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        if let Some(ref sp) = self.sp {
            super::eval::mul_right_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs);
        } else {
            super::eval::mul_right_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs);
        }
    }
}


impl<const N : usize, E:ExprTrait<N>> ExprTrait<N> for ExprMulScalar<N,E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        let (_shape,_ptr,_sp,_subj,cof) = rs.peek_expr_mut();
        cof.iter_mut().for_each(|c| *c *= self.lhs)
    }
}


////////////////////////////////////////////////////////////

//
///// Implement `Expr<N>.mul(s)` for scalar `s` and any `N`.
//impl<const N : usize,E> ExprRightMultipliable<N,E> for f64 
//    where E:ExprTrait<N> 
//{
//    type Result = ExprMulScalar<N,E>;
//    fn mul_right(self,other : E) -> Self::Result { ExprMulScalar{item : other, lhs : self} }
//}
//
///// Implement `s.mul(Expr<N>)` for scalar `s` and any `N`.
//impl<const N : usize, E> ExprLeftMultipliable<N,E> for f64 
//    where E:ExprTrait<N> 
//{
//    type Result = ExprMulScalar<N,E>;
//    fn mul(self,other : E) -> Self::Result { ExprMulScalar{item : other, lhs : self} }
//}
//
///// Implement `Expr<2>.mul(M)` for sparse matrix M
//impl<E:ExprTrait<2>> ExprRightMultipliable<2,E> for SparseMatrix {
//    type Result = ExprMulRightSparse<E>;
//    fn mul_right(self,other : E) -> Self::Result { ExprMulRightSparse{expr : other, data : self} }
//}
//
///// Implement `M.mul(Expr<2>)` for sparse matrix M
//impl<E:ExprTrait<2>> ExprLeftMultipliable<2,E> for SparseMatrix {
//    type Result = ExprMulLeftSparse<E>;
//    fn mul(self,other : E) -> Self::Result { ExprMulLeftSparse{expr : other, data : self} }
//}
//
///// Implement `Expr<1>.mul(M)` for sparse matrix M
//impl<E:ExprTrait<1>> ExprRightMultipliable<1,E> for SparseMatrix {
//    type Result = ExprReshapeOneRow<2,1,ExprMulRightSparse<ExprReshapeOneRow<1,2,E>>>;
//    fn mul_right(self,other : E) -> Self::Result { 
//        ExprReshapeOneRow{ 
//            dim : 0, 
//            item: ExprMulRightSparse{
//                expr : ExprReshapeOneRow { 
//                    item: other, 
//                    dim: 0 
//                }, 
//                data : self} 
//        } 
//    }
//}
//
///// Implement `M.mul(Expr<1>)` for sparse matrix M
//impl<E:ExprTrait<1>> ExprLeftMultipliable<1,E> for SparseMatrix {
//    type Result = ExprReshapeOneRow<2,1,ExprMulLeftSparse<ExprReshapeOneRow<1,2,E>>>;
//    fn mul(self,other : E) -> Self::Result {
//        ExprReshapeOneRow{
//            dim : 0,
//            item : ExprMulLeftSparse{
//                expr : ExprReshapeOneRow{
//                    dim : 1,
//                    item : other
//                }, 
//                data : self
//            }
//        }
//    }
//}
//
///// Implement `Expr<2>.mul(M)` for dense matrix M
//impl<E:ExprTrait<2>> ExprRightMultipliable<2,E> for DenseMatrix {
//    type Result = ExprMulRightDense<E>;
//    fn mul_right(self,other : E) -> Self::Result { ExprMulRightDense{item : other, rhs : self} }
//}
//
///// Implement `M.mul(Expr<2>)` for dense matrix M
//impl<E:ExprTrait<2>> ExprLeftMultipliable<2,E> for DenseMatrix {
//    type Result = ExprMulLeftDense<E>;
//    fn mul(self,other : E) -> Self::Result { ExprMulLeftDense{item : other, lhs : self} }
//}
//
///// Implement `Expr<1>.mul(M)` for dense matrix M
//impl<E:ExprTrait<1>> ExprRightMultipliable<1,E> for DenseMatrix {
//    type Result = ExprReshapeOneRow<2,1,ExprMulRightDense<ExprReshapeOneRow<1,2,E>>>;
//    fn mul_right(self,other : E) -> Self::Result { 
//        ExprReshapeOneRow{ 
//            dim : 0, 
//            item: ExprMulRightDense{
//                item : ExprReshapeOneRow { 
//                    item: other, 
//                    dim: 0 
//                }, 
//                rhs : self} 
//        } 
//    }
//}
//
///// Implement `M.mul(Expr<1>)` for dense matrix M
//impl<E:ExprTrait<1>> ExprLeftMultipliable<1,E> for DenseMatrix {
//    type Result = ExprReshapeOneRow<2,1,ExprMulLeftDense<ExprReshapeOneRow<1,2,E>>>;
//    fn mul(self,other : E) -> Self::Result {
//        ExprReshapeOneRow{
//            dim : 0,
//            item : ExprMulLeftDense{
//                item : ExprReshapeOneRow{
//                    dim : 1,
//                    item : other
//                }, 
//                lhs : self
//            }
//        }
//    }
//}
//
//
//impl<E:ExprTrait<2>> ExprRightElmMultipliable<2,E> for DenseMatrix {
//    type Result = ExprMulElmDense<2,E>;
//
//    fn mul_elm(self, expr : E) -> Self::Result { ExprMulElmDense{ expr, datashape : self.shape(), data : self.data().to_vec() }}
//}
//
//impl<E:ExprTrait<2>> ExprRightElmMultipliable<2,E> for SparseMatrix {
//    type Result = ExprMulElmSparse<2,E>;
//
//    fn mul_elm(self, expr : E) -> Self::Result { ExprMulElmSparse{ expr, datashape : self.shape(), datasparsity : self.sparsity().to_vec(), data : self.data().to_vec() }}
//}
//
//impl<const N : usize,E> ExprTrait<N> for ExprMulElmDense<N,E> where E : ExprTrait<N> {
//    fn eval(&self,_rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
//        panic!("TODO")
//    }
//}
//
//impl<const N : usize,E> ExprTrait<N> for ExprMulElmSparse<N,E> where E : ExprTrait<N> {
//    fn eval(&self,_rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
//        panic!("TODO")
//    }
//}
