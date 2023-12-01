use super::{ExprTrait};
use super::workstack::WorkStack;


//pub trait ExprAddable<const N : usize, E> where E : ExprTrait<N> {
//    type Result    : ExprTrait<N>;
//    fn add_internal(self, other : E) -> Self::Result;
//    fn sub_internal(self, other : E) -> Self::Result;
//}
//
//impl<const N : usize,E> ExprAddable<N,E> for E where E : ExprTrait<N> {
//    type Result    = ExprAdd<N,E,Self>;
//    fn add_internal(self, other : E) -> Self::Result {
//        ExprAdd{ lhs : other, rhs : self, lcof : 1.0, rcof : 1.0 } 
//    }
//    fn sub_internal(self, other : E) -> Self::Result {
//        ExprAdd{ lhs : other, rhs : other, lcof : 1.0, rcof : -1.0 }
//    }
//}
//
//impl<E> ExprAddable<2,E> for DenseMatrix where E : ExprTrait<2> {
//    type Result = ExprAdd<2,E,Expr<2>>;
//
//    fn add_internal(self, other : E) -> Self::Result {
//        let (shape,data,sp) = self.extract();
//        let nelm = data.len();
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&shape,
//                                 sp,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 data),
//                 lcof : 1.0,
//                 rcof : 1.0,
//        }
//    }
//    fn sub_internal(self, other : E) -> Self::Result {
//        let (shape,mut data,sp) = self.extract();
//        let nelm = data.len();
//        data.iter_mut().for_each(|v| *v *= -1.0 );
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&shape,
//                                 sp,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 data),
//                 lcof : 1.0,
//                 rcof : -1.0,
//        }
//    }
//}
//
//impl<E> ExprAddable<2,E> for SparseMatrix where E : ExprTrait<2> {
//    type Result = ExprAdd<2,E,Expr<2>>;
//
//    fn add_internal(self, other : E) -> Self::Result {
//        let (shape,data,sp) = self.extract();
//        let nelm = data.len();
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&shape,
//                                 sp,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 data),
//                 lcof : 1.0,
//                 rcof : 1.0,
//        }
//    }
//    fn sub_internal(self, other : E) -> Self::Result {
//        let (shape,mut data,sp) = self.extract();
//        let nelm = data.len();
//        data.iter_mut().for_each(|v| *v *= -1.0 );
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&shape,
//                                 sp,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 data),
//                 lcof : 1.0,
//                 rcof : -1.0,
//        }
//    }
//}
//
//impl<E> ExprAddable<1,E> for Vec<f64> where E : ExprTrait<1> {
//    type Result = ExprAdd<1,E,Expr<1>>;
//
//    fn add_internal(self, other : E) -> Self::Result {
//        let nelm = self.len();
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&[nelm],
//                                 None,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 self),
//                 lcof : 1.0,
//                 rcof : -1.0,
//        }
//    }
//    fn sub_internal(mut self, other : E) -> Self::Result {
//        let nelm = self.len();
//        self.iter_mut().for_each(|v| *v *= -1.0 );
//
//        ExprAdd{ lhs : other, 
//                 rhs : Expr::new(&[nelm],
//                                 None,
//                                 (0..nelm+1).collect(),
//                                 vec![0; nelm],
//                                 self),
//                 lcof : 1.0,
//                 rcof : -1.0,
//        }
//    }
//}



pub trait ExprAddRecTrait {
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize;
}

pub struct ExprAdd<const N : usize, L:ExprTrait<N>,R:ExprTrait<N>> {
    lhs : L,
    rhs : R,
    lcof : f64,
    rcof : f64
}

pub struct ExprAddRec<const N : usize,L,R>
    where
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    lhs : L,
    rhs : R,
    lcof : f64,
    rcof : f64
}

impl<const N : usize, L,R> ExprAdd<N,L,R> 
    where 
        L : ExprTrait<N>,
        R : ExprTrait<N>
{
    pub fn new(lhs : L, rhs : R, lcof : f64, rcof : f64) -> ExprAdd<N,L,R> {
        ExprAdd{ lhs,rhs,lcof,rcof}
    }
    /// This will by default override `add()` defined for `ExprTraint<N>`, allowing us to 
    /// create a recursive add structure in stead of just nested adds.
    pub fn add<T>(self,rhs : T) -> ExprAddRec<N,ExprAdd<N,L,R>,T::Result> 
        where T:super::IntoExpr<N>
    {
        ExprAddRec{lhs: self, rhs : rhs.into_expr(), lcof : 1.0, rcof : 1.0 }
    }
    /// This will by default override `add()` defined for `ExprTraint<N>`, allowing us to 
    /// create a recursive add structure in stead of just nested adds.
    pub fn sub<T>(self,rhs : T) -> ExprAddRec<N,ExprAdd<N,L,R>,T::Result> 
        where T:super::IntoExpr<N>
    {
        ExprAddRec{lhs: self, rhs : rhs.into_expr(), lcof : 1.0, rcof : -1.0 }
    }
}

impl<const N : usize,L,R> ExprAddRec<N,L,R> 
    where 
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    pub fn add<T:ExprTrait<N>>(self,rhs : T) -> ExprAddRec<N,Self,T> where Self : Sized { ExprAddRec{lhs: self, rhs, lcof : 1.0, rcof : 1.0} }
}

// Trait implementations
impl<const N : usize,L,R> ExprTrait<N> for ExprAdd<N,L,R> 
    where
        L : ExprTrait<N>,
        R : ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.lhs.eval(ws,rs,xs); ws.inplace_mul(self.lcof);
        self.rhs.eval(ws,rs,xs); ws.inplace_mul(self.rcof);

        super::eval::add(2,rs,ws,xs);
    }
}

impl<const N : usize,L,R> ExprAddRecTrait for ExprAdd<N,L,R> 
    where 
        L : ExprTrait<N>,
        R : ExprTrait<N>
{
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.rhs.eval(rs,ws,xs); rs.inplace_mul(self.rcof * c);
        self.lhs.eval(rs,ws,xs); rs.inplace_mul(self.lcof * c);
        2
    }
}

impl<const N : usize,L,R> ExprAddRecTrait for ExprAddRec<N,L,R> 
    where 
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.rhs.eval(rs,ws,xs); rs.inplace_mul(self.rcof * c);
        1+self.lhs.eval_rec(self.lcof*c,rs,ws,xs)
    }
}

impl<const N : usize,L,R> ExprTrait<N> for ExprAddRec<N,L,R> 
    where
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(1.0,ws,rs,xs);

        super::eval::add(n,rs,ws,xs);
    }
}
