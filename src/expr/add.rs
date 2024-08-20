use super::{ExprEvalError, ExprTrait, IntoExpr};
use super::workstack::WorkStack;


pub trait ExprAddRecTrait {
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<usize,ExprEvalError>;
}

pub struct ExprAdd<const N : usize, L:ExprTrait<N>,R:ExprTrait<N>> {
    lhs  : L,
    rhs  : R,
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
        where 
            Self: Sized,
            T:IntoExpr<N>
    {
        ExprAddRec{lhs: self, rhs:rhs.into(), lcof : 1.0, rcof : 1.0 }
    }
    /// This will by default override `add()` defined for `ExprTraint<N>`, allowing us to 
    /// create a recursive add structure in stead of just nested adds.
    pub fn sub<T>(self,rhs : T) -> ExprAddRec<N,ExprAdd<N,L,R>,T::Result> 
        where 
            Self: Sized,
            T:IntoExpr<N>
    {
        ExprAddRec{lhs: self, rhs:rhs.into(), lcof : 1.0, rcof : -1.0 }
    }
}

impl<const N : usize,L,R> ExprAddRec<N,L,R> 
    where 
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    pub fn add<T:ExprTrait<N>>(self,rhs : T) -> ExprAddRec<N,Self,T::Result> 
        where 
            Self : Sized,
            T : IntoExpr<N>
    { ExprAddRec{lhs: self, rhs : rhs.into(), lcof : 1.0, rcof : 1.0} }
}






// Trait implementations
impl<const N : usize,L,R> ExprTrait<N> for ExprAdd<N,L,R> 
    where
        L : ExprTrait<N>,
        R : ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> {
        self.lhs.eval(ws,rs,xs)?; ws.inplace_mul(self.lcof);
        self.rhs.eval(ws,rs,xs)?; ws.inplace_mul(self.rcof);

        super::eval::add(2,rs,ws,xs)
    }
}





impl<const N : usize,L,R> ExprAddRecTrait for ExprAdd<N,L,R> 
    where 
        L : ExprTrait<N>,
        R : ExprTrait<N>
{
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<usize,ExprEvalError> {
        self.rhs.eval(rs,ws,xs)?; rs.inplace_mul(self.rcof * c);
        self.lhs.eval(rs,ws,xs)?; rs.inplace_mul(self.lcof * c);
        Ok(2)
    }
}

impl<const N : usize,L,R> ExprAddRecTrait for ExprAddRec<N,L,R> 
    where 
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    fn eval_rec(&self, c : f64, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<usize,ExprEvalError> {
        self.rhs.eval(rs,ws,xs)?; rs.inplace_mul(self.rcof * c);
        Ok(1+self.lhs.eval_rec(self.lcof*c,rs,ws,xs)?)
    }
}

impl<const N : usize,L,R> ExprTrait<N> for ExprAddRec<N,L,R> 
    where
        L : ExprAddRecTrait,
        R : ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(), ExprEvalError> {
        let n = self.eval_rec(1.0,ws,rs,xs)?;

        super::eval::add(n,rs,ws,xs)
    }
}


/// Implemented by types that can be added to an expression by converting it into an expression.
pub trait ExprAddable<const N : usize> {
    type Result : ExprTrait<N>;
    fn to_expr(self) -> Self::Result;
}


