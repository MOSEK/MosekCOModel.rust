use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use super::{ExprReshape, ExprSlice2, ExprTrait};

pub trait ModelExprIndexElement { 
    fn expand(self) -> Range<Option<usize>>; 
}
pub trait ModelExprIndex<T> {
    type Output;
    fn index(self,expr : T) -> Self::Output;
}
impl ModelExprIndexElement for usize            { fn expand(self) -> Range<Option<usize>> { Range{ start: Some(self),       end: Some(self+1)   } } }
impl ModelExprIndexElement for Range<usize>     { fn expand(self) -> Range<Option<usize>> { Range{ start: Some(self.start), end: Some(self.end) } } }
impl ModelExprIndexElement for RangeFrom<usize> { fn expand(self) -> Range<Option<usize>> { Range{ start: Some(self.start), end: None           } } }
impl ModelExprIndexElement for RangeTo<usize>   { fn expand(self) -> Range<Option<usize>> { Range{ start: None,             end: Some(self.end) } } }
impl ModelExprIndexElement for RangeFull        { fn expand(self) -> Range<Option<usize>> { Range{ start: None,             end: None           } } }

impl<const N : usize,E> ModelExprIndex<E> for [Range<Option<usize>>;N] where E : ExprTrait<N>+Sized {
    type Output = ExprSlice2<N,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : self
        }
    }
}

impl<E> ModelExprIndex<E> for usize where E : ExprTrait<1>+Sized { 
    type Output = ExprReshape<1,0,ExprSlice2<1,E>>;
    fn index(self,expr : E) -> Self::Output { 
        ExprReshape{ 
            shape : [],
            item : ExprSlice2{
                expr,
                ranges : [Some(self)..Some(self+1)]
            }
        }
    }
}

impl<E> ModelExprIndex<E> for Range<usize> where E : ExprTrait<1>+Sized { 
    type Output = ExprSlice2<1,E>;
    fn index(self,expr : E) -> Self::Output { 
        ExprSlice2{
            expr,
            ranges : [Some(self.start)..Some(self.end)]
        }
    }
}

impl<E> ModelExprIndex<E> for RangeFrom<usize> where E : ExprTrait<1>+Sized { 
    type Output = ExprSlice2<1,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [Some(self.start)..None]
        }
    }
}

impl<E> ModelExprIndex<E> for RangeTo<usize> where E : ExprTrait<1>+Sized { 
    type Output = ExprSlice2<1,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [None..Some(self.end)]
        }
    }
}

impl<E> ModelExprIndex<E> for RangeFull where E : ExprTrait<1>+Sized { 
    type Output = ExprSlice2<1,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [None..None]
        }
    }
}



impl<const N : usize, E> ModelExprIndex<E> for [Range<usize>; N] 
    where 
        E : ExprTrait<N>+Sized
{
    type Output = ExprSlice2<N,E>;
    fn index(self, expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : self.map(|i| Some(i.start)..Some(i.end))
        }
    }
}

impl<const N : usize, E> ModelExprIndex<E> for [usize; N] 
    where 
        E : ExprTrait<N>+Sized
{
    type Output = ExprReshape<N,0,ExprSlice2<N,E>>;
    fn index(self, expr : E) -> Self::Output {
        ExprReshape{
            shape : [],
            item : ExprSlice2{
                expr,
                ranges : self.map(|i| Some(i)..Some(i+1))
            }
        }
    }
}

impl<E,I1,I2> ModelExprIndex<E> for (I1,I2)
    where E : ExprTrait<2>,
          I1 : ModelExprIndexElement,
          I2 :ModelExprIndexElement
{
    type Output = ExprSlice2<2,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [ self.0.expand(), self.1.expand() ]
        }
    }
}

impl<E,I1,I2,I3> ModelExprIndex<E> for (I1,I2,I3)
    where E : ExprTrait<3>,
          I1 : ModelExprIndexElement,
          I2 :ModelExprIndexElement,
          I3 :ModelExprIndexElement
{
    type Output = ExprSlice2<3,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [ self.0.expand(), self.1.expand(), self.2.expand() ]
        }
    }
}

impl<E,I1,I2,I3,I4> ModelExprIndex<E> for (I1,I2,I3,I4)
    where E : ExprTrait<4>,
          I1 : ModelExprIndexElement,
          I2 :ModelExprIndexElement,
          I3 :ModelExprIndexElement,
          I4 :ModelExprIndexElement
{
    type Output = ExprSlice2<4,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [ self.0.expand(), self.1.expand(), self.2.expand(), self.3.expand() ]
        }
    }
}

impl<E,I1,I2,I3,I4,I5> ModelExprIndex<E> for (I1,I2,I3,I4,I5)
    where E : ExprTrait<5>,
          I1 : ModelExprIndexElement,
          I2 :ModelExprIndexElement,
          I3 :ModelExprIndexElement,
          I4 :ModelExprIndexElement,
          I5 :ModelExprIndexElement
{
    type Output = ExprSlice2<5,E>;
    fn index(self,expr : E) -> Self::Output {
        ExprSlice2{
            expr,
            ranges : [ self.0.expand(), self.1.expand(), self.2.expand(), self.3.expand(), self.4.expand() ]
        }
    }
}
