//! 
//! This module defines various experimental exprssions whose usefulness has not been determined
//! yet.
//!
//! # Options expressions
//!
//! The [`EitherExpr`], [`EitherExpr3`], [`EitherExpr4`] and [`EitherExpr5`] define expressions
//! that can be one of multiple types. Since expression types are compile-time fixed (unless using
//! dynamic expressions), this is the only way to make conditional expressions.
//!
//! # Generator expressions
//!
//! These are expressions generated from an index set using a function. Generally speaking these
//! are less efficient than using vectorized expressions for most cases, but, for example,
//! when formulating an expression where each element is conditional it may be the most reasonable
//! option.
//!

use expr::ExprEvalError;
use itertools::iproduct;
use crate::utils::iter::*;
use crate::*;


/// An expression that is one of two fixed types, which must have same dimensionality.
///
/// # Example
///
/// ```
/// use mosekcomodel::*;
/// use mosekcomodel::experimental::*;
/// use mosekcomodel::dummy::Model;
/// let mut m = Model::new(None);
/// let x = m.variable(None,unbounded());
/// let y = m.variable(None,unbounded());
/// let b = true;
/// let e = if b { EitherExpr::Left(x.add(y)) } else { EitherExpr::Right(x.mul(5.0)) };
/// ```
pub enum EitherExpr<const N : usize,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    Left(A),
    Right(B)
}

pub fn either_left <const N : usize,A,B>(a : A) -> EitherExpr<N,A::Result,B::Result> 
    where 
        A : IntoExpr<N>, 
        B : IntoExpr<N>,
        A::Result : Sized,
        B::Result : Sized
{
    EitherExpr::Left(a.into()) 
}
pub fn either_right<const N : usize,A,B>(b : B) -> EitherExpr<N,A::Result,B::Result> 
    where 
        A : IntoExpr<N>, 
        B : IntoExpr<N>,
        A::Result : Sized,
        B::Result : Sized
{
    EitherExpr::Right(b.into()) 
}

/// An expression that can be one of three fixed types with same dimensionality.
///
/// Work the same way as [`EitherExpr`].
pub enum EitherExpr3<const N : usize,E1,E2,E3> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N> 
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3)
}

/// An expression that can be one of four fixed types with same dimensionality.
///
/// Work the same way as [`EitherExpr`].
pub enum EitherExpr4<const N : usize,E1,E2,E3,E4> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N> 
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3),
    Opt4(E4)
}

/// An expression that can be one of five fixed types with same dimensionality.
///
/// Work the same way as [`EitherExpr`].
pub enum EitherExpr5<const N : usize,E1,E2,E3,E4,E5> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>,
        E5 : Sized+ExprTrait<N>
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3),
    Opt4(E4),
    Opt5(E5)
}

impl<const N : usize,A,B> ExprTrait<N> for EitherExpr<N,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr::Left(e) => e.eval(rs,ws,xs),
            EitherExpr::Right(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3> ExprTrait<N> for EitherExpr3<N,E1,E2,E3> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N> 
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr3::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr3::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr3::Opt3(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3,E4> ExprTrait<N> for EitherExpr4<N,E1,E2,E3,E4> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr4::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt3(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt4(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3,E4,E5> ExprTrait<N> for EitherExpr5<N,E1,E2,E3,E4,E5> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>,
        E5 : Sized+ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr5::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt3(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt4(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt5(e) => e.eval(rs,ws,xs)
        }
    }
}

/// An N-dimensional expression constructed as the stack of scalar expressions generated by calling
/// a function for each index.
///
/// See [`genexpr`].
pub struct GeneratorExpr<const N : usize,F,R> 
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : IntoExpr<0>,
        Self : Sized
{
    shape : [usize; N],
    sp    : Option<Vec<usize>>,
    f : F
}



/// Generative expression. Generate expression as one scalar expression per index. Note that by
/// construction all generated expressions must have the same type, which means that conditional
/// expressions must be expressed either as dynamic expressions or as enumerated expressions. 
///
/// Generating expressions as a set of scalar expressions is generally less efficient than writing
/// them on vectorized form, but there may be situations where it is simpler, for example if the
/// expression uses complicated index arithmetic.
///
/// # Arguments
/// 
/// - `shape` Shape of the expression.
/// - `sp` Optional sparsity pattern. If given, the function will only be called for indexes in the
///   sparsity.
/// - `f` The generative function. This is a function that generates a scalar expression for each
///   index.
///
/// # Example
/// ```
/// use mosekcomodel::*;
/// use mosekcomodel::experimental::*;
/// use mosekcomodel::dummy::Model;
///
/// let mut m = Model::new(None);
/// let x = m.variable(None, &[5,5]);
/// // Generate expression as E + E'
/// _ = m.constraint(None, 
///                  genexpr([5,5], None, |i| Some(x.index(*i).add(x.index([i[1],i[0]])))),
///                  greater_than(0.0).with_shape(&[5,5]));
/// // Generate expression as E + E', except diagonal elements that are just E 
/// _ = m.constraint(None,
///                  genexpr([5,5], None, 
///                          |i| Some(if i[0] != i[1] {
///                                     EitherExpr::Left(x.index(*i).add(x.index([i[1],i[0]]).into_expr())) 
///                                   }
///                                   else {
///                                     EitherExpr::Right(x.index(*i).into_expr())
///                                   })),
///                  greater_than(0.0).with_shape(&[5,5]));
/// // Generate expression as the lower triangular part if E+E'
/// _ = m.constraint(None,
///                  genexpr([5,5], None,
///                          |i| if i[0] == i[1] {
///                                Some(EitherExpr::Right(x.index(*i).into_expr()))
///                              }
///                              else if i[0] > i[1] {
///                                Some(EitherExpr::Left(x.index(*i).add(x.index([i[1],i[0]])).into_expr()))
///                              } 
///                              else {
///                                None
///                              }),
///                  greater_than(0.0).with_shape(&[5,5]));
/// ```
///
pub fn genexpr<const N : usize,F,R>(shape : [usize; N], sp : Option<Vec<usize>>, f : F) -> GeneratorExpr<N,F,R>
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : IntoExpr<0>
{
    GeneratorExpr{ shape, sp, f }
}


impl<const N : usize,F,R> ExprTrait<N> for GeneratorExpr<N,F,R>
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : IntoExpr<0>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> 
    {
        let maxnelm = 
            if let Some(ref sp) = self.sp {
                sp.len()
            } else {
                self.shape.iter().product()
            };
        let mut spx = Vec::with_capacity(maxnelm);
        //let (spx,_) = xs.alloc(maxnelm,0);dd

        if let Some(sp) = &self.sp {
            for i in sp.iter() {
                let mut ii = [0usize;N];
                _ = ii.iter_mut().zip(self.shape.iter()).rev().fold(*i,|i,(k,&d)| { *k = i%d; i/d });
                if let Some(e) = (self.f)(&ii) {
                    spx.push(*i);

                    e.into().eval(ws,rs,xs)?;
                }
            }
        }
        else {
            for (i,ii) in self.shape.index_iterator().enumerate() {
                if let Some(e) = (self.f)(&ii) {
                    spx.push(i);

                    e.into().eval(ws,rs,xs)?;
                }
            }
        }
        let nelm = spx.len();

        let exprs = ws.pop_exprs(nelm);
        
        let nnz = exprs.iter().map(|(_,_,_,subj,_)| subj.len()).sum::<usize>();
        
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&self.shape, nnz, nelm);

        rptr[0] = 0;
        rptr[1..].iter_mut().zip(exprs.iter().rev()).fold(0,|p,(rp,(_,_,_,subj,_))| { *rp = p + subj.len(); *rp });
        rsubj.iter_mut().zip(exprs.iter().rev().flat_map(|(_,_,_,subj,_)| subj.iter())).for_each(|(rj,&j)| *rj = j);
        rcof.iter_mut().zip(exprs.iter().rev().flat_map(|(_,_,_,_,cof)| cof.iter())).for_each(|(rc,&c)| *rc = c);

        if let Some(rsp) = rsp {
            rsp.copy_from_slice(spx.as_slice());
        }

        Ok(())
    }
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/// Generator expression for dimension `N=1`. 
///
/// The generic arguments are
/// - `F` Type of the function that generates scalar expressions
/// - `E` The type of the result of `F`
/// - `T` The type of the object defining the range. This is anything that can be turned into an
///       iterator with known size. The iterator and the iterators elements must be [`Clone`].
/// - `I` The iterator that `T` can turn into. It must be clonable and have clonable elements.
pub struct ExprGenerator1<F,E,T,I> 
    where 
        T : IntoIterator<IntoIter = I>,
        I : ExactSizeIterator,
        F : Fn(usize, I::Item) -> Option<E>,
        E : IntoExpr<0>
{
    /// The object that defines the "indexes" provided to the generator function. For example
    /// `[1,7,5,9]` (integer list), `["a","b","c']` (string list), `(5..12).step_by(2)` (an
    /// iterator with known finite length).
    idx : T, 
    /// A functio that, given an index from `idx` returns a scalar expression.
    f : F
}

impl<F,E,T,I> ExprTrait<1> for ExprGenerator1<F,E,T,I> where 
        T : IntoIterator<IntoIter = I>+Clone,
        I : ExactSizeIterator,
        F : Fn(usize, I::Item) -> Option<E>,
        E : IntoExpr<0>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> 
    {
        let it = self.idx.clone().into_iter();
        let shape = [ it.len() ];
        let totsize = shape[0];
        let mut spx = Vec::with_capacity(totsize);
        
        for (i,v) in it.enumerate() {
            if let Some(e) = (self.f)(i,v) {
                spx.push(i);
                e.into().eval(ws,rs,xs)?;
            }
        }

        if spx.len() < totsize {
            expr::eval::stack_scalars(&shape, Some(spx.as_slice()), rs, ws, xs)
        }
        else {
            expr::eval::stack_scalars(&shape, None, rs, ws, xs)
        }
    }
}

/// Trait that extends objects with `genexpr` function that creates a [`ExprGenerator1`]
/// expression.
pub trait ExprGenerator1Ex<F,E,T,I>
    where 
        T : IntoIterator<IntoIter = I>+Clone,
        I : ExactSizeIterator+Clone,
        I::Item : Clone,
        E : IntoExpr<0>,
        F : Fn(usize,I::Item) -> Option<E>
{
    /// Create a generator expression using `self` for indexes.
    /// 
    /// # Example
    ///
    /// ```
    /// use mosekcomodel::*;
    /// use mosekcomodel::experimental::*;
    /// use mosekcomodel::dummy::Model;
    ///
    /// let mut m = Model::new(None);
    /// let x = m.variable(None,unbounded().with_shape(&[5]));
    /// let e = (0..5).rev().genexpr(|_,i| Some(x.index(i)));
    /// ```
    fn genexpr(self,f : F) -> ExprGenerator1<F,E,T,I>;
}

// This should allow us to write stuff like `(["alpha","beta","gamma"]).genexpr(|i,n| x.clone().index(j))`
impl<F,E,T,I> ExprGenerator1Ex<F,E,T,I> for T
    where 
        T : IntoIterator<IntoIter = I>+Clone,
        I : ExactSizeIterator+Clone,
        I::Item : Clone,
        E : IntoExpr<0>,
        F : Fn(usize,I::Item) -> Option<E>
{
    fn genexpr(self,f : F) -> ExprGenerator1<F,E,T,I> {
        ExprGenerator1{ 
            idx : self,
            f
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/// Generator expression for dimension `N=2`. 
///
/// The generic arguments are
/// - `F` Type of the function that generates scalar expressions
/// - `E` The type of the result of `F`
/// - `T0`, `T1` The type of the object defining the range. This is anything that can be turned
///   into an iterator with known size. The iterator and the iterators elements must be [`Clone`].
/// - `I0`, `I1` The iterators that `T0` and `T1` can turn into. It must be clonable and have
///   clonable elements.
pub struct ExprGenerator2<F,E,T0,T1,I0,I1> 
    where 
        T0 : IntoIterator<IntoIter = I0>,
        T1 : IntoIterator<IntoIter = I1>,
        I0 : ExactSizeIterator,
        I1 : ExactSizeIterator,
        F : Fn(usize,I0::Item,I1::Item) -> Option<E>,
        E : IntoExpr<0>
{
    idx : (T0,T1), 
    f : F
}

impl<F,E,T0,T1,I0,I1> ExprTrait<2> for ExprGenerator2<F,E,T0,T1,I0,I1> where 
        T0 : IntoIterator<IntoIter = I0>+Clone,
        T1 : IntoIterator<IntoIter = I1>+Clone,
        I0 : ExactSizeIterator+Clone,
        I1 : ExactSizeIterator+Clone,
        I0::Item : Clone,
        I1::Item : Clone,
        F : Fn(usize,I0::Item,I1::Item) -> Option<E>,
        E : IntoExpr<0>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> 
    {
        let it0 = self.idx.0.clone().into_iter();
        let it1 = self.idx.1.clone().into_iter();
        let shape = [ it0.len(), it1.len() ];
        let totsize = shape[0];
        let mut spx = Vec::with_capacity(totsize);
        
        for (i,(v0,v1)) in (0..).zip(iproduct!(it0,it1)) {
            if let Some(e) = (self.f)(i,v0,v1) {
                spx.push(i);
                e.into().eval(ws,rs,xs)?;
            }
        }

        if spx.len() < totsize {
            expr::eval::stack_scalars(&shape, Some(spx.as_slice()), rs, ws, xs)
        }
        else {
            expr::eval::stack_scalars(&shape, None, rs, ws, xs)
        }
    }
}

/// Trait that extends objects with `genexpr` function that creates a [`ExprGenerator2`]
/// expression.
pub trait ExprGenerator2Ex<F,E,T0,T1,I0,I1>
    where 
        T0 : IntoIterator<IntoIter = I0>+Clone,
        T1 : IntoIterator<IntoIter = I1>+Clone,
        I0 : ExactSizeIterator+Clone,
        I1 : ExactSizeIterator+Clone,
        I0::Item : Clone,
        I1::Item : Clone,
        E : IntoExpr<0>,
        F : Fn(usize,I0::Item,I1::Item) -> Option<E>
{
    /// Create a generator expression using `self` for indexes.
    /// 
    /// # Example
    ///
    /// ```
    /// use mosekcomodel::*;
    /// use mosekcomodel::experimental::*;
    /// use mosekcomodel::dummy::Model;
    ///
    /// let mut m = Model::new(None);
    /// let x = m.variable(None,unbounded().with_shape(&[5,5]));
    /// let e = (0..4,0..4).genexpr(|_,i,j| Some(x.index([i..i+2,j..j+2]).sum().mul(0.25)));
    /// ```
    fn genexpr(self,f : F) -> ExprGenerator2<F,E,T0,T1,I0,I1>;
}

impl<F,E,T0,T1,I0,I1> ExprGenerator2Ex<F,E,T0,T1,I0,I1> for (T0,T1)
    where 
        T0 : IntoIterator<IntoIter = I0>+Clone,
        T1 : IntoIterator<IntoIter = I1>+Clone,
        I0 : ExactSizeIterator+Clone,
        I1 : ExactSizeIterator+Clone,
        I0::Item : Clone,
        I1::Item : Clone,
        E : IntoExpr<0>,
        F : Fn(usize,I0::Item,I1::Item) -> Option<E>
{
    fn genexpr(self,f : F) -> ExprGenerator2<F,E,T0,T1,I0,I1> {
        ExprGenerator2{ 
            idx : self,
            f
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////





#[cfg(test)]
mod test {
    use crate::*;
    use std::collections::HashMap;

    type Model = crate::dummy::Model;

    use super::*;
    #[test]
    fn test_gen1() {
        let mut model = Model::new(None);
        let x = model.variable(None, unbounded().with_shape(&[5,5]));
        
        {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            hstack![ x.clone().index([0..1,0..1]),
                     x.clone().index([1..2,1..2]),
                     x.clone().index([2..3,2..3]),
                     x.clone().index([3..4,3..4]),
                     x.clone().index([4..5,4..5]) ].eval(&mut rs,&mut ws,&mut xs).unwrap();
            let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
            assert_eq!(shape, &[1,5]);
            assert!(sp.is_none());
            assert_eq!(ptr,&[0,1,2,3,4,5]);
            assert_eq!(subj,&[0,6,12,18,24]);
        }

        {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            genexpr([5,5], None, |i| {
                if i[0] >= i[1] {
                    Some(x.clone().index(*i))
                }
                else {
                    None
                }
            }).eval(&mut rs, &mut ws, &mut xs).unwrap();
            let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
            assert_eq!(shape, &[5,5]);
            assert!(sp.is_some());
            assert_eq!(sp.unwrap(),&[0, 5,6, 10,11,12, 15,16,17,18, 20,21,22,23,24]);
            assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
            assert_eq!(subj,&[0, 5,6, 10,11,12, 15,16,17,18, 20,21,22,23,24]);
        }
    }

    #[test]
    fn test_gen2() {
        let mut model = Model::new(None);
        let x = model.variable(None, unbounded().with_shape(&[5,5]));
        {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            genexpr([5,5], None, |i| 
                match i[0].cmp(&i[1]) {
                    std::cmp::Ordering::Equal   => Some(EitherExpr::Left(x.index(*i).into_expr())),
                    std::cmp::Ordering::Greater => Some(EitherExpr::Right(x.index(*i).add(x.index([i[1],i[0]])))),
                    std::cmp::Ordering::Less    => Some(EitherExpr::Right(x.index([i[1],i[0]]).add(x.index(*i)))),
                }
            ).eval(&mut rs, &mut ws, &mut xs).unwrap();
            let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
            assert_eq!(shape, &[5,5]);
            assert!(sp.is_none());
            assert_eq!(ptr,&[0,1,3,5,7,9, 11,12,14,16,18, 20,22,23,25,27, 29,31,33,34,36, 38,40,42,44,45 ]);
            assert_eq!(subj,&[ 0,     1,5,    2,10,  3,15,  4,20, 
                               1,5,   6,      7,11,  8,16,  9,21,
                               2,10,  7,11,  12,    13,17, 14,22,
                               3,15,  8,16,  13,17, 18,    19,23,
                               4,20,  9,21,  14,22, 19,23, 24 ]);
            //  1   2   3   4   5 
            //  6   7   8   9  10 
            // 11  12  13  14  15
            // 16  17  18  19  20
            // 21  22  23  24  25
        }
    }
    


    #[test]
    fn test_gen3() {
        let mut model = Model::new(None);
        let x = model.variable(None, unbounded().with_shape(&[15]));
        {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);

            (5..10).genexpr(|_,k : usize| Some(x.index(k).into_expr())).eval(& mut rs, & mut ws, & mut xs).unwrap();
            
            let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
            assert_eq!(shape, &[5]);
            assert!(sp.is_none());
            assert_eq!(ptr,&[0,1,2,3,4,5 ]);
            assert_eq!(subj,&[ 5,6,7,8,9 ])
        }
    }
    
    #[test]
    fn test_gen4() {
        let mut model = Model::new(None);
        let x = model.variable(None, unbounded().with_shape(&[10]));
        {
            let mut rs = WorkStack::new(1024);
            let mut ws = WorkStack::new(1024);
            let mut xs = WorkStack::new(1024);
            
            let names = ["alpha","beta","gamma"].map(String::from);

            let mut nm : HashMap<String,Variable<0>> = HashMap::new();
            for n in names.iter() {
                nm.insert(n.clone(), model.variable(Some(n.as_str()),unbounded()));
            }


            (1..5,names.iter()).genexpr(|_, i, n| Some(x.clone().index(i).add(nm.get(n).unwrap().clone())))
                .eval(& mut rs, & mut ws, & mut xs).unwrap();

            let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
            assert_eq!(shape, &[4,3]);
            assert!(sp.is_none());
            assert_eq!(ptr,&[0,2,4,6,8,10,12,14,16,18,20,22,24 ]);
            assert_eq!(subj,&[ 10,1,11,1,12,1,
                               10,2,11,2,12,2,
                               10,3,11,3,12,3,
                               10,4,11,4,12,4 ]);
        }
    }
}

