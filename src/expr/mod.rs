extern crate itertools;

mod eval;
pub mod workstack;

use itertools::{iproduct,izip};
use crate::matrix::SparseMatrix;

use super::utils::*;
use workstack::WorkStack;
use super::matrix;

pub trait ExprTrait {
    /// Evaluate the expression and put the result on the [rs] stack,
    /// using the [ws] to evaluate sub-expressions and [xs] for
    /// general storage.
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack);
    /// Evaluate the expression, then clean it up and put
    /// it on the [rs] stack. The result will guarantee that
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
    fn mul_left_dense(self, v : matrix::DenseMatrix) -> ExprMulLeftDense<Self> where Self:Sized { ExprMulLeftDense{item:self,lhs:v} }
    fn mul_right_dense(self, v : matrix::DenseMatrix) -> ExprMulRightDense<Self> where Self:Sized  { ExprMulRightDense{item:self,rhs:v} }
    // fn transpose(self) -> ExprPermuteAxes<Self>
    fn axispermute(self,perm : Vec<usize>) -> ExprPermuteAxes<Self> where Self:Sized { ExprPermuteAxes{item : self, perm } }
    // fn slice(self, range : &[(Range<usize>)])
    
    fn tril(self,with_diag:bool) -> ExprTriangularPart<Self> where Self:Sized { ExprTriangularPart{item:self,upper:false,with_diag} }
    fn triu(self,with_diag:bool) -> ExprTriangularPart<Self> where Self:Sized { ExprTriangularPart{item:self,upper:true,with_diag} }
    fn trilvec(self,with_diag:bool) -> ExprGatherToVec<ExprTriangularPart<Self>> where Self:Sized { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:false,with_diag} } }
    fn triuvec(self,with_diag:bool) -> ExprGatherToVec<ExprTriangularPart<Self>> where Self:Sized { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:true,with_diag} } }

    /// Sum all elements in an expression yielding a scalar expression.
    fn sum(self) -> ExprSum<Self> where Self:Sized { ExprSum{item:self} }

    fn mul<V>(self,other : V) -> V::Result where V : ExprRightMultipliable<Self>, Self:Sized { other.mul_right(self) }
    fn add<R:ExprTrait>(self,rhs : R) -> ExprAdd<Self,R>  where Self:Sized { ExprAdd{lhs:self,rhs} }
    fn sub<R:ExprTrait>(self,rhs : R) -> ExprAdd<Self,ExprMulScalar<R>>  where Self:Sized { ExprAdd{lhs:self, rhs:ExprMulScalar{item:rhs,lhs:-1.0}} }

    fn dot<V:ExprInnerProductFactorTrait<Self>>(self,v: V) -> V::Output where Self:Sized   { v.dot(self) }

    fn vstack<E:ExprTrait>(self,other : E) -> ExprStack<Self,E>  where Self:Sized { ExprStack::new(self,other,0) }
    fn hstack<E:ExprTrait>(self,other : E) -> ExprStack<Self,E>  where Self:Sized { ExprStack::new(self,other,1) }
    fn stack<E:ExprTrait>(self,dim : usize, other : E) -> ExprStack<Self,E> where Self:Sized { ExprStack::new(self,other,dim) }

    /// Reshape the experssion. The new shape must match the old
    /// shape, meaning that the product of the dimensions are the
    /// same.
    fn reshape(self,shape : Vec<usize>) -> ExprReshape<Self>  where Self:Sized { ExprReshape{item:self,shape} }

    /// Creates a sparse expression with the given shape and sparsity
    /// from the elements in the expression. The sparsity [sp] must
    /// match the actual number of elements in the expression.
    fn scatter(self,shape : Vec<usize>, sp : Vec<usize>) -> ExprScatter<Self>  where Self:Sized { ExprScatter::new(self, shape, sp) }

    /// Reshape a sparse expression into a dense expression with the
    /// given shape. The shape must match the actual number of
    /// elements in the expression.
    fn gather(self,shape : Vec<usize>) -> ExprGather<Self>  where Self:Sized { ExprGather{item:self, shape} }
    fn gather_vec(self) -> ExprGatherToVec<Self>  where Self:Sized { ExprGatherToVec{item:self} }
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

/// The Expr implementation
impl Expr {
    /// Create a new literal expression from data
    ///
    /// Arguments:
    /// * [shape] Shape of the expression. If `sparsity` is `None`,
    ///   the product of the dimensions in the shape must be equal to
    ///   the number of elements in the expression (`ptr.len()-1`)
    /// * [sparsity] If not `None`, this defines the sparsity
    ///   pattern. The pattern denotes the linear indexes if nonzeros in
    ///   the shape. It must be sorted, must contain no duplicates and
    ///   must fit within the `shape`.
    /// * [aptr] The number if elements is `aptr.len()-1`. [aptr] must
    ///   be ascending, so `aptr[i] <= aptr[i+1]`. `aptr` is a vector
    ///   if indexes of the starting points of each element in [asubj]
    ///   and [acof], so element `i` consists of nonzeros defined by
    ///   [asubj[aptr[i]..aptr[i+1]]], acof[aptr[i]..aptr[i+1]]`
    /// * [asubj] Variable subscripts.
    /// * [acof]  Coefficients.
    pub fn new(shape : Vec<usize>,
               sparsity : Option<Vec<usize>>,
               aptr  : Vec<usize>,
               asubj : Vec<usize>,
               acof  : Vec<f64>) -> Expr {
        let fullsize = shape.iter().product();
        if aptr.len() == 0 { panic!("Invalid aptr"); }
        if ! aptr[0..aptr.len()-1].iter().zip(aptr[1..].iter()).all(|(a,b)| a <= b) {
            panic!("Invalid aptr: Not sorted");
        }
        let & sz = aptr.last().unwrap();
        if sz != asubj.len() || sz != acof.len() {
            panic!("Mismatching aptr ({}) and lengths of asubj (= {}) and acof (= {})",sz,asubj.len(),acof.len());
        }

        if let Some(ref sp) = sparsity {
            if sp.len() != aptr.len()-1 {
                panic!("Sparsity pattern length (= {})does not match length of aptr (={})",sp.len(),aptr.len());
            }
            if sp.iter().max().map(|&i| i >= fullsize).unwrap_or(false) {
                panic!("Sparsity pattern out of bounds");
            }

            if ! sp.iter().zip(sp[1..].iter()).all(|(&i0,&i1)| i0 < i1) {
                panic!("Sparsity is not sorted or contains duplicates");
            }
        }
        else {
            if fullsize != aptr.len()-1 {
                panic!("Shape does not match number of elements");
            }
        }

        Expr{
            aptr,
            asubj,
            acof,
            shape,
            sparsity
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

pub struct ExprNil { shape : Vec<usize> }
impl ExprTrait for ExprNil {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,_,_,_) = rs.alloc_expr(self.shape.as_slice(),0,0);
        rptr[0] = 0;
    }
}
pub fn nil(shape : Vec<usize>) -> ExprNil {
    if shape.iter().product::<usize>() != 0 {
        panic!("Shape must have at least one zero-dimension");
    }
    ExprNil{shape}
}


////////////////////////////////////////////////////////////
// Multiply

///// Trait that indicates that the `v:T` implementing it supports
///// v.mul(expr)
//pub trait ExprMultiplyableLeft {
//    type O : ExprTrait;
//    fn mul<E:ExprTrait>(self,rhs : E) -> Self::O;
//}
//
///// Trait that indicates that the type implementing T it supports
///// expr.mul(t.mul(expr)
//pub trait ExprMultiplyableRight {
//    type O : ExprTrait;
//    fn mul<E:ExprTrait>(self,lhs : E) -> Self::O;
//}
//

/// Trait defining something that can be right-multiplied on an
/// expression
pub trait ExprRightMultipliable<E:ExprTrait> {
    type Result : ExprTrait;
    fn mul_right(self,other : E) -> Self::Result;
}

/// Trait defining something that can be left-multiplied on an
/// expression
pub trait ExprLeftMultipliable<E:ExprTrait> {
    type Result : ExprTrait;
    fn mul(self,other : E) -> Self::Result;
}

impl<E:ExprTrait> ExprRightMultipliable<E> for f64 {
    type Result = ExprMulScalar<E>;
    fn mul_right(self,other : E) -> Self::Result { ExprMulScalar{item : other, lhs : self} }
}
impl<E:ExprTrait> ExprLeftMultipliable<E> for f64 {
    type Result = ExprMulScalar<E>;
    fn mul(self,other : E) -> Self::Result { ExprMulScalar{item : other, lhs : self} }
}




pub struct ExprMulLeftDense<E:ExprTrait> {
    item : E,
    lhs  : matrix::DenseMatrix
}
pub struct ExprMulRightDense<E:ExprTrait> {
    item : E,
    rhs  : matrix::DenseMatrix
}
pub struct ExprMulScalar<E:ExprTrait> {
    item : E,
    lhs  : f64
}

pub trait ExprInnerProductFactorTrait<E:ExprTrait> {
    type Output;
    fn dot(self, expr : E) -> Self::Output;
}

impl<E:ExprTrait> ExprInnerProductFactorTrait<E> for &[f64] {
    type Output = ExprDotVec<E>;
    fn dot(self, expr : E) -> Self::Output {
        ExprDotVec{ expr, data:self.to_vec() }
    }
}

pub struct ExprDotVec<E:ExprTrait> {
    data : Vec<f64>,
    expr : E
}

impl<E:ExprTrait> ExprMulLeftDense<E> {
    pub fn new(item : E, lhs : matrix::DenseMatrix ) -> ExprMulLeftDense<E> {
        ExprMulLeftDense{item,lhs}
    }
}

impl<E:ExprTrait> ExprTrait for ExprMulLeftDense<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::mul_left_dense(self.lhs.data(),self.lhs.height(), self.lhs.width(),rs,ws,xs);
    }
}

impl<E:ExprTrait> ExprTrait for ExprMulRightDense<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::mul_right_dense(self.rhs.data(), self.rhs.height(),self.rhs.width(),rs,ws,xs);
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

impl<E:ExprTrait> ExprTrait for ExprDotVec<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::dot_vec(self.data.as_slice(),rs,ws,xs);
    }
}

////////////////////////////////////////////////////////////
//
pub struct ExprMulLeftSparse<E:ExprTrait> {
    data : matrix::SparseMatrix,
    expr : E
}
pub struct ExprMulRightSparse<E:ExprTrait> {
    data : matrix::SparseMatrix,
    expr : E
}

impl<E:ExprTrait> ExprTrait for ExprMulLeftSparse<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::mul_left_sparse(self.data.height(),
                              self.data.width(),
                              self.data.sparsity(),
                              self.data.data(),
                              rs,ws,xs);
    }
}

impl<E:ExprTrait> ExprTrait for ExprMulRightSparse<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::mul_right_sparse(self.data.height(),
                               self.data.width(),
                               self.data.sparsity(),
                               self.data.data(),
                               rs,ws,xs);
    }
}

impl<E:ExprTrait> ExprLeftMultipliable<E> for SparseMatrix {
    type Result = ExprMulLeftSparse<E>;
    fn mul(self,other : E) -> Self::Result { ExprMulLeftSparse{expr : other, data : self} }
}

impl<E:ExprTrait> ExprRightMultipliable<E> for SparseMatrix {
    type Result = ExprMulRightSparse<E>;
    fn mul_right(self,other : E) -> Self::Result { ExprMulRightSparse{expr : other, data : self} }
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
    pub fn add<T:ExprTrait>(self,rhs : T) -> ExprAddRec<Self,T> {
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


pub struct ExprReshape<E:ExprTrait> { item : E, shape : Vec<usize> }
impl<E:ExprTrait> ExprTrait for ExprReshape<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        if self.shape.iter().product::<usize>() != shape.iter().product() {
            panic!("Cannot reshape expression into given shape");
        }

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),subj.len(),ptr.len()-1);

        rptr.clone_from_slice(ptr);
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
        if let Some(rsp) = rsp {
            if let Some(sp) = sp {
                rsp.clone_from_slice(sp)
            }
        }
    }
}

pub struct ExprScatter<E:ExprTrait> { item : E, shape : Vec<usize>, sparsity : Vec<usize> }

impl<E:ExprTrait> ExprScatter<E> {
    pub fn new(item     : E,
               shape    : Vec<usize>,
               sparsity : Vec<usize>) -> ExprScatter<E> {

        if sparsity.iter().max().map(|&v| v >= shape.iter().product()).unwrap_or(false) {
            panic!("Sparsity pattern element out of bounds");
        }

        if sparsity.iter().zip(sparsity[1..].iter()).any(|(&i0,&i1)| i1 <= i0) {
            let mut perm : Vec<usize> = (0..sparsity.len()).collect();
            perm.sort_by_key(|&p| unsafe{ *sparsity.get_unchecked(p)});
            if perm.iter().zip(perm[1..].iter()).any(|(&p0,&p1)| unsafe{ *sparsity.get_unchecked(p0) >= *sparsity.get_unchecked(p1) }) {
                panic!("Sparsity pattern contains duplicates");
            }
            ExprScatter{ item,
                         shape,
                         sparsity : perm.iter().map(|&p| unsafe{ *sparsity.get_unchecked(p)}).collect() }
        }
        else {
            ExprScatter{ item, shape, sparsity }
        }
    }
}

impl<E:ExprTrait> ExprTrait for ExprScatter<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();

        if ptr.len()-1 != self.sparsity.len() {
            panic!("Sparsity pattern does not match number of elements in expression");
        }

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),ptr.len()-1,subj.len());

        rptr.clone_from_slice(ptr);
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);

        if let Some(rsp) = rsp {
            rsp.clone_from_slice(self.sparsity.as_slice())
        }
    }
}

pub struct ExprGather<E:ExprTrait> { item : E, shape : Vec<usize> }
impl<E:ExprTrait> ExprTrait for ExprGather<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();

        if ptr.len()-1 != self.shape.iter().product() {
            panic!("Shape does not match number of elements in expression");
        }

        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),ptr.len()-1,subj.len());

        rptr.clone_from_slice(ptr);
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
    }
}

pub struct ExprGatherToVec<E:ExprTrait> { item : E }
impl<E:ExprTrait> ExprTrait for ExprGatherToVec<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();

        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[ptr.last().copied().unwrap()],ptr.len()-1,subj.len());

        rptr.clone_from_slice(ptr);
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
    }
}


impl ExprTrait for f64 {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],1,1);
        rptr[0] = 0;
        rptr[1] = 1;
        rsubj[0] = 0;
        rcof[0] = *self;
    }
}

impl ExprTrait for Vec<f64> {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[self.len()],self.len(),self.len());
        rptr.iter_mut().enumerate().for_each(|(i,rp)| *rp = i);
        rsubj.fill(0);
        rcof.clone_from_slice(self.as_slice())
    }
}

////////////////////////////////////////////////////////////
//
// Stacking
//
// Recursive evaluation of recursive stacking
//

/// Stack a list of expressions in dimension 1
#[macro_export]
macro_rules! hstack {
    [ $x0:expr ] => { $x0 };
    [ $x0:expr , $( $x:expr ),* ] => {
        {
            $x0 $( .hstack( $x ) )*
        }
    }
}

/// Stack a list of expressions in dimension 0
#[macro_export]
 macro_rules! vstack {
    [ $x0:expr ] => { $x0 };
    [ $x0:expr , $( $x:expr ),* ] => {
        {
            $x0 $( .vstack( $x ))*
        }
    }
}

/// Stack a list of expressions in a given dimension
#[macro_export]
macro_rules! stack {
    [ $n:expr ; $x0:expr ] => { $x0 };
    [ $n:expr ; $x0:expr , $( $x:expr ),* ] => {
        {
            let n = $n;
            $x0 $( .stack( n , $x ))*
        }
    }
}


pub struct ExprStack<E1:ExprTrait,E2:ExprTrait> {
    item1 : E1,
    item2 : E2,
    dim   : usize
}

pub struct ExprStackRec<E1:ExprStackRecTrait,E2:ExprTrait> {
    item1 : E1,
    item2 : E2,
    dim   : usize
}

pub trait ExprStackRecTrait : ExprTrait {
    fn stack_dim(&self) -> usize;
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize;
}

impl<E1:ExprTrait,E2:ExprTrait> ExprStack<E1,E2> {
    pub fn new(item1 : E1, item2 : E2, dim : usize) -> Self { ExprStack{item1,item2,dim} }
    pub fn stack<T:ExprTrait>(self, dim : usize, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim} }
    pub fn vstack<T:ExprTrait>(self, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim:0} }
    pub fn hstack<T:ExprTrait>(self, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim:1} }
}

impl<E1:ExprStackRecTrait,E2:ExprTrait> ExprStackRec<E1,E2> {
    pub fn stack<T:ExprTrait>(self, dim : usize, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim} }
    pub fn vstack<T:ExprTrait>(self, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim:0} }
    pub fn hstack<T:ExprTrait>(self, other : T) -> ExprStackRec<Self,T> { ExprStackRec{item1:self,item2:other,dim:1} }
}

impl<E1:ExprTrait,E2:ExprTrait> ExprTrait for ExprStack<E1,E2> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(ws,rs,xs);
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
impl<E1:ExprTrait,E2:ExprTrait> ExprStackRecTrait for ExprStack<E1,E2> {
    fn stack_dim(&self) -> usize { self.dim }
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.item2.eval(rs,ws,xs);
        self.item1.eval(rs,ws,xs);
        2
    }
}

impl<E1:ExprStackRecTrait,E2:ExprTrait> ExprTrait for ExprStackRec<E1,E2> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(ws,rs,xs);
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
impl<E1:ExprStackRecTrait,E2:ExprTrait> ExprStackRecTrait for ExprStackRec<E1,E2> {
    fn stack_dim(&self) -> usize { self.dim }
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        // we can only do recursive stacking if everything is stacked
        // in the same dimension. If we encounter subexpression that
        // is stacked in a different dimensionm, we simply evaluate it
        // as a normal expression and end the recursion
        self.item2.eval(rs,ws,xs);
        if self.dim == self.item1.stack_dim() {
            1+self.item1.eval_rec(rs,ws,xs)
        }
        else {
            self.item1.eval(rs,ws,xs);
            2
        }
    }
}

/// Dynamic stacking. To stack a list of heterogenous expressions we
/// need to create a list of dynamic ExprTraits

pub struct ExprDynStack {
    exprs : Vec<Box<dyn ExprTrait>>,
    dim : usize
}

impl ExprTrait for ExprDynStack {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.exprs.len();
        for e in self.exprs.iter() {
            e.eval(ws,rs,xs);
        }
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
/// Stack a list of expressions. Since the exact types of the array
/// elements ay differ, we have to get the expressions as a dynamic
/// objects.
///
/// Arguments:
///
/// - dim : Dimension to stack in
/// - exprs : List of expressions
pub fn stack(dim : usize, exprs : Vec<Box<dyn ExprTrait>>) -> ExprDynStack {
    ExprDynStack{exprs,dim}
}
pub fn vstack(exprs : Vec<Box<dyn ExprTrait>>) -> ExprDynStack {
    ExprDynStack{exprs,dim:0}
}
pub fn hstack(exprs : Vec<Box<dyn ExprTrait>>) -> ExprDynStack {
    ExprDynStack{exprs,dim:1}
}


////////////////////////////////////////////////////////////
//


/// Expression that sums all elements in an expression
pub struct ExprSum<T:ExprTrait> {
    item : T
}

impl<T:ExprTrait> ExprTrait for ExprSum<T> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],*ptr.last().unwrap(),1);
        rptr[0] = 0;
        rptr[1] = *ptr.last().unwrap();
        rsubj.clone_from_slice(subj);
        rcof.clone_from_slice(cof);
    }
}
////////////////////////////////////////////////////////////
//

pub struct ExprTriangularPart<T:ExprTrait> {
    item : T,
    upper : bool,
    with_diag : bool
}


fn eval_sparse_pick<F:Fn(usize) -> bool>(pick : F,
                                         d:usize,ptr:&[usize],sp:&[usize],subj:&[usize],cof:&[f64],
                                         rs : & mut WorkStack) {
    let (rnelm,rnnz) : (usize,usize) = 
        izip!(sp.iter(), ptr.iter(),ptr[1..].iter())
            .filter(|(&i,_,_)| pick(i))
            .map(|(_,&p0,&p1)| p1-p0)
            .fold((0,0),|(elmi,nzi),n| (elmi+1,nzi+n));

    let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[d,d],rnnz,rnelm);
    rptr[0] = 0;
    let mut nzi = 0;
    izip!(sp.iter(), ptr.iter(),ptr[1..].iter())
        .filter(|(&i,_,_)| pick(i))
        .zip(rptr[1..].iter_mut())
        .for_each(|((_,&p0,&p1),rp)| {
           rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
           rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
           nzi += p1-p0;
           *rp = p1-p0;
        });
    if let Some(rsp) = rsp {
        izip!(sp.iter()).filter(|&&i| pick(i))
            .zip(rsp.iter_mut())
            .for_each(|(&i,ri)| *ri = i );
    }
    let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
}
impl<T:ExprTrait> ExprTrait for ExprTriangularPart<T> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        let nd = shape.len();

        if nd != 2 || shape[0] != shape[1] {
            panic!("Triangular parts can only be taken from square matrixes");
        }
        let d = shape[0];

        if let Some(sp) = sp {
            match (self.upper,self.with_diag) {
                (true,true)   => eval_sparse_pick(|i| i%d >= i/d,d,ptr,sp,subj,cof,rs),
                (true,false)  => eval_sparse_pick(|i| i%d > i/d,d,ptr,sp,subj,cof,rs),
                (false,true)  => eval_sparse_pick(|i| i%d <= i/d,d,ptr,sp,subj,cof,rs),
                (false,false) => eval_sparse_pick(|i| i%d < i/d,d,ptr,sp,subj,cof,rs),
            }
        }
        else {  
            let rnelm = d * (d+1)/2;
            let rnnz : usize= if self.upper {
                ptr.iter().step_by(d+1).zip(ptr[d..].iter().step_by(d)).map(|(&p0,&p1)| p1-p0).sum::<usize>()
            }
            else {
                ptr.iter().step_by(d).zip(ptr[1..].iter().step_by(d+1)).map(|(&p0,&p1)| p1-p0).sum::<usize>()
            };

            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);
            let mut nzi : usize = 0;
            let mut elmi : usize = 0;
            match (self.upper,self.with_diag) {
                (true,true) => {
                    izip!((0..d*d).step_by(d+1),
                          (d+1..d*d+1).step_by(d))
                        .map(|(i0,i1)| &ptr[i0..i1])
                        .for_each(|ptr| {
                            let n = ptr.last().unwrap() - ptr.first().unwrap();
                            let &ptrb = ptr.first().unwrap();
                            let &ptre = ptr.last().unwrap();
                            rsubj[nzi..nzi+n].clone_from_slice(&subj[ptrb..ptre]);
                            rcof[nzi..nzi+n].clone_from_slice(&cof[ptrb..ptre]);
                            izip!(rptr[elmi+1..elmi+ptr.len()-1].iter_mut(),
                                  ptr.iter(),
                                  ptr[1..].iter()).for_each(|(rp,&p0,&p1)| *rp = p1-p0);
                            nzi += n;
                            elmi += ptr.len();
                        });
                    if let Some(rsp) = rsp {
                        izip!((0..d*d).step_by(d+1),
                              (d+1..d*d+1).step_by(d))
                            .map(|(i0,i1)| i0..i1)
                            .flatten()
                            .zip(rsp.iter_mut())
                            .for_each(|(i,ri)| *ri = i);
                    }
                },
                (true,false) => {
                    izip!((1..d*d).step_by(d+1),
                          (d+1..d*d+1).step_by(d))
                        .map(|(i0,i1)| &ptr[i0..i1])
                        .for_each(|ptr| {
                            let n = ptr.last().unwrap() - ptr.first().unwrap();
                            let &ptrb = ptr.first().unwrap();
                            let &ptre = ptr.last().unwrap();
                            rsubj[nzi..nzi+n].clone_from_slice(&subj[ptrb..ptre]);
                            rcof[nzi..nzi+n].clone_from_slice(&cof[ptrb..ptre]);
                            izip!(rptr[elmi+1..elmi+ptr.len()-1].iter_mut(),
                                  ptr.iter(),
                                  ptr[1..].iter()).for_each(|(rp,&p0,&p1)| *rp = p1-p0);
                            nzi += n;
                            elmi += ptr.len();
                        });
                    if let Some(rsp) = rsp {
                        izip!((1..d*d).step_by(d+1),
                              (d+1..d*d+1).step_by(d))
                            .map(|(i0,i1)| i0..i1)
                            .flatten()
                            .zip(rsp.iter_mut())
                            .for_each(|(i,ri)| *ri = i);
                    }
                },
                (false,true) => {
                    izip!((0..d*d).step_by(d),
                          (1..d*d+1).step_by(d+1))
                        .map(|(i0,i1)| &ptr[i0..i1])
                        .for_each(|ptr| {
                            let n = ptr.last().unwrap() - ptr.first().unwrap();
                            let &ptrb = ptr.first().unwrap();
                            let &ptre = ptr.last().unwrap();
                            rsubj[nzi..nzi+n].clone_from_slice(&subj[ptrb..ptre]);
                            rcof[nzi..nzi+n].clone_from_slice(&cof[ptrb..ptre]);
                            izip!(rptr[elmi+1..elmi+ptr.len()-1].iter_mut(),
                                  ptr.iter(),
                                  ptr[1..].iter()).for_each(|(rp,&p0,&p1)| *rp = p1-p0);
                            nzi += n;
                            elmi += ptr.len();
                        });
                    if let Some(rsp) = rsp {
                        izip!((0..d*d).step_by(d),
                              (1..d*d+1).step_by(d+1))
                            .map(|(i0,i1)| i0..i1)
                            .flatten()
                            .zip(rsp.iter_mut())
                            .for_each(|(i,ri)| *ri = i);
                    }
                },
                (false,false) => {
                    izip!((0..d*d).step_by(d),
                          (0..d*d+1).step_by(d+1))
                        .map(|(i0,i1)| &ptr[i0..i1])
                        .for_each(|ptr| {
                            let n = ptr.last().unwrap() - ptr.first().unwrap();
                            let &ptrb = ptr.first().unwrap();
                            let &ptre = ptr.last().unwrap();
                            rsubj[nzi..nzi+n].clone_from_slice(&subj[ptrb..ptre]);
                            rcof[nzi..nzi+n].clone_from_slice(&cof[ptrb..ptre]);
                            izip!(rptr[elmi+1..elmi+ptr.len()-1].iter_mut(),
                                  ptr.iter(),
                                  ptr[1..].iter()).for_each(|(rp,&p0,&p1)| *rp = p1-p0);
                            nzi += n;
                            elmi += ptr.len();
                        });
                    if let Some(rsp) = rsp {
                        izip!((0..d*d).step_by(d),
                              (0..d*d+1).step_by(d+1))
                            .map(|(i0,i1)| i0..i1)
                            .flatten()
                            .zip(rsp.iter_mut())
                            .for_each(|(i,ri)| *ri = i);
                    }
                }
            };
            rptr[0] = 0;
            let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });

        }
    }
}

struct ExprDiag<E:ExprTrait> {
    item : E,
    anti : bool,
    index : i64
}

impl<E:ExprTrait> ExprTrait for ExprDiag<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        let nd = shape.len();

        if nd != 2 || shape[0] != shape[1] {
            panic!("Diagonals can only be taken from square matrixes");
        }
        let d = shape[0];
        if self.index.abs() as usize >= d {
            panic!("Diagonal index out of bounds");
        }

        let absidx = self.index.abs() as usize;
        if let Some(sp) = sp {
            let (first,num) = match (self.anti,self.index >= 0) {
                (false,true)  => (self.index as usize,       d - absidx),
                (false,false) => (d*(-self.index) as usize,  d - absidx),
                (true,true)   => (d-self.index as usize,     d - absidx),
                (true,false)  => (d*(-self.index) as usize-1,d - absidx)
            };
            let last = num*d;
            // Count elements and nonzeros
            let (rnnz,rnelm) = izip!(sp.iter(),
                                   ptr.iter(),
                                   ptr[1..].iter())
                .filter(|(&i,_,_)| (i < last && 
                                    (((!self.anti) && self.index >= 0 && i%d == i/d + absidx) ||
                                     ((!self.anti) && self.index <  0 && i%d - absidx == i/d) || 
                                     ( self.anti && self.index >= 0 && d-i%d - absidx == i/d) || 
                                     ( self.anti && self.index <  0 && d-i%d + absidx == i/d))))
                .fold((0,0),|(nzi,elmi),(_,&p0,&p1)| (nzi+p1-p0,elmi+1));

            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[d],rnnz,rnelm);

            let mut nzi = 0;
            rptr[0] = 0;
            if let Some(rsp) = rsp {
                izip!(sp.iter(),ptr.iter(),ptr[1..].iter())
                    .filter(|(&i,_,_)| (i < last && 
                                        ((!self.anti && self.index >= 0 && i%d == i/d + absidx) ||
                                         (!self.anti && self.index <  0 && i%d - absidx == i/d) || 
                                         ( self.anti && self.index >= 0 && d-i%d - absidx == i/d) || 
                                         ( self.anti && self.index <  0 && d-i%d + absidx == i/d))))
                    .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
                    .for_each(|((&i,&p0,&p1),(rp,ri))| {
                        *rp = p1-p0;
                        *ri = (i-first)/d;
                        rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                        rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                        nzi += p1-p0;
                    })
            }
            else {
                izip!(sp.iter(),ptr.iter(),ptr[1..].iter())
                    .filter(|(&i,_,_)| (i < last && 
                                        ((!self.anti && self.index >= 0 && i%d == i/d + absidx) ||
                                         (!self.anti && self.index <  0 && i%d - absidx == i/d) || 
                                         ( self.anti && self.index >= 0 && d-i%d - absidx == i/d) || 
                                         ( self.anti && self.index <  0 && d-i%d + absidx == i/d))))
                    .zip(rptr[1..].iter_mut())
                    .for_each(|((&i,&p0,&p1),rp)| {
                        *rp = p1-p0;
                        rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                        rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                        nzi += p1-p0;
                    })
            }   
        } 
        else {
            let (first,num,step) = match (self.anti,self.index >= 0) {
                (false,true)  => (absidx,    d-absidx, d+1),
                (false,false) => (d*absidx,  d-absidx, d+1),
                (true,true)   => (d-absidx,  d-absidx, d-1),
                (true,false)  => (d*absidx-1,d-absidx, d-1)
            };
            
            let rnnz = izip!(0..num,
                             ptr[first..].iter().step_by(step),
                             ptr[first+1..].iter().step_by(step))
                           .map(|(_,&p0,&p1)| p1-p0).sum();
            let rnelm = num;
            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[num],rnnz,rnelm);
            rptr[0] = 0;
            let mut nzi = 0;
            izip!(rptr[1..].iter_mut(),
                 ptr[first..].iter().step_by(step),
                 ptr[first+1..].iter().step_by(step))
                .for_each(|(rp,&p0,&p1)| {
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
                    *rp = p1-p0;
                });
            let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p } );
                        

        }
    }
}


pub struct ExprPermuteAxes<E:ExprTrait> {
    item : E,
    perm : Vec<usize>
}

impl<E:ExprTrait> ExprTrait for ExprPermuteAxes<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::permute_axes(self.perm.as_slice(),rs,ws,xs)
    }
    
}
////////////////////////////////////////////////////////////
//
// Tests

#[cfg(test)]
mod test {
    use super::*;

    fn eq<T:std::cmp::Eq>(a : &[T], b : &[T]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a,b)| *a == *b )
    }

    fn dense_expr() -> Expr {
        super::Expr::new(vec![3,3],
                         None,
                         vec![0,1,2,3,4,5,6,7,8,9],
                         vec![0,1,2,0,1,2,0,1,2],
                         vec![1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3])
    }

    fn sparse_expr() -> Expr {
        super::Expr::new(vec![3,3],
                         Some(vec![0,4,5,6,7]),
                         vec![0,1,2,3,4,5],
                         vec![0,1,2,3,4],
                         vec![1.1,2.2,3.3,4.4,5.5])
    }

    #[test]
    fn mul_left() {
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


    #[test]
    fn mul_right() {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        let m1 = matrix::dense(3,2,vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let m2 = matrix::dense(2,3,vec![1.0,2.0,3.0,4.0,5.0,6.0]);

        let e0 = dense_expr();
        let e1 = sparse_expr();

        let e0_1 = e0.clone().mul(m1.clone());
        let e0_2 = e0.clone().mul(2.0);

        let e1_1 = e1.clone().mul(m1.clone());
        let e1_2 = e1.clone().mul(2.0);

        e0_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e0_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();

        e1_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e1_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    }

    #[test]
    fn add() {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        let m1 = matrix::dense(3,3,vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]);

        let e0 = dense_expr().add(sparse_expr()).add(dense_expr().mul(m1));
        e0.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    }

    #[test]
    fn stack() {
        let e0 = super::Expr::new(vec![3,2,1],
                                  None,
                                  (0..7).collect(),
                                  (0..6).collect(),
                                  (0..6).map(|v| v as f64 * 1.1).collect());
        let e1 = super::Expr::new(vec![3,2,1],
                                  Some(vec![0,2,3,5]),
                                  (0..5).collect(),
                                  vec![6,8,9,11],
                                  (0..4).map(|v| v as f64 * 1.1).collect());
        let s1_0 = e0.clone().stack(0,e0.clone());
        let s1_1 = e0.clone().stack(1,e0.clone());
        let s1_2 = e0.clone().stack(2,e0.clone());
        let s2_0 = e0.clone().stack(0,e1.clone());
        let s2_1 = e0.clone().stack(1,e1.clone());
        let s2_2 = e0.clone().stack(2,e1.clone());

        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        s1_0.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        assert!(eq(shape,&[6,2,1]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]));
        assert!(eq(subj,&[0,1,2,3,4,5,0,1,2,3,4,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        s1_1.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        assert!(eq(shape,&[3,4,1]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]));
        assert!(eq(subj,&[0,1,0,1,2,3,2,3,4,5,4,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        s1_2.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        assert!(eq(shape,&[3,2,2]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]));
        assert!(eq(subj,&[0,0,1,1,2,2,3,3,4,4,5,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());


        s2_0.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        assert!(eq(shape,&[6,2,1]));
        assert!(eq(sp.unwrap(),&[0,1,2,3,4,5,6,8,9,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10]));
        assert!(eq(subj,&[0,1,2,3,4,5,6,8,9,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        s2_1.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        assert!(eq(shape,&[3,4,1]));
        assert!(eq(sp.unwrap(),&[0,1,2,4,5,6,7,8,9,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10]));
        assert!(eq(subj,&[0,1,6,2,3,8,9,4,5,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        s2_2.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[3,2,2]));
        assert!(eq(sp.unwrap(),&[0,1,2,4,5,6,7,8,10,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10]));
        assert!(eq(subj,&[0,6,1,2,8,3,9,4,5,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());


        let s3_0 = e1.clone().stack(0,e1.clone());
        s3_0.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[6,2,1]));
        assert!(eq(sp.unwrap(),&[0,2,3,5,6,8,9,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8]));
        assert!(eq(subj,&[6,8,9,11,6,8,9,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        let s3_1 = e1.clone().stack(1,e1.clone());
        s3_1.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[3,4,1]));
        assert!(eq(sp.unwrap(),&[0,2,4,5,6,7,9,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8]));
        assert!(eq(subj,&[6,6,8,9,8,9,11,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        let s3_2 = e1.clone().stack(2,e1.clone());
        s3_2.eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[3,2,2]));
        assert!(eq(sp.unwrap(),&[0,1,4,5,6,7,10,11]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8]));
        assert!(eq(subj,&[6,6,8,8,9,9,11,11]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        // TEST RECURSIVE EVALUATION
        e0.clone().stack(0,e1.clone()).stack(0,e0.clone()).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[9,2,1]));
        assert!(eq(sp.unwrap(),&[0,1,2,3,4,5,
                                 6,8,9,11,
                                 12,13,14,15,16,17]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]));
        assert!(eq(subj,&[0,1,2,3,4,5,
                          6,8,9,11,
                          0,1,2,3,4,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        e0.clone().stack(1,e1.clone()).stack(1,e0.clone()).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[3,6,1]));
        assert!(eq(sp.unwrap(),&[0,1,2,4,5,
                                 6,7,8,9,10,11,
                                 12,13,15,16,17]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]));
        assert!(eq(subj,&[0,1,6,0,1,2,3,8,9,2,3,4,5,11,4,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());

        e0.clone().stack(2,e1.clone()).stack(2,e0.clone()).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert!(eq(shape,&[3,2,3]));
        assert!(eq(sp.unwrap(),&[0,1,2,3,5,6,7,8,9,10,11,12,14,15,16,17]));
        assert!(eq(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]));
        assert!(eq(subj,&[0,6,0,
                          1,1,
                          2,8,2,
                          3,9,3,
                          4,4,
                          5,11,5]));
        assert!(rs.is_empty());
        assert!(ws.is_empty());
    }

}
