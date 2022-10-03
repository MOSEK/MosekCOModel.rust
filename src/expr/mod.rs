extern crate itertools;


mod eval;
pub mod workstack;

use itertools::{iproduct};
use super::utils::*;
use super::Variable;
use workstack::WorkStack;
use super::matrix;

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
    fn mul_left_dense(self, v : matrix::DenseMatrix) -> ExprMulLeftDense<Self> { ExprMulLeftDense{item:self,lhs:v} }
    fn mul_right_dense(self, v : matrix::DenseMatrix) -> ExprMulRightDense<Self> { ExprMulRightDense{item:self,rhs:v} }
    // fn transpose(self) -> ExprPermuteAxes<Self>
    // fn axispermute(self) -> ExprPermuteAxes<Self>
    // fn slice(self, range : &[(Range<usize>)])

    fn sum(self) -> ExprSum<Self> { ExprSum{item:self} }

    fn mul<V>(self,other : V) -> V::Result where V : ExprRightMultipliable<Self> { other.mul_right(self) }
    fn add<R:ExprTrait>(self,rhs : R) -> ExprAdd<Self,R> { ExprAdd{lhs:self,rhs} }

    fn vstack<E:ExprTrait>(self,other : E) -> ExprStack<Self,E> { ExprStack::new(self,other,0) }
    fn hstack<E:ExprTrait>(self,other : E) -> ExprStack<Self,E> { ExprStack::new(self,other,1) }
    fn stack<E:ExprTrait>(self,dim : usize, other : E) -> ExprStack<Self,E> { ExprStack::new(self,other,dim) }

    fn reshape(self,shape : Vec<usize>) -> ExprReshape<Self> { ExprReshape{item:self,shape} }
    fn scatter(self,shape : Vec<usize>, sp : Vec<usize>) -> ExprScatter<Self> { ExprScatter::new(self, shape, sp) }
    fn gather(self,shape : Vec<usize>) -> ExprGather<Self> { ExprGather{item:self, shape} }
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
    /// * `shape` Shape of the expression. If `sparsity` is `None`,
    ///   the product of the dimensions in the shape must be equal to
    ///   the number of elements in the expression (`ptr.len()-1`)
    /// * `sparsity` If not `None`, this defines the sparsity
    ///   pattern. The pattern denotes the linear indexes if nonzeros in
    ///   the shape. It must be sorted, must contain no duplicates and
    ///   must fit within the `shape`.
    /// * `aptr` The number if elements is `aptr.len()-1`. [aptr] must
    ///   be ascending, so `aptr[i] <= aptr[i+1]`. `aptr` is a vector
    ///   if indexes of the starting points of each element in [asubj]
    ///   and [acof], so element `i` consists of nonzeros defined by
    ///   `asubj[aptr[i]..aptr[i+1]], acof[aptr[i]..aptr[i+1]]`
    /// * `asubj` Variable subscripts.
    /// * `acof`  Coefficients.
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
        if sz != asubj.len() || asubj.len() != acof.len() {
            panic!("Mismatching aptr, asubj and acof");
        }

        if let Some(ref sp) = sparsity {
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


    // pub fn into_diag(self) -> Expr {
    //     if self.shape.len() != 1 {
    //         panic!("Diagonals can only be made from vector expressions");
    //     }

    //     let d = self.shape[0];
    //     Expr{
    //         aptr : self.aptr,
    //         asubj : self.asubj,
    //         acof : self.acof,
    //         shape : vec![d,d],
    //         sparsity : Some((0..d*d).step_by(d+1).collect())
    //     }
    // }

    // pub fn reshape(self,shape:&[usize]) -> Expr {
    //     if self.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
    //         panic!("Invalid shape for this expression");
    //     }

    //     Expr{
    //         aptr : self.aptr,
    //         asubj : self.asubj,
    //         acof : self.acof,
    //         shape : shape.to_vec(),
    //         sparsity : self.sparsity
    //     }
    // }
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


impl ExprTrait for Variable {
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
    fn mul<E:ExprTrait>(self,rhs : E) -> Self::O;
}

/// Trait that indicates that the type implementing T it supports
/// expr.mul(t.mul(expr)
pub trait ExprMultiplyableRight {
    type O : ExprTrait;
    fn mul<E:ExprTrait>(self,lhs : E) -> Self::O;
}


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


pub struct ExprReshape<E:ExprTrait> { item : E, shape : Vec<usize> }
impl<E:ExprTrait> ExprTrait for ExprReshape<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        if self.shape.iter().product::<usize>() != shape.iter().product() {
            panic!("Cannot reshape expression into given shape");
        }

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),ptr.len()-1,subj.len());

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

////////////////////////////////////////////////////////////
//
// Stacking
//
// Recursive evaluation of recursive stacking
//

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
        let n = self.eval_rec(rs,ws,xs);
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
impl<E1:ExprTrait,E2:ExprTrait> ExprStackRecTrait for ExprStack<E1,E2> {
    fn stack_dim(&self) -> usize { self.dim }
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.item2.eval(ws,rs,xs);
        self.item1.eval(ws,rs,xs);
        2
    }
}

impl<E1:ExprStackRecTrait,E2:ExprTrait> ExprTrait for ExprStackRec<E1,E2> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(rs,ws,xs);
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
        self.item2.eval(ws,rs,xs);
        if self.dim == self.item1.stack_dim() {
            1+self.item1.eval_rec(ws,rs,xs)
        }
        else {
            self.item1.eval(ws,rs,xs);
            2
        }
    }
}


////////////////////////////////////////////////////////////
//
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
// Tests

#[cfg(test)]
mod test {
    use super::*;

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
    fn test_mul_left() {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        let e0 = dense_expr();
        let e1 = sparse_expr();

        let m1 = matrix::dense(3,2,vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let m2 = matrix::dense(2,3,vec![1.0,2.0,3.0,4.0,5.0,6.0]);

        let e0_1 = m2.clone().mul(e0.clone());
        let e0_2 = 2.0.mul(e0.clone());

        let e1_1 = m2.clone().mul(e1.clone());
        let e1_2 = 2.0.mul(e1.clone());

        e0.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e0_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e0_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e1_1.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
        e1_2.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    }


    #[test]
    fn test_mul_right() {
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
    fn test_add() {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        let m1 = matrix::dense(3,3,vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]);

        let e0 = dense_expr().add(sparse_expr()).add(dense_expr().mul(m1));
        e0.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    }

    #[test]
    fn test_stack() {
        todo!("test_stack");
    }

    #[test]
    fn test_sum() {
        todo!("test_sum");
    }
}
