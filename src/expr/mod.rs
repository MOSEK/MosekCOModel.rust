extern crate itertools;

mod eval;
pub mod workstack;

use itertools::{iproduct};
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
    fn mul_left_dense(self, v : matrix::DenseMatrix) -> ExprMulLeftDense<Self> { ExprMulLeftDense{item:self,lhs:v} }
    fn mul_right_dense(self, v : matrix::DenseMatrix) -> ExprMulRightDense<Self> { ExprMulRightDense{item:self,rhs:v} }
    // fn transpose(self) -> ExprPermuteAxes<Self>
    // fn axispermute(self) -> ExprPermuteAxes<Self>
    // fn slice(self, range : &[(Range<usize>)])

    fn mul<V>(self,other : V) -> V::Result where V : ExprRightMultipliable<Self> { other.mul_right(self) }
    fn add<R:ExprTrait>(self,rhs : R) -> ExprAdd<Self,R> { ExprAdd{lhs:self,rhs} }
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


impl ExprTrait for super::Variable {
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


mod matrix {
    //use itertools::{izip};
    use super::{ExprRightMultipliable,ExprTrait,ExprMulLeftDense,ExprMulRightDense};

    pub fn dense(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix { DenseMatrix::new(height,width,data) }
    pub fn sparse(height : usize, width : usize,
                  subi : &[usize],
                  subj : &[usize],
                  cof  : &[f64]) -> SparseMatrix {
        if subi.len() != subj.len() || subi.len() != cof.len() {
            panic!("Invalid matrix data")
        }

        if subi.iter().max().copied().unwrap_or(0) >= height
            || subj.iter().max().copied().unwrap_or(0) >= width {
                panic!("Invalid matrix data")
            }

        let mut perm : Vec<usize> = (0..subi.len()).collect();
        perm.sort_by_key(|&k| unsafe{(*subi.get_unchecked(k),*subj.get_unchecked(k)) });

        if ! perm.iter().zip(perm[1..].iter()).map(|(&p0,&p1)| unsafe{(*subi.get_unchecked(p0),*subi.get_unchecked(p1),*subj.get_unchecked(p0),*subj.get_unchecked(p1))})
            .all(|(i0,i1,j0,j1)| i0 < i1 || (i0 == i1 && j0 < j1)) {
                panic!("Matrix contains duplicates");
            }

        SparseMatrix{
            dim : (height,width),
            sp  : perm.iter().map(|&p| unsafe{*subi.get_unchecked(p)}*width+unsafe{*subj.get_unchecked(p)}).collect(),
            data : perm.iter().map(|&p| unsafe{*cof.get_unchecked(p)}).collect()
        }
    }

    pub fn from_triplets(height : usize,
                         width : usize,
                         data : &[(usize,usize,f64)]) -> SparseMatrix {
        if data.iter().max_by_key(|&v| v.0).map(|&v| v.0 >= height).unwrap_or(false)
            || data.iter().max_by_key(|&v| v.1).map(|&v| v.1 >= width).unwrap_or(false) {
                panic!("Invalid matrix data")
            }

        let mut perm : Vec<usize> = (0..data.len()).collect();
        perm.sort_by_key(|&k| { let d = unsafe{*data.get_unchecked(k)}; (d.0,d.1) });

        if ! perm.iter().zip(perm[1..].iter()).map(|(&p0,&p1)| unsafe{(*data.get_unchecked(p0),*data.get_unchecked(p1))})
            .all(|(i0,i1)| i0.0 < i1.0 || (i0.0 == i1.0 && i0.1 < i1.1)) {
                panic!("Matrix contains duplicates");
            }
        SparseMatrix{
            dim : (height,width),
            sp  : perm.iter().map(|&p| { let i = unsafe{data.get_unchecked(p)}; i.0*width+i.1 }).collect(),
            data : perm.iter().map(|&p| unsafe{data.get_unchecked(p)}.2 ).collect()
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


    #[derive(Clone)]
    pub struct DenseMatrix {
        dim  : (usize,usize),
        data : Vec<f64>
    }

    #[derive(Clone)]
    pub struct SparseMatrix {
        dim  : (usize,usize),
        sp   : Vec<usize>,
        data : Vec<f64>,
    }

    impl DenseMatrix {
        pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
            if height*width != data.len() { panic!("Invalid data size for matrix")  }
            DenseMatrix{
                dim : (height,width),
                data : data
            }
        }
        pub fn dim(&self) -> (usize,usize) { self.dim }
        pub fn height(&self) -> usize { self.dim.0 }
        pub fn width(&self) -> usize { self.dim.1 }
        pub fn data(&self) -> &[f64] { self.data.as_slice() }
    }

    impl<E:ExprTrait> ExprRightMultipliable<E> for DenseMatrix {
        type Result = ExprMulRightDense<E>;
        fn mul_right(self,other : E) -> Self::Result { other.mul_right_dense(self) }
    }

    impl DenseMatrix {
        fn mul<E:ExprTrait>(self,other : E) -> ExprMulLeftDense<E> { ExprMulLeftDense{item : other,lhs : self} }
    }
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






#[cfg(test)]
mod test {
    #[test]
    fn test_exprs() {

    }
}
