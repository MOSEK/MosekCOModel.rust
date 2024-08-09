extern crate itertools;

pub mod eval;
pub mod workstack;
mod dot;
mod mul;
mod add;

use std::ops::Range;

use itertools::izip;

use crate::matrix::Matrix;

use workstack::WorkStack;
use super::matrix;

pub use dot::{Dot,ExprDot};
pub use mul::*;
pub use add::*;
pub use super::domain;

/// The `ExprTrait<N>` represents a `N`-dimensional expression.
pub trait ExprTrait<const N : usize> {
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

    /// Create a dynamic expression from an expression. Expression types generally depend on the
    /// types of all the sub-expressions, so it is not possible to make e.g. an array of
    /// expressions unless they are exactly the same types. Wrapping each expression in a dynamic
    /// expression allows us to create structures like arrays that requires all elements to have
    /// the same type. The down-side is heap-allocated objects and that `eval()` calls are dynamic.
    ///
    /// # Example
    /// ```
    /// use mosekmodel::*;
    /// let mut M = Model::new(None);
    /// let v = M.variable(None, greater_than(0.0));
    /// 
    /// // Create a list of heterogenous expressions:
    /// // let l = &[ v.clone(), v.clone().add(v.clone()), v.clone().mul(2.0) ]; // invalid!
    /// let l = &[ v.clone().dynamic(), 
    ///            v.clone().add(v.clone()).dynamic(),
    ///            v.clone().mul(2.0).dynamic() ];
    /// ```
    fn dynamic<'a>(self) -> ExprDynamic<'a,N> where Self : Sized+'a { ExprDynamic::new(self) }

    /// Permute axes of the expression. Permute the index coordinates of each entry in the
    /// expression, and similarly permute the shape.
    ///
    /// # Arguments
    /// - `perm` - The permutation. This must be a valid permutation, i.e. it must be a permutation
    ///   of the range `0..N`. If the permutation is not valid, the function will panic.
    fn axispermute(self,perm : &[usize; N]) -> ExprPermuteAxes<N,Self> where Self:Sized { ExprPermuteAxes{item : self, perm: *perm } }

    /// Sum all elements in an expression producing a scalar expression.
    fn sum(self) -> ExprSum<N,Self> where Self:Sized { ExprSum{item:self} }

    fn neg(self) -> ExprMulScalar<N,Self> where Self:Sized {
        self.mul(-1.0)
    }

    /// Sum over a number of axes.
    ///
    /// # Arguments
    /// - `axes` - list of axes to preserve; all other dimensions are summed. The list must be
    ///   sorted and not contain duplicates.
    ///
    /// NOTE: The construction may seem a  but backward, but it s because we need to explicitly
    /// specify the dimensionality of the output. Rust does not support aritmetic with generic
    /// constants.
    fn sum_on<const K : usize>(self, axes : &[usize; K]) -> ExprReduceShape<N,K,ExprSumLastDims<N,ExprPermuteAxes<N,Self>>> where Self:Sized { 
        if K > N {
            panic!("Invalid axis specification")
        }
        else if axes.iter().zip(axes[1..].iter()).any(|(a,b)| a >= b) {
            panic!("Axis specification is unsorted or contains duplicates")
        }
        else if let Some(&last) = axes.last() {
            if last >= N {
                panic!("Axis specification is unsorted or contains duplicates")
            }
        }

        let mut perm = [0usize; N];
        perm[0..K].clone_from_slice(axes);
        {
            let (_,perm1) = perm.split_at_mut(K);
            let mut i = 0;
            let mut j = 0;
            for &a in axes {
                for ii in i..a {
                    unsafe { *perm1.get_unchecked_mut(j) = ii };
                    j += 1;
                }
                i = a+1;
            }
            for ii in i..N {
                unsafe { *perm1.get_unchecked_mut(j) = ii };
                j += 1;
            }
        }
        ExprReduceShape{
            item : ExprSumLastDims{ 
                num : N-K,
                item : ExprPermuteAxes{ 
                    item : self, 
                    perm 
                }
            }
        }
    }

    /// Add an expression and an item that is addable to an expression, e.g. constants or other
    /// expressions.
    ///
    /// # Arguments
    /// - `rhs` Add two expressions. The expression shapes must match.
    fn add<RHS>(self, rhs : RHS, ) -> ExprAdd<N,Self,RHS> 
        where 
            RHS : ExprTrait<N>,
            Self : Sized
    {
        ExprAdd::new(self,rhs,1.0,1.0) 
    }

    /// Subtract expression and an item that is addable to an expression, e.g. constants or other
    /// expressions.
    ///
    /// # Arguments
    /// - `rhs` Subtract two expressions. The expression shapes must match.
    fn sub<RHS>(self, rhs : RHS) -> ExprAdd<N,Self,RHS> 
        where 
            RHS : ExprTrait<N>,
            Self : Sized
    {
        ExprAdd::new(self,rhs,1.0,-1.0) 
    }

    /// Element-wise multiplication of two operands. The operand shapes must be the same.
    ///
    /// # Arguments
    /// - `other` Multiply element-wise. The shapes of the operands must match. 
    fn mul_elem<RHS>(self, other : RHS) -> RHS::Result where Self : Sized, RHS : ExprRightElmMultipliable<N,Self> { other.mul_elem(self) }

    /// An expression that produces a vector of dot-products of the rows of the operands,
    /// specifically, `[r_0,r_1,...] = [dot(e_{0,*},m_{0,*},dot(e_{1,*},m_{1,*},...]`, where `e` if
    /// an expression and `m` is a matrix. The shapes of the operands must be identical.
    ///
    /// # Arguments
    /// - `other` Matrix operand.
    fn dot_rows<M>(self, other : M) -> ExprReduceShape<2,1,ExprSumLastDims<2,ExprMulElm<2,Self>>>
        where 
            Self : Sized+ExprTrait<2>, 
            M : Matrix
    { 
        let (shape,sparsity,data) = other.dissolve();
        ExprReduceShape{
            item : ExprSumLastDims{
                num : 1,
                item : ExprMulElm{
                    datashape : shape,
                    datasparsity : sparsity,
                    data,
                    expr : self,
                }
            }
        }
    }

    /// Stack vertically, i.e. in first dimension. The two operands have the same number of
    /// dimensions, and must have the same shapes except in the first dimension.
    ///
    /// # Arguments
    /// - `other` The second operand.
    fn vstack<E:ExprTrait<N>>(self,other : E) -> ExprStack<N,Self,E>  where Self:Sized { ExprStack::new(self,other,0) }

    /// Stack horizontally, i.e. stack in second dimension. The two operands have the same number of
    /// dimensions, and must have the same shapes except in the second dimension.
    ///
    /// # Arguments
    /// - `other` The second operand.
    fn hstack<E:ExprTrait<N>>(self,other : E) -> ExprStack<N,Self,E>  where Self:Sized { ExprStack::new(self,other,1) }

    /// Stack in arbitrary dimension. The two operands have the same number of
    /// dimensions, and must have the same shapes except in dimension `dim`.
    ///
    /// # Arguments
    /// - `dim` The dimension in which to stack. This must be strictly less than `N`.
    /// - `other` The second operand.
    fn stack<E:ExprTrait<N>>(self,dim : usize, other : E) -> ExprStack<N,Self,E> where Self:Sized { ExprStack::new(self,other,dim) }

    /// Repeat a fixed number of times in some dimension. 
    ///
    /// # Arguments
    /// - `dim` Dimension in which to repeat. This must be strictly less than `N`.
    /// - `num` Number of times to repeat
    fn repeat(self,dim : usize, num : usize) -> ExprRepeat<N,Self> where Self:Sized { ExprRepeat{ expr : self, dim, num } }

    /// Take a slice of an expression
    fn slice(self,ranges : &[Range<usize>;N]) -> ExprSlice<N,Self> where Self:Sized {
        assert!(ranges.iter().all(|r| r.start <= r.end));
        let mut begin = [0usize;N];
        let mut end   = [0usize;N];
        izip!(begin.iter_mut(),end.iter_mut(),ranges.iter()).for_each(|(b,e,r)| { *b = r.start; *e = r.end; } );

        ExprSlice{expr : self, begin, end} 
    }

    /// Reshape the experssion. The new shape must match the old
    /// shape, meaning that the product of the dimensions are the
    /// same.
    ///
    /// # Arguments
    /// - `shape` The new shape.
    fn reshape<const M : usize>(self,shape : &[usize; M]) -> ExprReshape<N,M,Self>  where Self:Sized { ExprReshape{item:self,shape:*shape} }

    /// Create an expression that is symmetric in dimension `dim` and `dim+1`. The shape must
    /// satisfy the following:
    /// ```text
    /// shape[dim] * shape[dim+1] == n * (n+1) / 2
    /// ```
    /// for some integer `n.
    fn into_symmetric(self, dim : usize) -> ExprIntoSymmetric<N,Self> where Self:Sized {
        if dim > N-2 {
            panic!("Invalid symmetrization dimension");
        }
        
        ExprIntoSymmetric{
            dim,
            expr : self
        }
    }


    /// Flatten the expression into a vector. Preserve sparsity.
    fn flatten(self) -> ExprReshapeOneRow<N,1,Self> where Self:Sized { ExprReshapeOneRow { item:self, dim : 0 } }

    /// Flatten expression into a column, i.e. an expression of size `[n,1]` where
    /// `n=shape.iter().product()`.
    fn into_column(self) -> ExprReshapeOneRow<N,2,Self> where Self:Sized { ExprReshapeOneRow { item:self, dim : 0 } }

    /// Reshape an expression into a vector expression, where all but (at most) one dimension are
    /// of size 1.
    ///
    /// # Arguments
    /// - `i` - The dimension where the vector is defined, i.e the non-one dimension.
    fn into_vec<const M : usize>(self, i : usize) -> ExprReshapeOneRow<N,M,Self> where Self:Sized+ExprTrait<1> { 
        if i >= M {
            panic!("Invalid dimension index")
        }
        ExprReshapeOneRow{item:self, dim : i }
    }

    /// Reshape a sparse expression into a dense expression with the
    /// given shape. The shape must match the actual number of
    /// elements in the expression.
    fn gather(self) -> ExprGatherToVec<N,Self>  where Self:Sized { ExprGatherToVec{item:self} }

    /// Multiply `(self * other)`, where other must be right-multipliable with `Self`.
    ///
    /// # Arguments
    /// - `other` The right-hand matrix argument.
    fn mul<RHS>(self,other : RHS) -> RHS::Result where Self: Sized, RHS : ExprRightMultipliable<N,Self> { other.mul_right(self) }

    /// Multiply `(other * self)`, where other must be left-multipliable with `Self`.
    ///
    /// # Arguments
    /// - `lhs` The left-hand matrix argument.
    fn rev_mul<LHS>(self, lhs: LHS) -> LHS::Result where Self: Sized, LHS : ExprLeftMultipliable<N,Self> { lhs.mul(self) }

    /// Transpose a two-dimensional expression.
    fn transpose(self) -> ExprPermuteAxes<2,Self> where Self:Sized+ExprTrait<2> { ExprPermuteAxes{ item : self, perm : [1,0]} }

    /// Create a new expression with only lower triangular nonzeros from the operand.
    ///
    /// # Arguments
    /// - `with_diag` Indicating if the diagonal is included in the triangular non-zeros.
    fn tril(self,with_diag:bool) -> ExprTriangularPart<Self> where Self:Sized+ExprTrait<2> { ExprTriangularPart{item:self,upper:false,with_diag} }

    /// Create a new expression with only upper triangular nonzeros from the operand.
    ///
    /// # Arguments
    /// - `with_diag` Indicating if the diagonal is included in the triangular non-zeros.
    fn triu(self,with_diag:bool) -> ExprTriangularPart<Self> where Self:Sized+ExprTrait<2> { ExprTriangularPart{item:self,upper:true,with_diag} }

    /// create a new expression with only lower triangular nonzeros from the operand put into a
    /// vector in row-order.
    ///
    /// # arguments
    /// - `with_diag` indicating if the diagonal is included in the triangular non-zeros.
    fn trilvec(self,with_diag:bool) -> ExprGatherToVec<2,ExprTriangularPart<Self>> where Self:Sized+ExprTrait<2> { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:false,with_diag} } } 

    /// Create a new expression with only upper triangular nonzeros from the operand.
    ///
    /// # Arguments
    /// - `with_diag` Indicating if the diagonal is included in the triangular non-zeros.
    fn triuvec(self,with_diag:bool) -> ExprGatherToVec<2,ExprTriangularPart<Self>> where Self:Sized+ExprTrait<2> { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:true,with_diag} } }


    /// Take the diagonal elements if a square matrix and return it as a new vector expression.
    fn diag(self) -> ExprDiag<Self> where Self:Sized+ExprTrait<2> { ExprDiag{ item : self, anti : false, index : 0 } }

    /// Create a sparse square matrix with the vector expression elements as diagonal.
    fn square_diag(self) -> ExprSquareDiag<Self> where Self:Sized+ExprTrait<1> { ExprSquareDiag{ item : self }}


    // Explicit functions for performing left and right multiplcation with different types
    fn mul_any_scalar(self, c : f64) -> ExprMulScalar<N,Self> where Self : Sized { ExprMulScalar{ item : self, lhs : c } }
    fn mul_matrix_const_matrix<M>(self, m : &M) -> ExprMulRight<Self> where Self : Sized+ExprTrait<2>, M : Matrix { 
        ExprMulRight{
            item : self,
            shape : m.shape(),
            data : m.data().to_vec(),
            sp : m.sparsity().map(|v| v.to_vec())
        }
    }
    fn mul_rev_matrix_const_matrix<M>(self, m : &M) -> ExprMulLeft<Self> where Self:Sized+ExprTrait<2>, M : Matrix {
        ExprMulLeft{     
            item : self,
            shape : m.shape(),
            data : m.data().to_vec(),
            sp : m.sparsity().map(|v| v.to_vec())
        }
    }

    fn mul_matrix_vec(self,v : Vec<f64>) -> ExprReshapeOneRow<2,1,ExprMulRight<Self>> where Self:Sized+ExprTrait<2> {
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : self,
                shape : [ v.len(),1],
                data : v,
                sp : None },
            dim : 0
        }
    }
    fn mul_rev_matrix_vec(self, v : Vec<f64>) -> ExprReshapeOneRow<2,1,ExprMulLeft<Self>> where Self:Sized+ExprTrait<2> {
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : self,
                shape : [1,v.len()],
                data : v,
                sp : None },
            dim : 0 
        }
    }

    fn mul_vec_matrix<M>(self, m : &M) -> ExprReshapeOneRow<2,1,ExprMulRight<ExprReshapeOneRow<1,2,Self>>> where Self:Sized+ExprTrait<1>, M : Matrix {
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : ExprReshapeOneRow{ item : self, dim : 1 },
                shape : m.shape(),
                data : m.data().to_vec(),
                sp : m.sparsity().map(|v| v.to_vec()) },
            dim : 0
        }
    }
    fn mul_rev_vec_matrix<M>(self, m : &M) -> ExprReshapeOneRow<2,1,ExprMulRight<ExprReshapeOneRow<1,2,Self>>> where Self:Sized+ExprTrait<1>, M : Matrix {
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : ExprReshapeOneRow{ item : self, dim : 0 },
                shape : m.shape(),
                data : m.data().to_vec(),
                sp : m.sparsity().map(|v| v.to_vec()) },
            dim : 0
        }
    }
    fn mul_scalar_matrix<M>(self, m : &M) -> ExprReshape<1, 2, ExprMulElm<1, ExprRepeat<1, ExprReshape<0, 1, Self>>>> where Self : Sized+ExprTrait<0>, M : Matrix { 
        ExprReshape{
            item : ExprMulElm{
                expr : ExprRepeat {
                    expr : ExprReshape{ item : self, shape : [1] },
                    dim : 0,
                    num : m.height()*m.width()
                },
                data : m.data().to_vec(),
                datasparsity : m.sparsity().map(|s| s.to_vec()),
                datashape : [m.height()*m.width()]
            },
            shape : m.shape()
        }
    }
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// Expression Helper objects

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// Expression objects
/// Expr defines a literal expression with no sub-expressions
#[derive(Clone)]
pub struct Expr<const N : usize> {
    shape : [usize; N],
    aptr  : Vec<usize>,
    asubj : Vec<usize>,
    acof  : Vec<f64>,
    sparsity : Option<Vec<usize>>
}

/// The Expr implementation
impl<const N : usize> Expr<N> {
    /// Create a new literal expression from data.
    ///
    /// Arguments:
    /// * `shape` Shape of the expression. If `sparsity` is `None`,
    ///   the product of the dimensions in the shape must be equal to
    ///   the number of elements in the expression (`ptr.len()-1`)
    /// * `sparsity` If not `None`, this defines the sparsity
    ///   pattern. The pattern denotes the linear indexes if nonzeros in
    ///   the shape. It must be sorted, must contain no duplicates and
    ///   must fit within the `shape`.
    /// * `aptr` The number if elements is `aptr.len()-1`. `aptr` must
    ///   be ascending, so `aptr[i] <= aptr[i+1]`. `aptr` is a vector
    ///   if indexes of the starting points of each element in `asubj`
    ///   and `acof`, so element `i` consists of nonzeros defined by
    ///   `[asubj[aptr[i]..aptr[i+1]]], acof[aptr[i]..aptr[i+1]]`
    /// * `asubj` Variable subscripts.
    /// * `acof`  Coefficients.
    pub fn new(shape : &[usize;N],
               sparsity : Option<Vec<usize>>,
               aptr  : Vec<usize>,
               asubj : Vec<usize>,
               acof  : Vec<f64>) -> Expr<N> {
        let fullsize = shape.iter().product();
        if aptr.is_empty() { panic!("Invalid aptr"); }
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

            if ! sp.is_empty() && ! sp.iter().zip(sp[1..].iter()).all(|(&i0,&i1)| i0 < i1) {
                panic!("Sparsity is not sorted or contains duplicates");
            }
        }
        else if fullsize != aptr.len()-1 {
            panic!("Shape does not match number of elements");
        }

        Expr{
            aptr,
            asubj,
            acof,
            shape:*shape,
            sparsity
        }
    }

    pub fn reshape<const M : usize>(self,shape:&[usize;M]) -> Expr<M> {
        if self.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
            //println!("Invalid shape {:?} for this expression which has shape {:?}",self.shape,shape);
            panic!("Invalid shape {:?} for expression with shape {:?}",shape,self.shape);
        }

        Expr{
            aptr : self.aptr,
            asubj : self.asubj,
            acof : self.acof,
            shape : *shape,
            sparsity : self.sparsity
        }
    }

}


impl<const N : usize> ExprTrait<N> for Expr<N> {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let nnz  = self.asubj.len();
        let nelm = self.aptr.len()-1;

        let (aptr,sp,asubj,acof) = rs.alloc_expr(self.shape.as_slice(),nnz,nelm);

        if let (Some(ref ssp),Some(dsp)) = (&self.sparsity,sp) {
            dsp.clone_from_slice(ssp.as_slice())
        }

        aptr.clone_from_slice(self.aptr.as_slice());
        asubj.clone_from_slice(self.asubj.as_slice());
        acof.clone_from_slice(self.acof.as_slice());
    }
}

// An expression of any shape or size containing no non-zeros.
#[derive(Clone)]
pub struct ExprNil<const N : usize> { shape : [usize; N] }

impl<const N : usize> ExprTrait<N> for ExprNil<N> {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let (rptr,_,_,_) = rs.alloc_expr(self.shape.as_slice(),0,0);
        rptr[0] = 0;
    }
}

pub fn zeros<const N : usize>(shape : &[usize;N]) -> Expr<N> {
    Expr{
        shape : *shape,
        aptr : vec![0],
        asubj : vec![],
        acof : vec![],
        sparsity : Some(vec![]),
    }
}

pub fn const_expr<const N : usize>(shape : &[usize;N], value : f64) -> Expr<N> {
    let nelm : usize = shape.iter().product();
    Expr{
        shape : *shape,
        aptr : (0..(nelm+1)).collect(),
        asubj : vec![0usize; nelm],
        acof : vec![value; nelm],
        sparsity : None
    }
}

pub fn ones<const N : usize>(shape : &[usize;N]) -> Expr<N> {
    const_expr(shape,1.0)
}

pub fn const_diag(n : usize,value:f64) -> Expr<2> {
    Expr{
        shape : [n,n],
        aptr : (0..n+1).collect(),
        asubj : vec![0usize; n],
        acof : vec![value; n],
        sparsity : Some((0..n*n).step_by(n+1).collect())
    }
}
pub fn eye(n : usize) -> Expr<2> {
    const_diag(n,1.0)
}

pub fn constants<const N : usize>(shape : &[usize;N], values : &[f64]) -> Expr<N> {
    if shape.iter().product::<usize>() != values.len() {
        panic!("Data and shape do not match");
    }
    Expr{
        shape : *shape,
        aptr : (0..values.len()+1).collect(),
        asubj : vec![0; values.len()], 
        acof : values.to_vec(),
        sparsity : None
    }
}

pub fn nil<const N : usize>(shape : &[usize; N]) -> ExprNil<N> {
    if shape.iter().product::<usize>() != 0 {
        panic!("Shape must have at least one zero-dimension");
    }
    ExprNil{shape:*shape}
}


/// Reduce (or increase) the number of dimensions in the shape from `N` to `M`. If `M<N`, the
/// trailing `N-M+1` dimensions are flattened into one dimension. If `N<M` the shape is padded with
/// ones.
#[derive(Clone)]
pub struct ExprReduceShape<const N : usize, const M : usize, E> where E : ExprTrait<N>+Sized { item : E }
impl<const N : usize, const M : usize, E> ExprTrait<M> for ExprReduceShape<N,M,E> 
    where E : ExprTrait<N> 
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        eval::inplace_reduce_shape(M, rs, xs);
//        rs.validate_top().unwrap(); 
//        let (rshape,_) = xs.alloc(M,0);
//        {
//            let (shape,_,_,_,_) = rs.peek_expr();
//            if M <= N {
//                rshape[..M-1].clone_from_slice(&shape[0..M-1]);
//                *rshape.last_mut().unwrap() = shape[M-1..].iter().product();
//            }
//            else {
//                rshape[0..N].clone_from_slice(shape);
//                rshape[N..].iter_mut().for_each(|s| *s = 1);
//            }
//        }
//        rs.inline_reshape_expr(rshape).unwrap()
    }
}

/// For internal use. Reshape an expression into an M-dimensional expression where all but one
/// dimensions are 1. Unlike Reshape we don't need to to know the actual dimensions of either the
/// original or the resulting expression.
pub struct ExprReshapeOneRow<const N : usize, const M : usize, E:ExprTrait<N>> { item : E, dim : usize } 
impl<const N : usize,const M : usize,E> ExprReshapeOneRow<N,M,E> 
    where 
        E:ExprTrait<N> 
{
    pub fn new(dim : usize, item : E) -> ExprReshapeOneRow<N,M,E> {
        ExprReshapeOneRow{item,dim}
    }
}

impl<const N : usize, const M : usize, E:ExprTrait<N>> ExprTrait<M> for ExprReshapeOneRow<N,M,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        if self.dim >= M { panic!("Invalid dimension given"); }
        self.item.eval(rs,ws,xs);
        eval::inplace_reshape_one_row(M, self.dim, rs, xs)
            
//        let mut newshape = [ 0usize; M ]; newshape.iter_mut().for_each(|s| *s = 1 );
//        newshape[self.dim] = {
//            let (shp,_,_,_,_) = rs.peek_expr();
//            shp.iter().product()
//        };
//
//        rs.inline_reshape_expr(&newshape).unwrap();
    }
}

/// Reshape expression. The number of elements in the original expression and in the resized
/// expression must be the same.
pub struct ExprReshape<const N : usize, const M : usize, E:ExprTrait<N>> { item : E, shape : [usize; M] }
impl<const N : usize, const M : usize, E:ExprTrait<N>> ExprTrait<M> for ExprReshape<N,M,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        eval::inplace_reshape(self.shape.as_slice(), rs, xs);
//        self.item.eval(ws,rs,xs);
//        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
//
//        if self.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
//            panic!("Cannot reshape expression into given shape");
//        }
//
//        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),subj.len(),ptr.len()-1);
//
//        rptr.clone_from_slice(ptr);
//        rsubj.clone_from_slice(subj);
//        rcof.clone_from_slice(cof);
//        if let Some(rsp) = rsp {
//            if let Some(sp) = sp {
//                rsp.clone_from_slice(sp)
//            }
//        }
    }
}

/// A sparse expression where non-zeros are taken from a vector expression.
pub struct ExprScatter<const M : usize, E:ExprTrait<1>> { item : E, shape : [usize; M], sparsity : Vec<usize> }

impl<const M : usize, E:ExprTrait<1>> ExprScatter<M,E> {
    pub fn new(item     : E,
               shape    : &[usize; M],
               sparsity : Vec<usize>) -> ExprScatter<M,E> {

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
                         shape:*shape,
                         sparsity : perm.iter().map(|&p| unsafe{ *sparsity.get_unchecked(p)}).collect() }
        }
        else {
            ExprScatter{ item, shape: *shape, sparsity }
        }
    }
}

impl<const M : usize, E:ExprTrait<1>> ExprTrait<M> for ExprScatter<M,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::scatter(self.shape.as_slice(),self.sparsity.as_slice(), rs, ws, xs);
//        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
//
//        if ptr.len()-1 != self.sparsity.len() {
//            panic!("Sparsity pattern does not match number of elements in expression");
//        }
//
//        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),ptr.len()-1,subj.len());
//
//        rptr.clone_from_slice(ptr);
//        rsubj.clone_from_slice(subj);
//        rcof.clone_from_slice(cof);
//
//        if let Some(rsp) = rsp {
//            rsp.clone_from_slice(self.sparsity.as_slice())
//        }
    }
}

/// Pick nonzeros from a sparse expression to produce a dense expression with the given shape.
pub struct ExprGather<const N : usize, const M : usize, E:ExprTrait<N>> { item : E, shape : [usize; N] }
impl<const N : usize, const M : usize, E:ExprTrait<N>> ExprTrait<M> for ExprGather<N,M,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::gather(self.shape.as_slice(),rs,ws,xs);
//        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
//
//        if ptr.len()-1 != self.shape.iter().product::<usize>() {
//            panic!("Shape does not match number of elements in expression");
//        }
//
//        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),ptr.len()-1,subj.len());
//
//        rptr.clone_from_slice(ptr);
//        rsubj.clone_from_slice(subj);
//        rcof.clone_from_slice(cof);
    }
}

/// Pick nonzeros from a sparse expression to produce a dense vector expression.
pub struct ExprGatherToVec<const N : usize, E:ExprTrait<N>> { item : E }
impl<const N : usize, E:ExprTrait<N>> ExprTrait<1> for ExprGatherToVec<N,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::gather_to_vec(rs, ws, xs);
//        let (_shape,ptr,_sp,subj,cof) = ws.pop_expr();
//        let rnelm = ptr.len()-1;
//        let rnnz  = subj.len();        
//
//        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr( &[rnelm],rnnz,rnelm);
//
//        rptr.clone_from_slice(ptr);
//        rsubj.clone_from_slice(subj);
//        rcof.clone_from_slice(cof);
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

#[macro_export]
macro_rules! exprcat {
    [ $e0:expr ] => { $e0 };
    [ $e0:expr , $( $es:expr ),+ ] => { hstack![ $e0 $( , $es )* ] };
    [ $e0:expr ; $( $rest:tt )+ ] => { $e0 . vstack( exprcat![ $( $rest )* ]) };
    [ $e0:expr , $( $es:expr ),+ ; $( $rest:tt )+ ] => { hstack![ $e0 $( , $es )* ].vstack( exprcat![ $( $rest )*] ) };
}

pub struct ExprStack<const N : usize,E1:ExprTrait<N>,E2:ExprTrait<N>> {
    item1 : E1,
    item2 : E2,
    dim   : usize
}

pub struct ExprStackRec<const N : usize, E1:ExprStackRecTrait<N>,E2:ExprTrait<N>> {
    item1 : E1,
    item2 : E2,
    dim   : usize
}

pub trait ExprStackRecTrait<const N : usize> : ExprTrait<N> {
    fn stack_dim(&self) -> usize;
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize;
}

impl<const N : usize, E1:ExprTrait<N>,E2:ExprTrait<N>> ExprStack<N,E1,E2> {
    pub fn new(item1 : E1, item2 : E2, dim : usize) -> Self { ExprStack{item1,item2,dim} }
    pub fn stack<T:ExprTrait<N>>(self, dim : usize, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim} }
    pub fn vstack<T:ExprTrait<N>>(self, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim:0} }
    pub fn hstack<T:ExprTrait<N>>(self, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim:1} }
}

impl<const N : usize, E1:ExprStackRecTrait<N>,E2:ExprTrait<N>> ExprStackRec<N,E1,E2> {
    pub fn stack<T:ExprTrait<N>>(self, dim : usize, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim} }
    pub fn vstack<T:ExprTrait<N>>(self, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim:0} }
    pub fn hstack<T:ExprTrait<N>>(self, other : T) -> ExprStackRec<N,Self,T> { ExprStackRec{item1:self,item2:other,dim:1} }
}

impl<const N : usize,E1:ExprTrait<N>,E2:ExprTrait<N>> ExprTrait<N> for ExprStack<N,E1,E2> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(ws,rs,xs);
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
impl<const N : usize, E1:ExprTrait<N>,E2:ExprTrait<N>> ExprStackRecTrait<N> for ExprStack<N,E1,E2> {
    fn stack_dim(&self) -> usize { self.dim }
    fn eval_rec(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> usize {
        self.item2.eval(rs,ws,xs);
        self.item1.eval(rs,ws,xs);
        2
    }
}

impl<const N : usize, E1:ExprStackRecTrait<N>,E2:ExprTrait<N>> ExprTrait<N> for ExprStackRec<N,E1,E2> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        let n = self.eval_rec(ws,rs,xs);
        eval::stack(self.dim,n,rs,ws,xs);
    }
}
impl<const N : usize, E1:ExprStackRecTrait<N>,E2:ExprTrait<N>> ExprStackRecTrait<N> for ExprStackRec<N,E1,E2> {
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

pub struct ExprRepeat<const N : usize, E : ExprTrait<N>> {
    expr : E,
    dim : usize,
    num : usize
}
impl<const N : usize, E : ExprTrait<N>> ExprTrait<N> for ExprRepeat<N,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::repeat(self.dim,self.num,rs,ws,xs);
    }
}



pub struct ExprDynamic<'a,const N : usize> {
    expr : Box<dyn ExprTrait<N>+'a>
}

impl<'a,const N : usize> ExprDynamic<'a,N> {
    fn new<E>(e : E) -> ExprDynamic<'a,N> where E : ExprTrait<N>+'a {
        ExprDynamic{
            expr : Box::new(e)
        }
    }
}

impl<'a,const N : usize> ExprTrait<N> for ExprDynamic<'a,N> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(rs,ws,xs);
    }
}


/// Dynamic stacking. To stack a list of heterogenous expressions we
/// need to create a list of dynamic ExprTraits

pub struct ExprDynStack<const N : usize> {
    exprs : Vec<ExprDynamic<'static,N>>,
    dim   : usize
}

impl<const N : usize> ExprTrait<N> for ExprDynStack<N> {
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
pub fn stack<const N : usize>(dim : usize, exprs : Vec<ExprDynamic<'static,N>>) -> ExprDynStack<N> {
    ExprDynStack{exprs,dim}
}
pub fn vstack<const N : usize>(exprs : Vec<ExprDynamic<'static, N>>) -> ExprDynStack<N> {
    ExprDynStack{exprs,dim:0}
}
pub fn hstack<const N : usize>(exprs : Vec<ExprDynamic<'static,N>>) -> ExprDynStack<N> {
    ExprDynStack{exprs,dim:1}
}


////////////////////////////////////////////////////////////
//
//
pub struct ExprSlice<const N : usize, E : ExprTrait<N>> {
    expr : E,
    begin : [usize; N],
    end : [usize; N]
}

impl<const N : usize, E> ExprTrait<N> for ExprSlice<N,E> where E : ExprTrait<N> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::slice(&self.begin,&self.end,rs,ws,xs);

//        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
//        let nnz = *ptr.last().unwrap();
//        let nelem = ptr.len()-1;
//
//        // check indexes
//        assert!(shape.len() == N);
//        if shape.iter().zip(self.end.iter()).any(|(a,b)| a < b) { panic!("Index out of bounds") }
//
//        let mut strides : [usize; N] = [0; N];
//        let mut rstrides : [usize; N] = [0; N];
//        let _ = strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v*d } );
//        let _ = izip!(rstrides.iter_mut(),self.begin.iter(),self.end.iter()).rev().fold(1,|v,(s,&b,&e)| { *s = v; v*(e-b) } );
//
//        let mut rshape = [0usize; N];
//        izip!(rshape.iter_mut(),self.begin.iter(),self.end.iter()).for_each(|(r,&b,&e)| *r = e-b );
//
//        if let Some(sp) = sp {
//            let (upart,xcof) = xs.alloc(nelem*2+1+nnz,nnz);
//            let (xptr,upart) = upart.split_at_mut(nelem+1);
//            let (xsp,xsubj) = upart.split_at_mut(nelem);
//            xptr[0] = 0;
//
//            let mut rnnz = 0usize;
//            let mut rnelem = 0usize;
//            for ((&spi,&p0,&p1),(xp,xs)) in 
//                izip!(sp.iter(), ptr.iter(), ptr[1..].iter())
//                    .filter(|(&spi,_,_)| {
//                        let mut ix = [0usize; N];
//                        let _ = ix.iter_mut().zip(strides.iter()).fold(spi,|v,(i,&s)| { *i = v / s; v % s});
//                       izip!(ix.iter(),self.begin.iter(),self.end.iter()).all(|(&i,&b,&e)| b <= i && i < e ) })
//                    .zip(xptr[1..].iter_mut().zip(xsp.iter_mut())) {
//
//                let mut ix = [0usize; N];
//                let _ = ix.iter_mut().zip(strides.iter()).fold(spi,|v,(i,&s)| { *i = v / s; v % s});
//               
//                (*xs,_) = izip!(strides.iter(),rstrides.iter(),self.begin.iter()).fold((spi,0),|(i,r),(&s,&rs,&b)| (i%s, r + (i/s-b) * rs) );
//                xsubj.clone_from_slice(&subj[p0..p1]);
//                xcof.clone_from_slice(&cof[p0..p1]);
//                rnnz += p1-p0;                    
//                *xp = rnnz;
//                rnelem += 1;
//            }
//
//            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelem);
//            rptr[0] = 0;
//            rptr.clone_from_slice(&xptr[..rnelem+1]);
//            rsubj.clone_from_slice(&xsubj[..rnnz]);
//            rcof.clone_from_slice(&xcof[..rnnz]);
//            if let Some(rsp) = rsp {
//                rsp.clone_from_slice(&xsp[..rnelem]);
//            }
//        } 
//        else {
//            let rnelem : usize = self.begin.iter().zip(self.end.iter()).map(|(&a,&b)| b-a).product(); 
//            let (upart,xcof) = xs.alloc(rnelem+1+nnz,nnz);
//            let (xptr,xsubj) = upart.split_at_mut(rnelem+1);
//            let mut rnnz = 0usize;
//            {
//                xptr[0] = 0;
//                //let mut idx : [usize; N] = self.begin;
//                for (xp,ridx) in xptr[1..].iter_mut().zip(rshape.index_iterator()) {
//                    let sofs = izip!(ridx.iter(), strides.iter(), self.begin.iter())
//                        .fold(0,|v,(&i,&s,&b)| v+(i+b)*s);
//
//                    let (&p0,&p1) = unsafe{ (ptr.get_unchecked(sofs),ptr.get_unchecked(sofs+1)) };
//                    
//                    xsubj[rnnz..rnnz+p1-p0].copy_from_slice(&subj[p0..p1]);
//                    xcof[rnnz..rnnz+p1-p0].copy_from_slice(&cof[p0..p1]);
//                    rnnz += p1-p0;
//                    *xp = rnnz;
//                }
//            }
//            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, rnelem);
//            rptr.clone_from_slice(xptr);
//            rsubj.clone_from_slice(&xsubj[..rnnz]);
//            rcof.clone_from_slice(&xcof[..rnnz]);
//        }
    }
}


////////////////////////////////////////////////////////////
//

/// Expression that sums all elements in an expression
pub struct ExprSum<const N : usize, T:ExprTrait<N>> {
    item : T
}

pub struct ExprSumLastDims<const N : usize, T : ExprTrait<N>> {
    item : T,
    num : usize
}

impl<const N : usize, T:ExprTrait<N>> ExprTrait<0> for ExprSum<N,T> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::sum(rs,ws,xs);
    }
}

impl<const N : usize, E:ExprTrait<N>> ExprTrait<N> for ExprSumLastDims<N,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::sum_last(self.num,rs,ws,xs)
    }
}

////////////////////////////////////////////////////////////
//

/// 
pub struct ExprTriangularPart<T:ExprTrait<2>> {
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
impl<T:ExprTrait<2>> ExprTrait<2> for ExprTriangularPart<T> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::triangular_part(self.upper, self.with_diag, rs, ws, xs);

//        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
//
//        let nd = shape.len();
//
//        if nd != 2 || shape[0] != shape[1] {
//            panic!("Triangular parts can only be taken from square matrixes");
//        }
//        let d = shape[0];
//
//        if let Some(sp) = sp {
//            let rnelm = sp.iter()
//                .filter(|&spi| { let (i,j) = (spi / d, spi % d); (self.upper && i < j) || (! self.upper && i > j) || (self.with_diag && i == j) })
//                .count(); 
//            let rnnz = izip!(sp.iter().map(|i| (i/d,i%d)),ptr.iter(),ptr[1..].iter())
//                .filter(|((i,j),_,_)| (self.upper && i < j) || (! self.upper && i > j) || (self.with_diag && i == j) )
//                .map(|(_,pb,pe)| pe-pb)
//                .sum();
//            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);
//
//            izip!(subj.chunks_by(ptr),
//                  cof.chunks_by(ptr),
//                  sp.iter().map(|i| (i/d,i%d)))
//                .filter(|(_,_,(i,j))| (self.upper && i < j) || (! self.upper && i > j) || (self.with_diag && i == j))
//                .flat_map(|(subj,cof,_)| subj.iter().zip(cof.iter()))
//                .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
//                .for_each(|((&sj,&sc),(tj,tc))| { *tj = sj; *tc = sc; });
//
//            rptr[0] = 0;
//            let rsp = rsp.unwrap();
//            izip!(ptr.iter(),ptr[1..].iter(),sp.iter().map(|i| (i/d,i%d)))
//                .filter(|(_,_,(i,j))| (self.upper && i < j) || (! self.upper && i > j) || (self.with_diag && i == j))
//                .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
//                .for_each(|((&pb,&pe,(i,j)),(rp,spi))|  { *rp = pe-pb; *spi = i * d + j } );
//            rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
//        }
//        else {  
//            //println!("  case 2: Dense");
//            let rnelm = if self.with_diag { d * (d+1)/2 } else { d * (d-1) / 2 };
//            let rnnz : usize = match (self.upper,self.with_diag) {
//                (true,true)   => ptr.iter().step_by(d+1).zip(ptr[d..].iter().step_by(d)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
//                (true,false)  => ptr[1..].iter().step_by(d+1).zip(ptr[d..].iter().step_by(d)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
//                (false,true)  => ptr.iter().step_by(d).zip(ptr[1..].iter().step_by(d+1)).map(|(&p0,&p1)| p1-p0).sum::<usize>(),
//                (false,false) => ptr[d..].iter().step_by(d).zip(ptr[d+1..].iter().step_by(d+1)).map(|(&p0,&p1)| p1-p0).sum::<usize>()
//            };
//
//            //println!("======== upper = {}, ptr = {:?}, subj = {:?}",self.upper,ptr,subj);
//            //println!("  Alloc: shape = {:?}, rnnz = {}, rnelm = {}",shape,rnnz,rnelm);
//            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,rnnz,rnelm);
//
//            izip!(subj.chunks_by(ptr),
//                  cof.chunks_by(ptr),
//                  iproduct!(0..d,0..d))
//                .filter(|(_,_,(i,j))| 
//                    (self.upper     && i  < j) ||
//                    (! self.upper   && i  > j) ||
//                    (self.with_diag && i == j))
//                .flat_map(|(subj,cof,_)| subj.iter().zip(cof.iter()))
//                .zip(rsubj.iter_mut().zip(rcof.iter_mut()))
//                .for_each(|((&sj,&sc),(tj,tc))| { *tj = sj; *tc = sc; });
//
//            rptr[0] = 0;
//            let rsp = rsp.unwrap();
//            izip!(ptr.iter(),ptr[1..].iter(),iproduct!(0..d,0..d))
//                .filter(|(_,_,(i,j))| 
//                    (self.upper     && i  < j) ||
//                    (! self.upper   && i  > j) ||
//                    (self.with_diag && i == j))
//                .zip(rptr[1..].iter_mut().zip(rsp.iter_mut()))
//                .for_each(|((&pb,&pe,(i,j)),(rp,spi))|  { *rp = pe-pb; *spi = i * d + j } );
//            rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
//
//            let _ = rptr.iter_mut().fold(0,|v,p| { *p += v; *p });
//        }
    }
}

pub struct ExprDiag<E:ExprTrait<2>> {
    item : E,
    anti : bool,
    index : i64
}

impl<E:ExprTrait<2>> ExprTrait<1> for ExprDiag<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);

        eval::diag(self.anti, self.index, rs, ws, xs);
    }
}

pub struct ExprSquareDiag<E : ExprTrait<1>> {
    item : E
}

impl<E:ExprTrait<1>> ExprTrait<2> for ExprSquareDiag<E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);

        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        if shape.len() != 1 { panic!("Operand has invalid shape {:?}, expected a vector", shape); }
        let n = shape[0];

        let rshape = [n,n];
        let rnnz = *ptr.last().unwrap();
        let rnelm = n;

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&rshape, rnnz, n);
        
        rptr.copy_from_slice(ptr);
        rsubj.copy_from_slice(subj);
        rcof.copy_from_slice(cof);

        if let Some(sp) = sp {
            rsp.unwrap().iter_mut().zip(sp.iter()).for_each(|(ri,&i)| *ri = i * (n+1));
        }
        else {
            rsp.unwrap().iter_mut().enumerate().for_each(|(i,ri)| *ri = i * (n+1));
        }
    }
}


pub struct ExprIntoSymmetric<const N : usize, E : ExprTrait<N>> {
    dim : usize,
    expr : E
}

impl<const N : usize, E : ExprTrait<N>> ExprIntoSymmetric<N,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        eval::into_symmetric(self.dim,rs,ws,xs)
    }
}

pub struct ExprPermuteAxes<const N : usize, E:ExprTrait<N>> {
    item : E,
    perm : [usize; N]
}

impl<const N : usize, E:ExprTrait<N>> ExprTrait<N> for ExprPermuteAxes<N,E> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        eval::permute_axes(&self.perm,rs,ws,xs)
    }
}


//impl From<f64> for Expr<0> {
//    fn from(self) -> expr<0> { expr::new(&[], none, vec![0,1], vec![0], vec![self]) }
//}




//impl Into<Expr<0>> for f64 {
//    fn into(self) -> Expr<0> { Expr::new(&[], None, vec![0,1], vec![0], vec![self]) }
//}
//
//impl Into<Expr<1>> for &[f64] {
//    fn into(self) -> Expr<1> { Expr::new(&[self.len()], None, (0..self.len()+1).collect(), vec![0; self.len()], self.to_vec()) }
//}
//
//impl Into<Expr<1>> for Vec<f64> {
//    fn into(self) -> Expr<1> { Expr::new(&[self.len()], None, (0..self.len()+1).collect(), vec![0; self.len()], self.clone()) }
//}
//

impl From<f64> for Expr<0> {
    fn from(v : f64) -> Expr<0> { Expr::new(&[], None, vec![0,1], vec![0], vec![v]) }
}

impl From<&[f64]> for Expr<1> {
    fn from(v : &[f64]) -> Expr<1> { Expr::new(&[v.len()], None, (0..v.len()+1).collect(), vec![0; v.len()], v.to_vec()) }
}

impl From<Vec<f64>> for Expr<1> {
    fn from(v : Vec<f64>) -> Expr<1> { Expr::new(&[v.len()], None, (0..v.len()+1).collect(), vec![0; v.len()], v) }
}


////////////////////////////////////////////////////////////
//
// Tests

#[allow(unused)]
#[cfg(test)]
mod test {
    use crate::*;
    use crate::matrix::*;
    use crate::expr::*;
    use crate::variable::*;

    fn eq<T:std::cmp::Eq>(a : &[T], b : &[T]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a,b)| *a == *b )
    }

    fn dense_expr() -> Expr<2> {
        super::Expr::new(&[3,3],
                         None,
                         vec![0,1,2,3,4,5,6,7,8,9],
                         vec![0,1,2,0,1,2,0,1,2],
                         vec![1.1,1.2,1.3,2.1,2.2,2.3,3.1,3.2,3.3])
    }

    fn sparse_expr() -> Expr<2> {
        super::Expr::new(&[3,3],
                         Some(vec![0,4,5,6,7]),
                         vec![0,1,2,3,4,5],
                         vec![0,1,2,3,4],
                         vec![1.1,2.2,3.3,4.4,5.5])
    }


    #[test]
    fn into_symmetric() {
        { // dense
            let mut rs = WorkStack::new(512);
            let mut ws = WorkStack::new(512);
            let mut xs = WorkStack::new(512);
            // 0 1 3
            // 1 2 4
            // 3 4 5
            let e = Expr::new(&[6,1],
                              None,
                              vec![0,1,2,3,4,5,6],
                              vec![0,1,2,3,4,5],
                              vec![1.1,2.1,2.2,3.1,3.2,3.3]);
            let es = e.into_symmetric(0);
            es.eval(& mut rs,& mut ws,& mut xs);

            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert!(sp.is_none());
            assert!(shape.len() == 2);
            assert!(shape[0] == 3 && shape[1] == 3);
            assert_eq!(ptr,  &[0,1,2,3,4,5,6,7,8,9]);
            assert_eq!(subj, &[0,1,3,1,2,4,3,4,5]);
            assert!(ws.is_empty());
            assert!(rs.is_empty());
        }

        { // sparse
            let mut rs = WorkStack::new(512);
            let mut ws = WorkStack::new(512);
            let mut xs = WorkStack::new(512);

            // 0 . 2 3 . 5
            //
            // | 0     |
            // | . 2   |
            // | 3 . 5 |
            // 
            let e = Expr::new(&[6,1],
                              Some(vec![0,2,3,5]),
                              vec![0,1,2,3,4],
                              vec![0,2,3,5],
                              vec![1.1,2.2,3.1,3.3]);
            let es = e.into_symmetric(0);
            es.eval(& mut rs,& mut ws,& mut xs);

            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(sp.unwrap(), &[0,2,4,6,8]);
            assert!(shape.len() == 2);
            assert!(shape[0] == 3 && shape[1] == 3);
            assert_eq!(ptr,  &[0,1,2,3,4,5]);
            assert_eq!(subj, &[0,3,2,3,5]);
            assert!(ws.is_empty());
            assert!(rs.is_empty());
        }
    }

    #[test]
    fn mul_left() {
        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        let e0 = dense_expr();
        let e1 = sparse_expr();

        let m1 = matrix::dense([3,2],vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let m2 = matrix::dense([2,3],vec![1.0,2.0,3.0,4.0,5.0,6.0]);

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

        let m1 = matrix::dense([3,2],vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let m2 = matrix::dense([2,3],vec![1.0,2.0,3.0,4.0,5.0,6.0]);

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

        let m1 = matrix::dense([3,3],vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]);

        let e0 = dense_expr().add(sparse_expr()).add(dense_expr().mul(m1));
        e0.eval(& mut rs,& mut ws,& mut xs); assert!(ws.is_empty()); rs.clear();
    }

    #[test]
    fn repeat() {
        let ed = super::Expr::new(&[3,2,1],
                                  None,
                                  (0..7).collect(),
                                  (0..6).collect(),
                                  (0..6).map(|v| v as f64 * 1.1).collect());
        let es = super::Expr::new(&[3,2,1],
                                  Some(vec![0,2,3,5]),
                                  (0..5).collect(),
                                  vec![6,8,9,11],
                                  (0..4).map(|v| v as f64 * 1.1).collect());

        // | 0 1 |      | 6  . |
        // | 2 3 |      | 8  9 |
        // | 4 5 |      | . 11 |

        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);

        ed.clone().repeat(0,2).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape[0],6);
        assert_eq!(shape[1],2);
        assert_eq!(shape[2],1);
        assert_eq!(*ptr.last().unwrap(), 12);
        assert!(sp.is_none());
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]);
        assert_eq!(subj,&[0,1,2,3,4,5,0,1,2,3,4,5]);

        rs.clear();
        ws.clear();
        xs.clear();

        ed.clone().repeat(1,2).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape[0],3);
        assert_eq!(shape[1],4);
        assert_eq!(shape[2],1);
        assert_eq!(*ptr.last().unwrap(), 12);
        assert!(sp.is_none());
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]);
        assert_eq!(subj,&[0,1,0,1,2,3,2,3,4,5,4,5]);
        
        rs.clear();
        ws.clear();
        xs.clear();

        ed.clone().repeat(2,2).eval(& mut rs,& mut ws,& mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape[0],3);
        assert_eq!(shape[1],2);
        assert_eq!(shape[2],2);
        assert_eq!(*ptr.last().unwrap(), 12);
        assert!(sp.is_none());
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12]);
        assert_eq!(subj,&[0,0,1,1,2,2,3,3,4,4,5,5]);
        
        // | 0 1 |      | 6  . |
        // | 2 3 |      | 8  9 |
        // | 4 5 |      | . 11 |
        rs.clear();
        ws.clear();
        xs.clear();

        es.clone().repeat(0,2).eval(&mut rs, &mut ws, &mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape,&[6,2,1]);
        assert_eq!(*ptr.last().unwrap(), 8);
        assert!(sp.is_some());
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8]);
        assert_eq!(subj,&[6,8,9,11,6,8,9,11]);

        rs.clear(); 
        ws.clear();
        xs.clear();

        es.clone().repeat(1,2).eval(&mut rs, &mut ws, &mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape,&[3,4,1]);
        assert_eq!(*ptr.last().unwrap(), 8);
        assert!(sp.is_some());
        assert_eq!(sp.unwrap(),&[0,2,4,5,6,7,9,11]);
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8]);
        assert_eq!(subj,&[6,6,8,9,8,9,11,11]);
        
        rs.clear();
        ws.clear();
        xs.clear();

        es.clone().repeat(2,2).eval(&mut rs, &mut ws, &mut xs);
        let (shape,ptr,sp,subj,_cof) = rs.pop_expr();
        assert_eq!(shape.len(),3);
        assert_eq!(shape,&[3,2,2]);
        assert_eq!(*ptr.last().unwrap(), 8);
        assert!(sp.is_some());
        assert_eq!(sp.unwrap(),&[0,1,4,5,6,7,10,11]);
        assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8]);
        assert_eq!(subj,&[6, 6, 8, 8,9,9,11,11]);
    }
        
    #[test]
    fn stack() {
        let e0 = super::Expr::new(&[3,2,1],
                                  None,
                                  (0..7).collect(),
                                  (0..6).collect(),
                                  (0..6).map(|v| v as f64 * 1.1).collect());
        let e1 = super::Expr::new(&[3,2,1],
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



        {
            let mut rs = WorkStack::new(512);
            let mut ws = WorkStack::new(512);
            let mut xs = WorkStack::new(512);
            let ed = Expr::new(&[2,2],None,vec![0,1,2,3,4],vec![1,2,3,4],vec![1.1,1.2,2.1,2.2]);
            let es = Expr::new(&[2,2],Some(vec![]), vec![0], vec![], vec![]);

            vstack![hstack![ed.clone(),es.clone()], hstack![es,ed]].eval(& mut rs,& mut ws,& mut xs);

            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(shape,&[4,4]);
            assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8]);
            assert!(sp.is_some());
            assert_eq!(sp.unwrap(),&[0,1,4,5,10,11,14,15]);
            assert_eq!(subj,&[1,2,3,4,1,2,3,4]);
        }
        {
            let mut rs = WorkStack::new(512);
            let mut ws = WorkStack::new(512);
            let mut xs = WorkStack::new(512);
            let ed = Expr::new(&[2,2],None,vec![0,1,2,3,4],vec![1,2,3,4],vec![1.1,1.2,2.1,2.2]);
            let es = Expr::new(&[2,2],Some(vec![]), vec![0], vec![], vec![]);

            exprcat![
                ed.clone(), es.clone() ;
                es.clone(), ed.clone() ].eval(& mut rs,& mut ws,& mut xs);

            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(shape,&[4,4]);
            assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8]);
            assert!(sp.is_some());
            assert_eq!(sp.unwrap(),&[0,1,4,5,10,11,14,15]);
            assert_eq!(subj,&[1,2,3,4,1,2,3,4]);                        
        }
    }

    #[allow(non_snake_case)]
    #[test]
    fn slice() {
        let mut m = Model::new(None);
        let t = m.variable(Some("t"),unbounded().with_shape(&[2])); // 1,2
        let X = m.variable(Some("X"), in_psd_cone(4)); // 3,4,5,6, 7,8,9, 10,11, 12
        //     | 3 4  5  6 |
        // X = | 4 7  8  9 |
        //     | 5 8 10 11 |
        //     | 6 9 11 12 |
        let Y = m.variable(Some("Y"), in_psd_cone(2)); // 13,14,15
        let mx = dense([2,2], vec![1.1,2.2,3.3,4.4]);

        m.constraint(Some("X-Y"), &X.clone().slice(&[0..2,0..2]).sub(Y.clone().sub((&mx).mul_right(t.clone().clone().index(0)))), domain::zeros(&[2,2]));
        m.write_problem("X_minus_Y.ptf");

        let mut rs = WorkStack::new(512);
        let mut ws = WorkStack::new(512);
        let mut xs = WorkStack::new(512);
        {
            rs.clear(); ws.clear(); xs.clear();
            X.clone().eval(&mut rs,&mut ws,&mut xs);
            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(shape,&[4,4]);
            assert_eq!(ptr,&[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
            assert_eq!(subj,&[3,4,5,6, 4,7,8,9, 5,8,10,11, 6,9,11,12]);
            println!("subj = {:?}",subj);
        }
        {
            rs.clear(); ws.clear(); xs.clear();
            X.clone().slice(&[0..2,0..2]).eval(&mut rs,&mut ws,&mut xs);
            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(shape,&[2,2]);
            assert_eq!(ptr,&[0,1,2,3,4]);
            assert_eq!(subj,&[3,4,4,7]);
            println!("subj = {:?}",subj);
        }
        {
            rs.clear(); ws.clear(); xs.clear();
            X.clone().slice(&[0..2,0..2]).sub(Y.sub((&mx).mul_right(t.clone().index(0)))).eval(&mut rs,&mut ws,&mut xs);
            let (shape,ptr,sp,subj,cof) = rs.pop_expr();
            assert_eq!(shape,&[2,2]);
            assert_eq!(ptr,&[0,3,6,9,12]);
            assert_eq!(subj,&[1,13,3, 1,14,4, 1,14,4, 1,15,7]);
            println!("subj = {:?}",subj);
        }
    }



}
