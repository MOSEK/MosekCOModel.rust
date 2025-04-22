//! Structures and functions for formulating disjunctive constraints.
//!
//! A disjunctive constraint is a constraint of the form 
//! $$
//!   A_1x+b_1\\in K_1\\ \\mathrm{or}\\ldots\\ \\mathrm{or}\\ A_nx+b_n\\in K_n
//! $$
//! Each \\(A_ix+b_i\\) is called a _term_ (see [crate::model::constr]). A term can be composed if multiple _clauses_:
//! $$
//!   \\left[
//!     \\begin{array}{c}
//!       A_{i,1}x+b_{i,1}\\\\
//!       \\vdots \\\\
//!       A_{i,m_i}x+b{i,m_i}
//!     \\end{array}
//!   \\right]x
//!   +\\left[\\begin{array}{c}b_{i,1}\\\\ \\vdots \\\\ b_{i,m_i} \\end{array}\\right]
//!   \\in 
//!   K_i = K_{i,1}\\times\\cdots K_{i,m_i}
//! $$
//!
#![doc = include_str!("../js/mathjax.tag")]

use crate::domain::{AnyConicDomain, IntoConicDomain,IntoShapedDomain};
use crate::expr::workstack::WorkStack;
use crate::expr::{ExprTrait, IntoExpr};

/// Trait represeting one or more constraint blocks, i.e.
///
/// $$
/// \\left[
///     \\begin{array}{l}
///         A_1x+b_1 \\in K_1\\
///         \\vdots  \\
///         A_kx+b_k \\in K_k
///     \\end{}
/// \\right]
/// $$
pub trait ConjunctionTrait 
{
    /// Evaluate conjunction expression and pushes them on the stack in order.
    ///
    /// # Arguments
    /// - `rs`, `ws`, `xs` Work stacks. The results are pushed on the `rs` stack.
    /// # Returns
    /// On success, returns the number of expressions pushed on the stack.
    fn eval(&mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String>;

    /// Create an object that hides the underlying object types. This allows putting objects of
    /// different types into a heterogenous container like a `Vec`.
    fn dynamic(self) -> Box<dyn ConjunctionTrait> where Self:Sized+'static { Box::new(self) }
}
/// Trait representing a disjunction of two or more conjunction blocks:
///
/// $$
/// \\begin{array}{l}
/// \\left[
///     \\begin{array}
///         A_{1,1}x +b_{1,1} \\in K_{1,1} \\\\
///         \\vdots \\\\
///         A_{1,k_1}x +b_{1,k_1} \\in K_{1,k_1}
///     \\end{array}
/// \\right]
/// \\
/// \\mathrm{or}\\\\
/// \\vdots\\\\
/// \\mathrm{or}\\\\
/// \\left[
///     \\begin{array}
///         A_{n,1}x +b_{n,1} \\in K_{n,1} \\\\
///         \\vdots \\\\
///         A_{n,k_n}x +b_{n,k_1} \\in K_{n,k_n}
///     \\end{array}
/// \\right]
/// \\end{array}
/// $$
pub trait DisjunctionTrait
{
    /// Evaluate conjunction expression and pushes them on the stack in order.
    ///
    /// # Arguments
    /// - `rs`, `ws`, `xs` Work stacks. The results are pushed on the `rs` stack.
    /// # Returns
    /// On success, returns the number of expressions pushed on the stack.
    fn eval(& mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            term_size : & mut Vec<usize>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<(),String>;
    /// Create an object that hides the underlying object types. This allows putting objects of
    /// different types into a heterogenous container like a `Vec`.
    fn dynamic(self) -> Box<dyn DisjunctionTrait> where Self: Sized+'static { Box::new(self) }
}

/// A single constraint block `A x + b ∈ D`.
pub struct AffineConstraint<const N : usize,E,D> 
    where E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>,
{
    expr   : E,
    domain : Option<D>,
}

/// Create a structure encapsulating an expression and a domain for constructing disjunctive
/// constraints.
///
/// # Arguments
/// - `expr` Something that can be turned into an expression via [IntoExpr]. 
/// - `domain` Something that can be turned into a [ConicDomain]. Currently this is any linear or
///    conic domain, but not semidefinite domain.
pub fn constr<const N : usize,I,E,D>(expr : I, domain : D) -> AffineConstraint<N,E,D>
    where I : IntoExpr<N,Result=E>,
          E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>,
{
    AffineConstraint{
        expr   : expr.into_expr(),
        domain : Some(domain),
    }
}





impl ConjunctionTrait for Box<dyn ConjunctionTrait> {
    fn dynamic(self) -> Box<dyn ConjunctionTrait> {
        self
    }
    fn eval(&mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        self.as_mut().eval(domains, rs, ws, xs)
    }
}


impl DisjunctionTrait for Box<dyn DisjunctionTrait> {
    fn eval(&mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            term_size : & mut Vec<usize>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<(),String> {
        self.as_mut().eval(domains,term_size, rs, ws, xs)
    }
    fn dynamic(self) -> Box<dyn DisjunctionTrait> {
        self
    }
}


impl<const N : usize,E,D> AffineConstraint<N,E,D>
    where E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>
{
    pub fn and<C2>(self, other : C2) -> AffineConstraintsAnd<Self,C2> where C2 : ConjunctionTrait {
        AffineConstraintsAnd { c0: self, c1: other }
    }

    pub fn or<D2>(self, other : D2) -> DisjunctionOr<Self,D2> where D2 : DisjunctionTrait {
        DisjunctionOr { c0: self, c1: other }
    }
}

impl<const N : usize,E,D> ConjunctionTrait for AffineConstraint<N,E,D> 
    where E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>
{
    fn eval(& mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> 
    {
        self.expr.eval_finalize(rs,ws,xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,_,_,_,_) = rs.peek_expr();
        if eshape.len() != N { return Err(format!("Evaluated expression dimension {} does not match domain {}",eshape.len(),N)); }
        let mut shape = [0usize;N]; shape.copy_from_slice(eshape);

        let dom = self.domain.take().unwrap().try_into_domain(shape)?.into_conic();

        let (_,_,dshape,_,is_integer) = dom.get();

        if dshape != eshape {
            return Err("Mismatching expression and domain shapes in constraint".to_string());
        }


        domains.push(Box::new(dom));

        if is_integer {
            Err(format!("Constraint domains cannot be integer"))
        }
        else {
            Ok(1)
        }
    }
}

impl<C> DisjunctionTrait for C
    where C : ConjunctionTrait,
{
    fn eval(& mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            term_size : & mut Vec<usize>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<(),String> {
        term_size.push(ConjunctionTrait::eval(self,domains,rs,ws,xs)?);
        Ok(())
    }
}

impl<C> DisjunctionTrait for Vec<C> where C : ConjunctionTrait {
    fn eval(&mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            term_size : & mut Vec<usize>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<(),String> {

        for c in self.iter_mut() {
            term_size.push(c.eval(domains, rs, ws, xs)?);
        }
        Ok(())
    }
}

//-----------------------------------------------------------------------------
// AffineConstraintsAnd

/// Represents the construction `A_1x+b_1 ∈ K_1 AND A_2x+b_2 ∈ K_2`.
pub struct AffineConstraintsAnd<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : ConjunctionTrait
{
     c0 : C0,
     c1 : C1
}

impl<C0,C1> AffineConstraintsAnd<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : ConjunctionTrait
{
    pub fn and<C2>(self, other : C2) -> AffineConstraintsAnd<Self,C2> where C2 : ConjunctionTrait {
        AffineConstraintsAnd { c0: self, c1: other }
    }
    pub fn or<D2>(self, other : D2) -> DisjunctionOr<Self,D2> where D2 : DisjunctionTrait {
        DisjunctionOr { c0: self, c1: other }
    }
}
impl<C0,C1> ConjunctionTrait for AffineConstraintsAnd<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : ConjunctionTrait
{
    fn eval(& mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        Ok(self.c0.eval(domains,rs,ws,xs)? + self.c1.eval(domains,rs,ws,xs)?)
    }
}


//-----------------------------------------------------------------------------
// DisjunctionOr

/// Represents the construction `A OR B` where `A` is set of affine constraints and `B` is another
/// disjunction clause
pub struct DisjunctionOr<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : DisjunctionTrait
{
     c0 : C0,
     c1 : C1
}

impl<C0,C1> DisjunctionOr<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : DisjunctionTrait
{
    pub fn or<C2>(self, other : C2) -> DisjunctionOr<C2,Self> where C2 : ConjunctionTrait {
        DisjunctionOr { c0: other, c1: self }
    }
}

impl<C0,C1> DisjunctionTrait for DisjunctionOr<C0,C1> 
    where 
        C0 : ConjunctionTrait,
        C1 : DisjunctionTrait
{
    fn eval(& mut self,
            domains : & mut Vec<Box<dyn AnyConicDomain>>,
            term_size : & mut Vec<usize>,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<(),String> {
        term_size.push(self.c0.eval(domains, rs, ws, xs)?);

        self.c1.eval(domains,term_size, rs, ws, xs)
    }
}
