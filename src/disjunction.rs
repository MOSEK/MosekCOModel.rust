//! Structures and functions for formulating disjunctive constraints.
//!
//! A disjunctive constraint is a constraint of the form 
//! $$
//!   A_1x+b_1\\in K_1\\mathbb{ or }\\ldots A_nx+b_n\\in K_n
//! $$
//! Each \\(A_ix+b_i\\) is called a _term_ (see [term]). A term can be composed if multiple _clauses_:
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

use crate::{model::VarAtom, expr::workstack::WorkStack};


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
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String>;


    /// Add affine expressions and domains to the `task`. The already evaluated expressions are
    /// passed in `exprs` in reverse order (last evaluated expression at index 0)
    ///
    /// # Arguments
    /// - `task` Task to add clauses to
    /// - `vars` The mapping of variable indexes to underlying variable elements.
    /// - `exprs` The array of evaluated expressions
    /// - `element_dom` Vector of domains added to the task. `element_dom[i]` is the index of the
    ///    domain if constraint block `i`.
    /// - `element_afei` Vector of AFE indexes from the underlying task
    /// - `element_ptr` Vector of pointers to consraint blocks. `element_ptr[i]` points to the
    ///   first item in `element_afei` of constraint block `i`.
    fn append_clause_data(&self, 
                          task  : & mut mosek::TaskCB,
                          vars  : &[VarAtom],
                          exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>,
                          
                          element_dom  : &mut Vec<i64>,
                          element_ptr  : &mut Vec<usize>,
                          element_afei : &mut Vec<i64>,
                          element_b    : &mut Vec<f64>);
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
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String>;
    /// Add affine expressions and domains to the `task`. The already evaluated expressions are
    /// passed in `exprs` in reverse order (last evaluated expression at index 0)
    ///
    /// # Arguments
    /// - `task` Task to add clauses to
    /// - `vars` The mapping of variable indexes to underlying variable elements.
    /// - `exprs` The array of evaluated expressions
    /// - `element_dom` Vector of domains added to the task. `element_dom[i]` is the index of the
    ///    domain if constraint block `i`.
    /// - `element_afei` Vector of AFE indexes from the underlying task
    /// - `element_ptr` Vector of pointers to consraint blocks. `element_ptr[i]` points to the
    ///   first item in `element_afei` of constraint block `i`.
    /// - `term_ptr` Pointers to disjunction clauses. `term_ptr[i]` is the index into `element_ptr`
    ///   and `element_dom` of the first constraint block of disjunction term `i`.
    fn append_disjunction_data(&self, 
                               task  : & mut mosek::TaskCB,
                               vars  : &[VarAtom],
                               exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>, 
                              
                               term_ptr     : &mut Vec<usize>,
                               element_dom  : &mut Vec<i64>,
                               element_ptr  : &mut Vec<usize>,
                               element_afei : &mut Vec<i64>,
                               element_b    : &mut Vec<f64>);
    /// Create an object that hides the underlying object types. This allows putting objects of
    /// different types into a heterogenous container like a `Vec`.
    fn dynamic(self) -> Box<dyn DisjunctionTrait> where Self: Sized+'static { Box::new(self) }
}

impl ConjunctionTrait for Box<dyn ConjunctionTrait> {
    fn dynamic(self) -> Box<dyn ConjunctionTrait> {
        self
    }
    fn eval(&mut self,
                numvarelm : usize,
                rs : &mut WorkStack,
                ws : &mut WorkStack,
                xs : &mut WorkStack) -> Result<usize,String> {
        self.as_mut().eval(numvarelm, rs, ws, xs)
    }
    fn append_clause_data(&self, 
                              task  : & mut mosek::TaskCB,
                              vars  : &[VarAtom],
                              exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>, 

                              element_dom  : &mut Vec<i64>,
                              element_ptr  : &mut Vec<usize>,
                              element_afei : &mut Vec<i64>,
                              element_b    : &mut Vec<f64>) {
        self.as_ref().append_clause_data(task, vars, exprs, element_dom, element_ptr, element_afei, element_b)
    }
}


impl DisjunctionTrait for Box<dyn DisjunctionTrait> {
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        self.as_mut().eval(numvarelm, rs, ws, xs)
    }
    fn dynamic(self) -> Box<dyn DisjunctionTrait> {
        self
    }
    fn append_disjunction_data(&self, 
                               task  : & mut mosek::TaskCB,
                               vars  : &[VarAtom],
                               exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>, 

                               term_ptr     : &mut Vec<usize>,
                               element_dom  : &mut Vec<i64>,
                               element_ptr  : &mut Vec<usize>,
                               element_afei : &mut Vec<i64>,
                               element_b    : &mut Vec<f64>) {
        (&self).as_ref().append_disjunction_data(task, vars, exprs, term_ptr, element_dom, element_ptr, element_afei, element_b)
    }
}

