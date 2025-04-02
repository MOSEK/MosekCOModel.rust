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
///         A_{1,1}x +b_{1,1} \in K_{1,1} \\
///         \\vdots \\
///
///         A_{1,k_1}x +b_{1,k_1} \in K_{1,k_1} \\
///     \\end{array}
/// \\right]
/// \\
/// \\mathrm{or}\\
/// \\vdots\\
/// \\mathrm{or}\\
/// \\left[
///     \\begin{array}
///         A_{n,1}x +b_{n,1} \in K_{n,1} \\
///         \\vdots \\
///
///         A_{n,k_n}x +b_{n,k_1} \in K_{n,k_n} \\
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




























/*

/// Trait that can be converted to a conic constraint.
pub trait ConicDomainTrait<const N : usize>  {
    fn add_disjunction_clause<E>(&self, model: & mut Model, e : &E) where E : ExprTrait<N>;
}

/// Defines a clause in a disjunction, i.e. either a single affine constraint `Ax+b ∈ K` or a
/// conjunction of multiple affine constraints.
pub trait ClauseTrait {
    /// Create the conjunction of two clauses, i.e. `A₁x+b₁ ∈ K₁ AND A₂x+b₂ ∈ K₂`
    fn and<const N2 : usize, E2, D2>(self,expr : E2, dom : D2) -> ClauseAndClause<Self,Clause<N2,E2::Result,D2>> 
        where 
            E2 : IntoExpr<N2>, 
            D2 : ConicDomainTrait<N2>, 
            Self : Sized 
    {
        ClauseAndClause{
            clause1 : self,
            clause2 : Clause{
                expr : expr.into(),
                dom }
        }
    }
    /// Add the clause to the model to the disjunction currently being built.
    fn add_to_model(&self, model : & mut Model);
}

/// Defines either a single clause a disjunction of multiple clauses.
pub trait DisjunctionTrait {
    /// Produce a disjunction from a disjunction and terms.
    fn or<T2>(self, term2 : T2) -> DisjunctionOrDisjunction<Self,T2> where T2 : DisjunctionTrait+Sized, Self : Sized {
        DisjunctionOrDisjunction{
            term1 : self,
            term2
        }
    }
    fn add_to_model(&self, model : & mut Model);
}

/// A clause in a disjunction.
pub struct Clause<const N : usize,E,D> 
    where 
        E : ExprTrait<N>, 
        D : ConicDomainTrait<N>  
{
    expr : E,
    dom  : D
}

pub struct ClauseAndClause<C1,C2> where C1 : ClauseTrait, C2 : ClauseTrait {
    clause1 : C1,
    clause2 : C2
}

/// Heterogen list of clauses 
pub struct ClauseList {
    clauses : Vec<Box<dyn ClauseTrait>>
}

pub struct DisjunctionOrDisjunction<T1,T2> where T1 : DisjunctionTrait, T2 : DisjunctionTrait {
    term1 : T1,
    term2 : T2
}

impl<const N : usize> ConicDomainTrait<N> for ConicDomain<N> {
    fn add_disjunction_clause<E>(&self, model: & mut Model, e : &E) where E : ExprTrait<N> {
        model.add_disjunction_clause(e,self);
    }
}

impl<const N : usize> ConicDomainTrait<N> for LinearDomain<N> {
    fn add_disjunction_clause<E>(&self, model: & mut Model, e : &E) where E : ExprTrait<N> {
        model.add_disjunction_clause(e,&self.to_conic());
    }
}

pub struct DisjunctionList {
    terms : Vec<Box<dyn DisjunctionTrait>>
}

impl<const N : usize,E> ClauseTrait for (E,ConicDomain<N>) where E : ExprTrait<N> {
    fn add_to_model(&self, model : & mut Model) {
        model.add_disjunction_clause(&self.0, &self.1);
    }
}
impl<const N : usize,E,D> ClauseTrait for Clause<N,E,D> where E : ExprTrait<N>, D : ConicDomainTrait<N> { 
    fn add_to_model(&self, model : & mut Model) {
        self.dom.add_disjunction_clause(model, &self.expr)
    }
}
impl<C1,C2> ClauseTrait for ClauseAndClause<C1,C2> where C1 : ClauseTrait, C2 : ClauseTrait { 
    fn add_to_model(&self, model : & mut Model) {
        self.clause1.add_to_model(model);
        self.clause2.add_to_model(model);
    }
}

impl ClauseList {
    pub fn append<C>(&mut self, c : C) -> & mut Self where C : 'static+ClauseTrait {
        self.clauses.push(Box::new(c));
        self
    }
}
impl ClauseTrait for ClauseList {
    fn add_to_model(&self, model : & mut Model) {
        for c in self.clauses.iter() {
            c.add_to_model(model);
        }
    }
}

impl ClauseTrait for [Box<dyn ClauseTrait>] {
    fn add_to_model(&self, model : & mut Model) {
        for c in self.iter() {
            c.add_to_model(model);
        }
    }
}

impl ClauseTrait for Vec<Box<dyn ClauseTrait>> {
    fn add_to_model(&self, model : & mut Model) {
        self.as_slice().add_to_model(model);
    }
}

impl<C> DisjunctionTrait for C where C : ClauseTrait {
    fn add_to_model(&self, model : & mut Model) {
        model.start_term();
        ClauseTrait::add_to_model(self,model);
        model.end_term();
    }
}

impl<T1,T2> DisjunctionTrait for DisjunctionOrDisjunction<T1,T2> where T1 : DisjunctionTrait, T2 : DisjunctionTrait {
    fn add_to_model(&self, model : & mut Model) {
        self.term1.add_to_model(model);
        self.term2.add_to_model(model);
    }
}

impl DisjunctionList {
    pub fn append<D>(&mut self, c : D) -> & mut Self where D : 'static+DisjunctionTrait{
        self.terms.push(Box::new(c));
        self
    }
}

impl DisjunctionTrait for DisjunctionList {
    fn add_to_model(&self, model : & mut Model) {
        for t in self.terms.iter() {
            t.add_to_model(model);
        }
    }
}

/// Construct single clause term for a disjunctive constraint.
///
/// This constructs a single clause term of the form 
/// $$
///    Ax+b\\in K
/// $$
#[doc = include_str!("../js/mathjax.tag")]
pub fn term<const N : usize, E,D>(expr : E, dom : D) -> Clause<N,E::Result,D> 
    where 
        E : IntoExpr<N>, 
        D : ConicDomainTrait<N> {
    Clause{ expr : expr.into(), dom}
}

///  
pub fn clauses() -> ClauseList {
    ClauseList{ clauses : Vec::new() }
}

pub fn terms() -> DisjunctionList {
    DisjunctionList{ terms : Vec::new() }
}


*/
