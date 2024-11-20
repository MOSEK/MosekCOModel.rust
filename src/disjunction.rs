use super::*;


/// Trait that can be converted to a conic constraint.
pub trait ConicDomainTrait<const N : usize>  {
    fn add_disjunction_clause<E>(&self, model: & mut Model, e : &E) where E : ExprTrait<N>;
}

/// Defines a clause in a disjunction, i.e. either a single affine constraint `Ax+b ∈ K` or a
/// conjunction of multiple affine constraints.
pub trait ClauseTrait {
    /// Create the conjunction of two clauses, i.e. `A₁x+b₁ ∈ K₁ AND A₂x+b₂ ∈ K₂`
    fn and<const N2 : usize, E2, D2>(self,expr : E2, dom : D2) -> ClauseAndClause<Self,Clause<N2,E2,D2>> where E2 : ExprTrait<N2>, D2 : ConicDomainTrait<N2>, Self : Sized {
        ClauseAndClause{
            clause1 : self,
            clause2 : Clause{
                expr,
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

//pub struct DisjunctionList {
//    terms : Vec<Box<dyn DisjunctionTrait>>
//}

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

//impl DisjunctionTrait for DisjunctionList {
//    fn add_to_model(&self, model : & mut Model) {
//        for t in self.terms.iter() {
//            t.add_to_model(model);
//        }
//    }
//`}

pub fn term<const N : usize, E,D>(expr : E, dom : D) -> Clause<N,E,D> where E : ExprTrait<N>, D : ConicDomainTrait<N> {
    Clause{ expr, dom}
}
pub fn terms() -> ClauseList {
    ClauseList{ clauses : Vec::new() }
}

