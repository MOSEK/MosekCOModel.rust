use std::ops::ControlFlow;
use std::path::Path;

use model::Disjunction;

use crate::*;
use crate::domain::*;

// Defines the public interface for a Model
trait BaseModel {
    fn new(name : Option<&str>) -> Self;
    fn write_problem<P>(&self, filename : P) where P : AsRef<Path>;
    fn try_variable<I,D>(& mut self, name : Option<&str>, dom : I) -> Result<D::Result,String>
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait;
    fn variable<I,D>(& mut self, name : Option<&str>, dom : I) -> D::Result
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait
    {
        self.try_variable(name,dom).unwrap()
    }
    fn try_ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> Result<(Variable<N>,Variable<N>),String> 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>;
    fn ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> (Variable<N>,Variable<N>) 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>
    {
        self.try_ranged_variable(name,dom).unwrap()
    }
    
    fn try_ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> Result<(Constraint<N>,Constraint<N>),String> 
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>;
    fn ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> (Constraint<N>,Constraint<N>)
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>
    {
        self.try_ranged_constraint(name,expr,dom).unwrap()
    }

    fn try_constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Result<Constraint<N>,String>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>;
    
    fn constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Constraint<N>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>
    {
        self.try_constraint(name,expr,dom).unwrap()
    }

    fn try_disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Result<Disjunction,String> where D : disjunction::DisjunctionTrait;
    fn disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Disjunction where D : disjunction::DisjunctionTrait {
        self.try_disjunction(name, terms).unwrap()
    }

    fn update<const N : usize, E>(&mut self, item : &Constraint<N>, e : E) where E : expr::IntoExpr<N>;

    fn objective<E : expr::IntoExpr<0>>(& mut self,
                                             name  : Option<&str>,
                                             sense : Sense,
                                             expr  : E);
    fn set_parameter<T>(& mut self, parname : &str, parval : T) 
        where T : SolverParameterValue;

    fn set_double_parameter(&mut self, parname : &str, parval : f64);
    fn set_int_parameter(&mut self, parname : &str, parval : i32);
    fn set_str_parameter(&mut self, parname : &str, parval : &str);
    fn put_optserver(&mut self, hostname : &str, access_token : Option<&str>);
    fn clear_optserver(&mut self);
    fn solve(& mut self);
    fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus);
    fn primal_objective(&self, solid : SolutionType) -> Option<f64>;
    fn primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String>;
    fn sparse_primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String>; 
    fn dual_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String>;
    fn primal_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String>;
    fn dual_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String>;
    fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String>;
}
trait ModelWithCallbacks {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str);
    fn set_solution_callback<F>(&mut self, func : F) where F : 'static+FnMut(&Model); 
    fn set_callback<F>(&mut self, func : F) where F : 'static+FnMut() -> ControlFlow<(),()>;
}
