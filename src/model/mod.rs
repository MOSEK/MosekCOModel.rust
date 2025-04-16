//!
use itertools::{merge_join_by, EitherOrBoth};
#[doc = include_str!("../js/mathjax.tag")]

use itertools::{iproduct, izip};
use std::fmt::Debug;
use std::ops::ControlFlow;
use std::{iter::once, path::Path};
use crate::disjunction::ConjunctionTrait;
use crate::{disjunction, expr, IntoExpr, ExprTrait, NDArray, DisjunctionTrait};
use crate::utils::iter::*;
use crate::utils::*;
use crate::domain::*;
use crate::variable::*;
use crate::WorkStack;
use crate::constraint::*;

pub mod mosekmodel;

/// Objective sense
#[derive(Clone,Copy)]
pub enum Sense {
    Maximize,
    Minimize
}

#[derive(Clone,Copy,Debug)]
pub enum WhichLinearBound {
    Lower,
    Upper,
    Both
}

#[derive(Clone,Copy,Debug)]
pub enum VarAtom {
    // Task variable index
    Linear(i32, WhichLinearBound), // (vari,which bound)
    // Task bar element (barj,k,l)
    BarElm(i32,usize),
    // Conic variable (j,offset)
    ConicElm(i32,usize)
}
#[allow(unused)]
#[derive(Clone,Copy,Debug)]
enum ConAtom {
    // Conic constraint element (acci, offset)
    ConicElm{acci : i64, afei : i64, accoffset : usize},
    BarElm{acci : i64, accoffset : i64, afei : i64, barj : i32, offset : usize},
    Linear(i32,f64,i32,WhichLinearBound) // (coni, rhs,bk, which bound)
}



/*************************************************************************************************
 *
 * Solution structs and enums
 *
 *************************************************************************************************/


/// Solution type selector
#[derive(Clone,Copy)]
pub enum SolutionType {
    /// Default indicates to automatically select which solution to use, in order of priority:
    /// `Integer`, `Basic`, `Interior`
    Default,
    /// Basic solution, the result of the simplex solver or basis identification.
    Basic,
    /// Interior solution, the result of the interior point solver.
    Interior,
    /// Integer solution, the only solution available for integer problems.
    Integer
}

/// Solution status indicator. It is used to indicate the status of either the primal or the dual
/// part of a solution.
#[derive(Clone,Copy,Debug)]
pub enum SolutionStatus {
    /// Indicates that the solution is optimal within tolerances.
    Optimal,
    /// Indicates that the solution is feasible within tolerances.
    Feasible,
    /// Indicates that the solution is a certificate of either primal or dual infeasibility. A
    /// primal certificate prooves dual infesibility, and a dual certificate indicates primal
    /// infeasibility.
    CertInfeas,
    /// Indicates that the solution is a certificate of either primal or dual illposedness. A
    /// primal certificate prooves dual illposedness, and a dual certificate indicates primal
    /// illposedness.
    CertIllposed,
    /// Indicates that the solution status is not known, basically it can be arbitrary values. 
    Unknown,
    /// Indicates that the solution is not available.
    Undefined
}

#[derive(Default)]
struct SolutionPart {
    status : SolutionStatus,
    var    : Vec<f64>,
    con    : Vec<f64>,
    obj    : f64,

}

impl SolutionPart {
    fn new(numvar : usize, numcon : usize) -> SolutionPart { SolutionPart{status : SolutionStatus::Unknown, var : vec![0.0; numvar], con : vec![0.0; numcon], obj : 0.0} }
    fn resize(& mut self,numvar : usize, numcon : usize) {
        self.var.resize(numvar, 0.0);
        self.con.resize(numcon, 0.0);
    }
}

#[derive(Default)]
struct Solution {
    primal : SolutionPart,
    dual   : SolutionPart
}

impl Solution {
    fn new() -> Solution { Solution{primal : SolutionPart::new(0,0) , dual : SolutionPart::new(0,0)  } }
}


/*************************************************************************************************
 *
 * VarDomainTrait
 *
 *************************************************************************************************/

/// Represents something that can be used as a domain for a variable.
pub trait VarDomainTrait<M> 
{
    type Result; 
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String>;
}

impl<const N : usize,M> VarDomainTrait<M> for ConicDomain<N> where M : ConicModelTrait 
{
    type Result = Variable<N>;
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.conic_variable(name,self)
    }
}

impl<const N : usize,M> VarDomainTrait<M> for &[usize;N] where M : BaseModelTrait 
{
    type Result = Variable<N>;
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.free_variable(name,self)
    }
}

impl<const N : usize,M> VarDomainTrait<M> for LinearDomain<N> where M : BaseModelTrait 
{
    type Result = Variable<N>;
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.linear_variable::<N,Self::Result>(name,self)
    }
}

impl<M> VarDomainTrait<M> for usize where M : BaseModelTrait 
{
    type Result = Variable<1>;
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.free_variable(name,&[self])
    }
}

impl<const N : usize,M> VarDomainTrait<M> for LinearRangeDomain<N> where M : BaseModelTrait 
{
    type Result = (Variable<N>,Variable<N>);
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.ranged_variable::<N,Self::Result>(name,self)
    }
}

impl<const N : usize, M> VarDomainTrait<M> for PSDDomain<N> where M : PSDModelTrait 
{
    type Result = Variable<N>;
    fn create(self, m : & mut M, name : Option<&str>) -> Result<Self::Result,String> {
        m.psd_variable(name,self)
    }
}





//======================================================
// Model
//======================================================

pub trait BaseModelTrait {
    fn free_variable<const N : usize>
        (&mut self,
         name : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized;

    fn linear_variable<const N : usize,R>
        (&mut self, 
         name : Option<&str>,
         dom : LinearDomain<N>) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized;
    fn linear_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearDomain<N>) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
        where 
            Self : Sized;
    fn ranged_variable<const N : usize,R>(&mut self, name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized;
    fn ranged_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearRangeDomain<N>,shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<<LinearRangeDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
        where 
            Self : Sized;

    fn update(& mut self, idxs : &[usize]) -> Result<(),String>;

    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>;

    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, solitg : &mut Solution) -> Result<(),String>;
}

/// An inner model object must implement this to support conic vector constraints and variables
pub trait ConicModelTrait {
    fn conic_variable<const N : usize>(&mut self, name : Option<&str>,dom : ConicDomain<N>) -> Result<Variable<N>,String>;
    fn conic_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : ConicDomain<N>,shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<Constraint<N>,String>;
}

/// An inner model object must implement this to support PSD variables and constraints
pub trait PSDModelTrait {
    fn psd_variable<const N : usize>(&mut self, name : Option<&str>, dom : PSDDomain<N>) -> Result<Variable<N>,String>;
    fn psd_constraint<const N : usize>(& mut self, name : Option<&str>, dom : PSDDomain<N>,shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<Constraint<N>,String>;
}

/// An inner model object must implement this to support disjunctive constraints
pub trait DJCModelTrait {
    fn disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Result<Disjunction,String> where D : disjunction::DisjunctionTrait;
}

/// An inner model object must implement this to support log callbacks.
pub trait ModelWithLogCallback {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str);
}

/// An inner model object must implement this to support integer solution callbacks
pub trait ModelWithSolutionCallback {
    fn set_solution_callback<F>(&mut self, func : F) where F : 'static+FnMut(&ModelAPI<Self>);
}

/// An inner model object must implement this to support control callbacks (callbacks that allow us
/// to ask the solver to terminate nicely).
pub trait ModelWithControlCallback {
    fn set_callback<F>(&mut self, func : F) where F : 'static+FnMut() -> ControlFlow<(),()>;
}


pub struct ModelAPI<T> where T : BaseModelTrait {
    inner : T,

    /// Basis solution
    sol_bas : Solution,
    /// Interior solution
    sol_itr : Solution,
    /// Integer solution
    sol_itg : Solution,


    rs : WorkStack,
    ws : WorkStack,
    xs : WorkStack
}

impl<T> ModelAPI<T> where T : BaseModelTrait {
    pub fn new() -> ModelAPI<T> where T : Default {
        ModelAPI{
            inner : Default::default(),
            rs : WorkStack::new(1024),
            ws : WorkStack::new(1024),
            xs : WorkStack::new(1024),

            sol_bas : Default::default(),
            sol_itr : Default::default(),
            sol_itg : Default::default(),
        }
    }

    pub fn from(inner : T) -> ModelAPI<T> { 
        ModelAPI {
            inner,
            rs : WorkStack::new(1024),
            ws : WorkStack::new(1024),
            xs : WorkStack::new(1024),
            
            sol_bas : Default::default(),
            sol_itr : Default::default(),
            sol_itg : Default::default(),
        }
    }


    /// Attach a log printer callback to the Model. This will receive messages from the solver
    /// while solving and during a few other calls like file reading/writing. 
    ///
    /// # Arguments
    /// - `func` A function that will be called with strings from the log. Individual lines may be
    ///   written in multiple chunks to there is no guarantee that the strings will end with a
    ///   newline.
    pub fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str), T : ModelWithLogCallback {
        self.inner.set_log_handler(func);
    }

    /// Attach a solution callback function. This is called for each new integer solution 
    pub fn set_int_solution_callback<F>(&mut self, mut func : F) where F : 'static+FnMut(&mut Self), T : ModelWithSolutionCallback {
        // NOTE: We cheat here. 
        // 1. We pass self as a pointer to bypass the whole lifetime issue. This
        // is acceptable because we KNOW self will outlive the underlying Task.
        // 2. The constuction is acceptable because we know that the callback is called from inside
        //    the `.solve()` call, which holds a mutable reference to model.
        let modelp :  *const Self = self;
        
        self.inner.set_int_solution_callback(move |&mut Solution| {
            let model : & mut MosekModel = unsafe { & mut (* (modelp as * mut MosekModel)) };

            model.sol_itg.primal.status = SolutionStatus::Feasible;
            model.sol_itg.dual.status = SolutionStatus::Undefined;
            if model.sol_itg.primal.var.len() != model.vars.len() {
                model.sol_itg.primal.var = vec![0.0; model.vars.len()];
            }
            if model.sol_itg.primal.con.len() != model.cons.len() {
                model.sol_itg.primal.con = vec![0.0; model.cons.len()];
            }

            for (s,v) in model.sol_itr.primal.var.iter_mut().zip(model.vars.iter()) {
                match v {
                    VarAtom::Linear(i,_) => if (*i as usize) < xx.len() { unsafe{ *s = *xx.get_unchecked(*i as usize) } },
                    _ => {}
                }
            }

            func(model);
        }).unwrap();
    }

    pub fn set_callback<F>(&mut self, mut func : F) where F : 'static+FnMut() -> ControlFlow<(),()> {
        self.task.put_codecallback(move |_code| {
            match func() {
                ControlFlow::Break(_) => false,
                ControlFlow::Continue(_) => true,
            }
        }).unwrap();
    }







    /// Add a Variable.
    ///
    ///
    /// If the domain defines a sparsity pattern, elements outside of the sparsity pattern are treated as
    /// fixed to 0.0. For example, for
    ///  ```rust
    ///  use mosekcomodel::*;
    ///  let dom = greater_than(vec![1.0,1.0,1.0]).with_shape_and_sparsity(&[6],&[[0],[2],[4]]);
    ///  ```
    ///  `dom` would define a variable of length 6 where element 0, 2 and 4 are greater than 1.0,
    ///  while elements 1,3,5 are fixed to 0.0.
    ///
    ///  The domain is required to define the shape in some meaningful way. For example,
    ///  - [LinearProtoDomain], [ConicProtoDomain], [PSDProtoDomain] for example from [zeros],
    ///   `greater_than(vec![1.0,1.0])` will produce a variable of the shape defined by the domain.
    ///  - [ScalableLinearDomain] as produced by for example [zero], [unbounded] or `greater_than(1.0)` the result is a scalar variable.
    ///  - [ScalableConicDomain], [ScalablePSDDomain] will fail as they define no meaningful shape.
    ///
    /// # Arguments
    /// - `name` Optional constraint name. This is currently only used to generate names passed to
    ///   the underlying task.
    /// - `dom` The domain of the variable. This defines the bound
    ///   type, shape and sparsity of the variable. For sparse
    ///   variables, elements outside of the sparsity pattern are
    ///   treated as variables fixed to 0.0.
    /// # Returns
    /// - On success, return an `N`-dimensional variable object is returned. The `Variable` object
    ///   may be dense or sparse, where "sparse" means that all entries outside the sparsity
    ///   pattern are fixed to 0.
    /// - On a recoverable failure (i.e. when the [Model] is in a consistent state), return a
    ///   string describing the error.
    /// - On non-recoverable errors: Panic.
    pub fn try_variable<I,D,R>(& mut self, name : Option<&str>, dom : I) -> Result<R,String>
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait<Self,Result = R>,
            Self : Sized
    {
        dom.try_into_domain()?.create(self.inner,name)
    }

    /// Add a Variable. See [Model::try_variable].
    ///
    /// # Returns
    /// An `N`-dimensional variable object is returned. The `Variable` object may be dense or
    /// sparse, where "sparse" means that all entries outside the sparsity pattern are fixed to 0.
    ///
    /// Panics on any error.
    pub fn variable<I,D,R>(& mut self, name : Option<&str>, dom : I) -> R
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait<Self,Result = R>,
            Self : Sized
    {
        self.try_variable(name,dom).unwrap()
    }

    /// Add a constraint
    ///
    /// Note that even if the domain or the expression are sparse, a constraint will always be
    /// full, and all elements outside of the sparsity pattern are intereted as zeros. Unlike for
    /// variables, an entry in the domain outside the sparsity pattern will NOT cause the
    /// corresponding expression element to be fixed to 0.0. So, for example in
    ///  ```rust
    ///  use mosekcomodel::*;
    ///  let mut m = Model::new(None);
    ///  let x = m.variable(None, unbounded().with_shape(&[3]));
    ///  let c1 = m.constraint(None, &x,greater_than(vec![1.0,1.0]).with_shape_and_sparsity(&[3],&[[0],[2]]));
    ///  let c2 = m.constraint(None, &x,greater_than(vec![1.0,0.0,1.0]));
    ///  ```
    ///  The constraints `c1` and `c2` mean exactly the same.
    ///
    ///  The domain is checked or expanded according to the shape of the `expr` argument. For
    ///  example:
    ///  - [LinearProtoDomain], [ConicProtoDomain], [PSDProtoDomain] for example from [zeros],
    ///   `greater_than(vec![1.0,1.0])`, the expression shape must exactly match the domain shape.
    ///  - [ScalableLinearDomain], [ScalableConicDomain] will be expanded to match the shape of the
    ///    expression, and it will be checked that the shape is valid for the domain type - like
    ///    that an exponential cone has size 3. For conic domains, the cones are by default in the inner-most
    ///    dimension, so if  the expression shape is `[2,3,4]`, the cones in the domain will have
    ///    size 4. This can be changed with the `::cone_dim` method.
    ///  - [ScalablePSDDomain] will be expanded to match the domain, and as above, the cones are by
    ///    default placed in the inner-most dimensions, so if the expression shape is `[2,3,4,4]` the PSD
    ///    cones will have dimension 4. If the two cone dimensions to not have the same size, it
    ///    will cause an error.
    /// # Arguments
    /// - `name` Optional constraint name. Currently this is only used to generate names passed to
    ///   the underlting task.
    /// - `expr` Constraint expression. Note that the shape of the expression and the domain must match exactly.
    /// - `dom`  The domain of the constraint. This defines the bound type and shape.
    /// # Returns
    /// - On success, return a N-dimensional constraint object that can be used to access
    ///   solution values.
    /// - On any recoverable failure, i.e. failure where the [Model] is in a consistent state:
    ///   Return a string describing the error.
    /// - On any non-recoverable error: Panic.
    pub fn try_constraint<const N : usize,E,I,D>(& mut self, name : Option<&str>, expr :  E, dom : I) -> Result<D::Result,String>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            I : IntoShapedDomain<N,Result=D>,
            D : ConstraintDomain<N,Self>,
            Self : Sized
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        dom.try_into_domain(shape)?.add_constraint(self.inner,name,eshape,ptr,subj,cof)
    }

    /// Add a constraint. See [Model::try_constraint].
    ///
    /// # Returns
    /// - On success, return a N-dimensional constraint object that can be used to access
    ///   solution values.
    /// - On any failure: Panic.
    pub fn constraint<const N : usize,E,I,D>(& mut self, name : Option<&str>, expr :  E, dom : I) -> D::Result
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            I : IntoShapedDomain<N,Result = D>,
            D : ConstraintDomain<N,Self>,
            Self : Sized
    {
        self.try_constraint(name, expr, dom).unwrap()
    }
    
    /// Update the expression of a constraint in the Model.
    pub fn try_update<const N : usize, E>(&mut self, item : &Constraint<N>, expr : E) -> Result<(),String>
        where 
            E    : expr::IntoExpr<N>
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        self.inner.update(item.idxs.as_slice(),shape,ptr,subj,cof)
    }
    
    pub fn update<const N : usize, E>(&mut self, item : &Constraint<N>, e : E)
        where             
            E    : expr::IntoExpr<N>
    {
        self.try_update(item,e).unwrap()
    }
    
    /// Write problem to a file. The file is written by the underlying solver task, so no
    /// structural information will be written.
    ///
    /// # Arguments
    /// - `filename` The filename extension determines the file format to use. If the
    ///   file extension is not recognized, the MPS format is used.
    ///
    pub fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        self.inner.write_problem(filename)
    }

    /// Solve the problem and extract the solution.
    ///
    /// This will fail if the optimizer fails with an error. Not producing a solution (stalling or
    /// otherwise failing), producing a non-optimal solution or a certificate of infeasibility 
    /// is *not* an error.
    pub fn solve(& mut self) -> Result<(),String> {
        self.sol_bas.primal.status = SolutionStatus::Undefined;
        self.sol_bas.dual.status = SolutionStatus::Undefined;
        self.sol_itr.primal.status = SolutionStatus::Undefined;
        self.sol_itr.dual.status = SolutionStatus::Undefined;
        self.sol_itg.primal.status = SolutionStatus::Undefined;
        self.sol_itg.dual.status = SolutionStatus::Undefined;

        self.inner.solve(&mut self.sol_itr, &mut self.sol_bas, & mut self.sol_itg)
    }

    /// Get solution status for the given solution.
    ///
    /// Solution status is returned as a pair: Status of the primal solution and status of the dual
    /// solution. In some cases, only part of the solution is available and meaningful:
    /// - An integer problem does not have any dual information
    /// - A problem that is primally infeasible will return a dual ray and have no meaningful
    ///   primal solution values.
    /// - A problem that is dual infeasible will return a primal ray and have no meaningful dual
    ///  solution information.
    ///
    /// # Arguments
    /// - `solid` Which solution to request status for.
    ///
    /// # Returns
    /// - `(psolsta,dsolsta)` Primal and dual solution status.
    pub fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus) {
        if let Some(sol) = self.select_sol(solid) {
            (sol.primal.status,sol.dual.status)
        }
        else {
            (SolutionStatus::Undefined,SolutionStatus::Undefined)
        }
    }

    /// Get primal objective value, if available.
    ///
    /// The primal objective is only available if the primal solution is defined.
    pub fn primal_objective_value(&self, solid : SolutionType) -> Option<f64> {
        if let Some(sol) = self.select_sol(solid) {
            Some(sol.primal.obj)
        }
        else {
            None
        }
    }
    
    /// Get dual objective value, if available.
    ///
    /// The dual objective is only available if the dual solution is defined.
    pub fn dual_objective_value(&self, solid : SolutionType) -> Option<f64> {
        if let Some(sol) = self.select_sol(solid) {
            Some(sol.dual.obj)
        }
        else {
            None
        }
    }

    /// Get primal solution values for an a variable or constraint.
    ///
    /// # Arguments
    /// - `solid` Choose which solution to ask for if multiple are available.
    /// - `item` The constraint or variable for which the solution values are wanted. If the item
    /// is a sparse variable, the result is filled out with zeros where necessary.
    ///
    /// # Returns
    /// If solution item is defined, return the solution, otherwise an error message.
    pub fn primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> where Self : Sized+BaseModelTrait {
        item.primal(self,solid)
    }
    
    /// Get primal solution values for an a sparse variable or constraint.
    ///
    /// # Arguments
    /// - `solid` Choose which solution to ask for if multiple are available.
    /// - `item` The constraint or variable for which the solution values are wanted.
    ///
    /// # Returns
    /// - `Err(msg)` is returned if the requested solution is not available 
    /// - `Some((vals,idxs))` is returned, where 
    ///   - `vals` are the solution values for non-zero entries
    ///   - `idxs` are the indexes.
    pub fn sparse_primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String> where Self : Sized+BaseModelTrait {
        item.sparse_primal(self,solid)
    }

    /// Get dual solution values for an item
    ///
    /// Returns: If solution item is defined, return the solution, otherwise a n error message.
    pub fn dual_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> where Self : Sized+BaseModelTrait {
        item.dual(self,solid)
    }

    /// Get primal solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn primal_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> where Self : Sized+BaseModelTrait {
        item.primal_into(self,solid,res)
    }

    /// Get dual solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn dual_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> where Self : Sized+BaseModelTrait {
        item.primal_into(self,solid,res)
    }

    
    /// Evaluate an expression in the (primal) solution. 
    ///
    /// # Arguments
    /// - `solid` The solution in which to evaluate the expression.
    /// - `expr` The expression to evaluate.
    pub fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String> where E : IntoExpr<N>, Self : Sized+BaseModelTrait {
        expr.into_expr().eval(& mut self.rs,&mut self.ws,&mut self.xs).map_err(|e| format!("{:?}",e))?;
        let mut shape = [0usize; N];
        let (val,sp) = self.evaluate_primal_internal(solid, &mut shape)?;
        self.rs.clear();
        NDArray::new(shape,sp,val)
    }















    

    fn select_sol(&self, solid : SolutionType) -> Option<&Solution> {
        match solid {
            SolutionType::Basic    => Some(&self.sol_bas),
            SolutionType::Interior => Some(&self.sol_itr),
            SolutionType::Integer  => Some(&self.sol_itg),
            SolutionType::Default  => {
                match self.sol_itg.primal.status {
                    SolutionStatus::Undefined =>
                        match (self.sol_bas.primal.status,self.sol_bas.dual.status) {
                            (SolutionStatus::Undefined,SolutionStatus::Undefined) =>
                                match (self.sol_itr.primal.status,self.sol_itr.dual.status) {
                                    (SolutionStatus::Undefined,SolutionStatus::Undefined) => None,
                                    _ => Some(&self.sol_itr)
                                }
                            _ => Some(&self.sol_bas)
                        },
                    _ => Some(& self.sol_itg)
                }
            }
        }
    }

    fn evaluate_primal_internal(&self, solid : SolutionType, resshape : & mut [usize]) -> Result<(Vec<f64>,Option<Vec<usize>>),String> {
        let (shape,ptr,sp,subj,cof) = {
            self.rs.peek_expr()
        };
        let sol = 
            if let Some(sol) = self.select_sol(solid) {
                if let SolutionStatus::Undefined = sol.primal.status {
                    return Err("Solution part is not defined".to_string())
                }
                else {
                    sol.primal.var.as_slice()
                }
            }
            else {
                return Err("Solution value is undefined".to_string());
            };

        if let Some(v) = subj.iter().max() { if *v >= sol.len() { return Err("Invalid expression: Index out of bounds for this solution".to_string()) } }
        let mut res = vec![0.0; ptr.len()-1];
        izip!(res.iter_mut(),subj.chunks_ptr2(ptr,&ptr[1..]),cof.chunks_ptr2(ptr,&ptr[1..]))
    .for_each(|(r,subj,cof)| *r = subj.iter().zip(cof.iter()).map(|(&j,&c)| c * unsafe{ *sol.get_unchecked(j) } ).sum() );
        resshape.copy_from_slice(shape);

        Ok(( res,sp.map(|v| v.to_vec()) ))
    }

}

















//======================================================
// ModelItem
//======================================================

/// The `ModelItem` represents either a variable or a constraint belonging to a [Model]. It is used
/// by the [Model] object when accessing solution assist overloading and determine which solution part to access.
pub trait ModelItem<const N : usize,M>  {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn shape(&self) -> [usize;N];
    //fn numnonzeros(&self) -> usize;
    fn sparse_primal(&self,m : &ModelAPI<M>,solid : SolutionType) -> Result<(Vec<f64>,Vec<[usize;N]>),String> {
        let res = self.primal(m,solid)?;
        let dflt = [0; N];
        let mut idx = vec![dflt; res.len()];
        let mut strides = [0; N];
        _ = strides.iter_mut().zip(self.shape().iter()).rev().fold(1,|c,(s,&d)| { *s = c; *s * d} );
        for (i,ix) in idx.iter_mut().enumerate() {
            let _ = strides.iter().zip(ix.iter_mut()).fold(i, |i,(&s,ix)| { *ix = i / s; i % s } );
        }
        Ok((res,idx))
    }
    fn primal(&self,m : &ModelAPI<M>,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.primal_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn dual(&self,m : &ModelAPI<M>,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.dual_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn primal_into(&self,m : &ModelAPI<M>,solid : SolutionType, res : & mut [f64]) -> Result<usize,String>;
    fn dual_into(&self,m : &ModelAPI<M>,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String>;
}

//======================================================
// Variable and Constraint
//======================================================

/// Support trait for Constraint::index, Variable::index and ExprTrait::index
pub trait ModelItemIndex<T> {
    type Output;
    fn index(self,obj : &T) -> Self::Output;
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct Disjunction {
    index : i64
}

impl<const N : usize,M> ModelItem<N,M> for Constraint <N> where M : BaseModelTrait {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn shape(&self) -> [usize; N] { self.shape }
    fn primal_into(&self,m : &ModelAPI<M>,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &ModelAPI<M>,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
}

pub trait SolverParameterValue<M : BaseModelTrait> {
    fn set(self,parname : &str, model : & mut ModelAPI<M>);
}

impl<M : BaseModelTrait> SolverParameterValue<M> for f64 {
    fn set(self, parname : &str,model : & mut ModelAPI<M>) { model.set_double_parameter(parname,self) }
}

impl<M : BaseModelTrait> SolverParameterValue<M> for i32 {
    fn set(self, parname : &str,model : & mut ModelAPI<M>) { model.set_int_parameter(parname,self) }
}

impl<M : BaseModelTrait> SolverParameterValue<M> for &str {
    fn set(self, parname : &str,model : & mut ModelAPI<M>) { model.set_str_parameter(parname,self) }
}

//======================================================
// Model
//======================================================




//impl ModelAPI for MosekModel {
//
//
//    fn dual_objective_value(&self, solid : SolutionType) -> Option<f64> {
//        if let Some(sol) = self.select_sol(solid) {
//            Some(sol.dual.obj)
//        }
//        else {
//            None
//        }
//    }
//
//    fn primal_objective_value(&self, solid : SolutionType) -> Option<f64> { 
//        if let Some(sol) = self.select_sol(solid) {
//            Some(sol.primal.obj)
//        }
//        else {
//            None
//        }
//    }
//
//    fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus) {
//        if let Some(sol) = self.select_sol(solid) {
//            (sol.primal.status,sol.dual.status)
//        }
//        else {
//            (SolutionStatus::Undefined,SolutionStatus::Undefined)
//        }
//    }
//    fn primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.primal(self,solid) }
//    
//    fn sparse_primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String> { item.sparse_primal(self,solid) }
//
//    fn dual_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.dual(self,solid) }
//
//    fn primal_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }
//
//    fn dual_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }
//    fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String> where E : IntoExpr<N>     
//    {
//        expr.into_expr().eval(& mut self.rs,&mut self.ws,&mut self.xs).map_err(|e| format!("{:?}",e))?;
//        let mut shape = [0usize; N];
//        let (val,sp) = self.evaluate_primal_internal(solid, &mut shape)?;
//        self.rs.clear();
//        NDArray::new(shape,sp,val)
//    }
//} // impl ModelAPI for Model








//-----------------------------------------------------------------------------
// Single affine constraint

/// A single constraint block `A x + b ∈ D`.
pub struct AffineConstraint<const N : usize,E,D> 
    where E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>,
{
    expr    : E,
    pdomain : Option<D>,
    domain  : Option<ConicDomain<N>>
}


/// Create a structure encapsulating an expression and a domain for constructing disjunctive
/// constraints.
///
/// # Arguments
/// - `expr` Something that can be turned into an expression via [IntoExpr]. 
/// - `domain` Something that can be turned into a [ConicDomain]. Currently this is any linear or
///    conic domain, but not semidefinite domain.
pub fn constraint<const N : usize,I,E,D>(expr : I, domain : D) -> AffineConstraint<N,E,D>
    where I : IntoExpr<N,Result=E>,
          E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>,
{
    AffineConstraint{
        expr    : expr.into_expr(),
        pdomain : Some(domain),
        domain  : None,
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

impl<const N : usize,E,D> disjunction::ConjunctionTrait for AffineConstraint<N,E,D> 
    where E : ExprTrait<N>,
          D : IntoShapedDomain<N>,
          D::Result : IntoConicDomain<N>
{
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> 
    {
        self.expr.eval_finalize(rs,ws,xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,_,_,subj,_) = rs.peek_expr();        
        let mut shape = [0usize;N]; shape.copy_from_slice(eshape);
       
        if let Some(dom) = self.pdomain.take() {
            self.domain = Some(dom.try_into_domain(shape)?.into_conic());
        }


        let (_,_,&dshape,_,is_integer) = self.domain.as_ref().unwrap().get();

        
        if let Some(&i) = subj.iter().max() {
            if i >= numvarelm {
                return Err(format!("Expression has variable element index that is out of bounds for the Model"))
            }
        }

        if is_integer {
            Err(format!("Constraint domains cannot be integer"))
        }
        else if eshape != dshape {
            Err(format!("Mismatching expression and domain shapes"))
        }
        else {
            Ok(1)
        }
    }

    fn append_clause_data(&self, 
                          task  : & mut mosek::TaskCB,
                          vars  : &[VarAtom],
                          exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>, 

                          element_dom  : &mut Vec<i64>,
                          element_ptr  : &mut Vec<usize>,
                          element_afei : &mut Vec<i64>,
                          element_b    : &mut Vec<f64>)
    {
        let (dt,offset,shape,conedim,_) = self.domain.as_ref().unwrap().get();
        let (_,ptr,_,subj,cof) = exprs.pop().unwrap();
        let conesize = if N == 0 { 1 } else { shape[conedim] };
        let numcone  = shape.iter().product::<usize>() / conesize;

        let nelm = ptr.len()-1;

        let afei = task.get_num_afe().unwrap();

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,vars);


        let domidx = match dt {
            ConicDomainType::NonNegative                => task.append_rplus_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::NonPositive                => task.append_rminus_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::Free                       => task.append_r_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::Zero                       => task.append_rzero_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::SVecPSDCone                => task.append_svec_psd_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::QuadraticCone              => task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone       => task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::GeometricMeanCone          => task.append_primal_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::DualGeometricMeanCone      => task.append_dual_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::ExponentialCone            => task.append_primal_exp_cone_domain().unwrap(),
            ConicDomainType::DualExponentialCone        => task.append_dual_exp_cone_domain().unwrap(),
            ConicDomainType::PrimalPowerCone(ref alpha) => task.append_primal_power_cone_domain(conesize.try_into().unwrap(), alpha.as_slice()).unwrap(),
            ConicDomainType::DualPowerCone(ref alpha)   => task.append_dual_power_cone_domain(conesize.try_into().unwrap(), alpha.as_slice()).unwrap(),
        };
        
        let d0 : usize = if N == 0 { 1 } else { shape[0..conedim].iter().product() };
        let d1 : usize = if N == 0 { 1 } else { shape[conedim] };
        let d2 : usize = if N == 0 { 1 } else { shape[conedim+1..].iter().product() };

        let element_afefi_p0 = element_afei.len();
        element_afei.resize(element_afefi_p0+nelm,0);
        let afeidxs = &mut element_afei[element_afefi_p0..];
        afeidxs.copy_from_iter(
            iproduct!(0..d0,0..d2,0..d1)
                .map(|(i0,i2,i1)| afei + (i0*d1*d2 + i1*d2 + i2) as i64));

        element_dom.resize(element_dom.len()+numcone,domidx);
        element_ptr.reserve(numcone);
        {
            let n0 = element_b.len();
            for i in (n0..n0+numcone*conesize).step_by(conesize) { element_ptr.push(i+conesize) }
        }
        element_b.extend_from_slice(offset);

        task.append_afes(nelm as i64).unwrap();
        if asubj.len() > 0 {
            task.put_afe_f_row_list(afeidxs,
                                    aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                    &aptr[..nelm],
                                    asubj.as_slice(),
                                    acof.as_slice()).unwrap();
        }        
       
        task.put_afe_g_list(afeidxs,afix.as_slice()).unwrap();
        if abarsubi.len() > 0 {
            let mut p0 = 0usize;
            for (i,j,p) in izip!(abarsubi.iter(),
                                 abarsubi[1..].iter(),
                                 abarsubj.iter(),
                                 abarsubj[1..].iter())
                .enumerate()
                .filter_map(|(k,(&i0,&i1,&j0,&j1))| if i0 != i1 || j0 != j1 { Some((i0,j0,k+1)) } else { None } )
                .chain(once((*abarsubi.last().unwrap(),*abarsubj.last().unwrap(),abarsubi.len()))) {
               
                let subk = &abarsubk[p0..p];
                let subl = &abarsubl[p0..p];
                let cof  = &abarcof[p0..p];
                p0 = p;

                let dimbarj = task.get_dim_barvar_j(j).unwrap();
                let matidx = task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
            }
        }
    }
}

impl<C> disjunction::DisjunctionTrait for C
    where C : ConjunctionTrait,
{
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        disjunction::ConjunctionTrait::eval(self,numvarelm,rs,ws,xs)
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
        disjunction::ConjunctionTrait::append_clause_data(self,task, vars, exprs, element_dom, element_ptr, element_afei,element_b);
        term_ptr.push(element_dom.len());
    }
}

impl<C> disjunction::DisjunctionTrait for Vec<C> where C : ConjunctionTrait {
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        let mut n = 0;
        for c in self.iter_mut() {
            n += c.eval(numvarelm, rs, ws, xs)?;
        }
        Ok(n)
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
        for c in self.iter() {
            disjunction::ConjunctionTrait::append_clause_data(c,task, vars, exprs, element_dom, element_ptr, element_afei,element_b);
            term_ptr.push(element_dom.len());
        }
    }

}

//-----------------------------------------------------------------------------
// AffineConstraintsAnd

/// Represents the construction `A_1x+b_1 ∈ K_1 AND A_2x+b_2 ∈ K_2`.
pub struct AffineConstraintsAnd<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::ConjunctionTrait
{
     c0 : C0,
     c1 : C1
}

impl<C0,C1> AffineConstraintsAnd<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::ConjunctionTrait
{
    pub fn and<C2>(self, other : C2) -> AffineConstraintsAnd<Self,C2> where C2 : ConjunctionTrait {
        AffineConstraintsAnd { c0: self, c1: other }
    }
    pub fn or<D2>(self, other : D2) -> DisjunctionOr<Self,D2> where D2 : DisjunctionTrait {
        DisjunctionOr { c0: self, c1: other }
    }
}
impl<C0,C1> disjunction::ConjunctionTrait for AffineConstraintsAnd<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::ConjunctionTrait
{
    fn eval(&mut self,
                numvarelm : usize,
                rs : &mut WorkStack,
                ws : &mut WorkStack,
                xs : &mut WorkStack) -> Result<usize,String> {
        Ok(self.c0.eval(numvarelm,rs,ws,xs)? + self.c1.eval(numvarelm,rs,ws,xs)?)
    }

    fn append_clause_data(&self, 
                              task  : & mut mosek::TaskCB,
                              vars  : &[VarAtom],
                              exprs : & mut Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])>, 

                              element_dom  : &mut Vec<i64>,
                              element_ptr  : &mut Vec<usize>,
                              element_afei : &mut Vec<i64>,
                              element_b    : &mut Vec<f64>) {
        self.c0.append_clause_data(task, vars, exprs, element_dom, element_ptr, element_afei, element_b);
        self.c1.append_clause_data(task, vars, exprs, element_dom, element_ptr, element_afei, element_b);
    }
}


//-----------------------------------------------------------------------------
// DisjunctionOr

/// Represents the construction `A OR B` where `A` is set of affine constraints and `B` is another
/// disjunction clause
pub struct DisjunctionOr<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::DisjunctionTrait
{
     c0 : C0,
     c1 : C1
}

impl<C0,C1> DisjunctionOr<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::DisjunctionTrait
{
    pub fn or<C2>(self, other : C2) -> DisjunctionOr<C2,Self> where C2 : ConjunctionTrait {
        DisjunctionOr { c0: other, c1: self }
    }
}

impl<C0,C1> disjunction::DisjunctionTrait for DisjunctionOr<C0,C1> 
    where 
        C0 : disjunction::ConjunctionTrait,
        C1 : disjunction::DisjunctionTrait
{
    fn eval(&mut self,
            numvarelm : usize,
            rs : &mut WorkStack,
            ws : &mut WorkStack,
            xs : &mut WorkStack) -> Result<usize,String> {
        Ok(self.c0.eval(numvarelm, rs, ws, xs).map_err(|e| e.to_string())? +
            self.c1.eval(numvarelm, rs, ws, xs)?)
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
        self.c0.append_clause_data(task, vars, exprs, element_dom, element_ptr, element_afei,element_b);
        term_ptr.push(element_dom.len());
        self.c1.append_disjunction_data(task, vars, exprs, term_ptr, element_dom, element_ptr, element_afei,element_b);
    }
}


/// Split an evaluated expression into linear and semidefinite parts
fn split_expr(eptr    : &[usize],
              esubj   : &[usize],
              ecof    : &[f64],
              vars    : &[VarAtom]) -> (Vec<i32>, //subj
                                        Vec<f64>, //cof
                                        Vec<i64>, //ptr
                                        Vec<f64>, //fix
                                        Vec<i64>, //barsubi
                                        Vec<i32>, //barsubj
                                        Vec<i32>, //barsubk
                                        Vec<i32>, //barsubl
                                        Vec<f64>) { //barcof
    let nnz = *eptr.last().unwrap();
    let nelm = eptr.len()-1;
    let nlinnz = esubj.iter().filter(|&&j| if let VarAtom::BarElm(_,_) = unsafe { *vars.get_unchecked(j) } { false } else { true } ).count();
    let npsdnz = nnz - nlinnz;

    let mut subj : Vec<i32> = Vec::with_capacity(nlinnz);
    let mut cof  : Vec<f64> = Vec::with_capacity(nlinnz);
    let mut ptr  : Vec<i64> = Vec::with_capacity(nelm+1);
    let mut fix  : Vec<f64> = Vec::with_capacity(nelm);
    let mut barsubi : Vec<i64> = Vec::with_capacity(npsdnz);
    let mut barsubj : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barsubk : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barsubl : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barcof  : Vec<f64> = Vec::with_capacity(npsdnz);

    ptr.push(0);
    eptr[..nelm].iter().zip(eptr[1..].iter()).enumerate().for_each(|(i,(&p0,&p1))| {
        let mut cfix = 0.0;
        esubj[p0..p1].iter().zip(ecof[p0..p1].iter()).for_each(|(&idx,&c)| {
            if idx == 0 {
                cfix += c;
            }
            else if c < 0.0 || c > 0.0 {
                match *unsafe{ vars.get_unchecked(idx) } {
                    VarAtom::Linear(j,_) => {
                        subj.push(j);
                        cof.push(c);
                    },
                    VarAtom::ConicElm(j,_coni) => {
                        subj.push(j);
                        cof.push(c);
                    },
                    VarAtom::BarElm(j,ofs) => {
                        let (k,l) = row_major_offset_to_ij(ofs);
                        barsubi.push(i as i64);
                        barsubj.push(j);
                        barsubk.push(k as i32);
                        barsubl.push(l as i32);
                        if k == l {
                            barcof.push(c);
                        }
                        else {
                            barcof.push(0.5 * c);
                        }
                    }
                }
            }
        });
        ptr.push(subj.len() as i64);
        fix.push(cfix);
    });

    (subj,
     cof,
     ptr,
     fix,
     barsubi,
     barsubj,
     barsubk,
     barsubl,
     barcof)
}

fn split_sol_sta(whichsol : i32, solsta : i32) -> (SolutionStatus,SolutionStatus) {
    let (psta,dsta) = 
        match solsta {
            mosek::Solsta::UNKNOWN => (SolutionStatus::Unknown,SolutionStatus::Unknown),
            mosek::Solsta::OPTIMAL => (SolutionStatus::Optimal,SolutionStatus::Optimal),
            mosek::Solsta::PRIM_FEAS => (SolutionStatus::Feasible,SolutionStatus::Unknown),
            mosek::Solsta::DUAL_FEAS => (SolutionStatus::Feasible,SolutionStatus::Unknown),
            mosek::Solsta::PRIM_AND_DUAL_FEAS => (SolutionStatus::Unknown,SolutionStatus::Feasible),
            mosek::Solsta::PRIM_INFEAS_CER => (SolutionStatus::Undefined,SolutionStatus::CertInfeas),
            mosek::Solsta::DUAL_INFEAS_CER => (SolutionStatus::CertInfeas,SolutionStatus::Undefined),
            mosek::Solsta::PRIM_ILLPOSED_CER => (SolutionStatus::Undefined,SolutionStatus::CertIllposed),
            mosek::Solsta::DUAL_ILLPOSED_CER => (SolutionStatus::CertIllposed,SolutionStatus::Undefined),
            mosek::Solsta::INTEGER_OPTIMAL => (SolutionStatus::Optimal,SolutionStatus::Undefined),
            _ => (SolutionStatus::Unknown,SolutionStatus::Unknown)
        };

    if whichsol == mosek::Soltype::ITG {
        (psta,SolutionStatus::Undefined)
    }
    else {
        (psta,dsta)
    }
}

/// Convert linear row-major order offset into a lower triangular
/// matrix to an (i,j) pair.
fn row_major_offset_to_ij(ofs : usize) -> (usize,usize) {
    let i = (((1.0+8.0*ofs as f64).sqrt()-1.0)/2.0).floor() as usize;
    let j = ofs - i*(i+1)/2;
    (i,j)
}
/// Convert linear column-major order offset into a lower triangular
/// matrix and a dimension to a (i,j) pair`
fn row_major_offset_to_col_major(ofs : usize, dim : usize) -> usize {
    let (i,j) = row_major_offset_to_ij(ofs);
    ((2*dim-1)*j - j*j)/2 + i
}

//======================================================
// TEST

#[cfg(test)]
mod tests {
    use matrix::dense;

    use utils::iter::*;
    use crate::*;

    fn eq<T:std::cmp::Eq>(a : &[T], b : &[T]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a,b)| *a == *b )
    }

    #[test]
    fn it_works() {
        let mut m = Model::new(Some("SuperModel"));
        let _v1 = m.variable(None, greater_than(5.0));
        let _v2 = m.variable(None, 10);
        let _v3 = m.variable(None, &[3,3]);
        let _v4 = m.variable(None, in_quadratic_cone().with_shape(&[5]));
        let _v5 = m.variable(None, greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(&[2,2]));
        let _v6 = m.variable(None, greater_than(vec![1.0,3.0]).with_shape_and_sparsity(&[2,2],&[[0,0],[1,1]]));
    }

    #[test]
    fn variable_stack() {
        let mut m = Model::new(Some("SuperModel"));
        let v1 = m.variable(None, &[3,2,1]);
        let v2 = m.variable(None, &[3,2,1]);
        let v3 = m.variable(None, equal_to(vec![1.0,2.0,3.0,4.0]).with_shape_and_sparsity(&[3,2,1],&[[0,0,0],[1,0,0],[1,1,0],[2,1,0]]));

        let w_0 = Variable::stack(0,&[&v1,&v2]);
        let w_1 = Variable::stack(1,&[&v1,&v2]);
        let _w_2 = Variable::stack(2,&[&v1,&v2]);

        assert!(eq(w_0.shape(),&[6,2,1]));
        assert!(eq(w_0.idxs(),&[1,2,3,4,5,6,7,8,9,10,11,12]));
        assert!(eq(w_1.shape(),&[3,4,1]));
        assert!(eq(w_1.idxs(),&[1,2,7,8,3,4,9,10,5,6,11,12]));

        let u_0 = Variable::stack(0,&[&v1,&v3]);
        let u_1 = Variable::stack(1,&[&v1,&v3]);
        let u_2 = Variable::stack(2,&[&v1,&v3]);

        assert!(eq(u_0.shape(),&[6,2,1]));
        assert!(eq(u_0.idxs(),&[1,2,3,4,5,6,13,14,15,16]));
        assert!(eq(u_0.sparsity().unwrap(),&[0,1,2,3,4,5,6,8,9,11]));
        assert!(eq(u_1.shape(),&[3,4,1]));
        assert!(eq(u_1.idxs(),     &[1,2,13,3,4,14,15,5,6,16]));
        assert!(eq(u_1.sparsity().unwrap(), &[0,1,2,4,5,6,7,8,9,11]));
        assert!(eq(u_2.shape(),&[3,2,2]));
        assert!(eq(u_2.idxs(),     &[1,13,2,3,14,4,15,5,6,16]));
        assert!(eq(u_2.sparsity().unwrap(), &[0,1,2,4,5,6,7,8,10,11]));
    }

    #[test]
    fn psd_constraint() {
        let mut m = Model::new(Some("SuperModel"));
        let x = m.variable(Some("x"), unbounded().with_shape(&[3,2,3]));
        //let y = m.variable(Some("y"), unbounded().with_shape(&[3,2,3]));
        let z = m.variable(Some("z"), zero());

        let c = m.constraint(Some("c"), x.sub(dense([3,2,3],vec![0.0,-1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0,-9.0,-10.0,-11.0,-12.0,-13.0,-14.0,-15.0,-16.0,-17.0])), in_psd_cones(&[3,2,3]).with_conedims(0,2));
        m.objective(Some("obj"), Sense::Minimize, &z);

        m.solve();
        m.write_problem("psd.ptf");

        let csol = m.primal_solution(SolutionType::Default, &c).unwrap();

        let shape = [3,2,3];
        let m1 : Vec<f64> = shape.index_iterator().zip(csol.iter()).filter_map(|(index,v)| if index[1] == 0 { Some(*v) } else { None } ).collect();
        let m2 : Vec<f64> = shape.index_iterator().zip(csol.iter()).filter_map(|(index,v)| if index[1] == 1 { Some(*v) } else { None } ).collect();

        let mut m1eig = [0.0; 3];
        let mut m2eig = [0.0; 3];
        mosek::syeig(mosek::Uplo::LO, 3, &m1, &mut m1eig).unwrap();
        mosek::syeig(mosek::Uplo::LO, 3, &m2, &mut m2eig).unwrap();
    
        println!("eig1 = {:?}",m1eig);
        println!("m1 = {:?}",m1);
        println!("c = {:?}",c.idxs);

        assert!(m1eig.iter().all(|v| *v >= 0.0));
        assert!(m2eig.iter().all(|v| *v >= 0.0));
    }


    #[test]
    fn eval_expr() {
        let mut m = Model::new(None);
        let x = m.variable(None, equal_to(vec![1.0,2.0,3.0,4.0,5.0,6.0]));
        let y = m.variable(None, equal_to(vec![1.0,2.0,1.0,2.0,1.0,2.0]));
        m.solve();

        {
            let r = m.evaluate_primal(SolutionType::Default, &x).unwrap();
            let (shape,sp,val) = r.dissolve();
            assert!( sp.is_none());
            assert_eq!(shape,[6]);
            assert_eq!(val,&[1.0,2.0,3.0,4.0,5.0,6.0]);
        }
        {
            let r = m.evaluate_primal(SolutionType::Default, x.add(y.mul(2.0))).unwrap();
            let (shape,sp,val) = r.dissolve();
            assert!(sp.is_none());
            assert_eq!(shape,[6]);
            assert_eq!(val,&[3.0,6.0,5.0,8.0,7.0,10.0]);
        }
    }
}
