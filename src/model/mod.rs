#![doc = include_str!("../../js/mathjax.tag")]

use itertools::izip;
use std::fmt::Debug;
use std::ops::ControlFlow;
use std::path::Path;
use crate::{disjunction, IntoExpr, ExprTrait, NDArray};
use crate::utils::iter::*;
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
impl Default for SolutionStatus { fn default() -> Self { SolutionStatus::Undefined } }

#[derive(Default)]
pub struct SolutionPart {
    status : SolutionStatus,
    var    : Vec<f64>,
    con    : Vec<f64>,
    obj    : f64,

}

impl SolutionPart {
    pub fn new(numvar : usize, numcon : usize) -> SolutionPart { SolutionPart{status : SolutionStatus::Unknown, var : vec![0.0; numvar], con : vec![0.0; numcon], obj : 0.0} }
    fn resize(& mut self,numvar : usize, numcon : usize) {
        self.var.resize(numvar, 0.0);
        self.con.resize(numcon, 0.0);
    }
}

#[derive(Default)]
pub struct Solution {
    primal : SolutionPart,
    dual   : SolutionPart
}

impl Solution {
    pub fn new() -> Solution { Solution{primal : SolutionPart::new(0,0) , dual : SolutionPart::new(0,0)  } }
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
    fn new(name : Option<&str>) -> Self;

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
    fn linear_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearDomain<N>,shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
        where 
            Self : Sized;
    fn ranged_variable<const N : usize,R>(&mut self, name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized;
    fn ranged_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearRangeDomain<N>,shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<<LinearRangeDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
        where 
            Self : Sized;

    fn update(& mut self, idxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>;

    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>;

    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, solitg : &mut Solution) -> Result<(),String>;

    fn objective(&mut self, name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>;

    fn set_param<V>(&mut self, parname : &str, parval : V) -> Result<(),String> where V : SolverParameterValue<Self>,Self: Sized;
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
    fn disjunction(& mut self, name : Option<&str>, 
                   exprs     : &[(&[usize],&[usize],&[usize],&[f64])], 
                   domains   : &[Box<dyn AnyConicDomain>],
                   term_size : &[usize]) -> Result<Disjunction,String>;
}

/// An inner model object must implement this to support log callbacks.
pub trait ModelWithLogCallback {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str);
}

/// An inner model object must implement this to support integer solution callbacks
pub trait ModelWithIntSolutionCallback {
    fn set_solution_callback<F>(&mut self, func : F) where F : 'static+FnMut(f64, &[f64],&[f64]);
}

/// An inner model object must implement this to support control callbacks (callbacks that allow us
/// to ask the solver to terminate nicely).
pub trait ModelWithControlCallback {
    fn set_callback<F>(&mut self, func : F) where F : 'static+FnMut() -> ControlFlow<(),()>;
}

pub struct ModelAPI<T : BaseModelTrait> {
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

impl<T : BaseModelTrait+Default> Default for ModelAPI<T> {
    fn default() -> Self {
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
}

impl<T> ModelAPI<T> where T : BaseModelTrait {
    pub fn new(name : Option<&str>) -> ModelAPI<T> {
        ModelAPI{
            inner : T::new(name),
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

    /// Attach a solution callback function. This is called for each new integer solution. The new
    /// solution can be accessed though the [ModelAPI]
    pub fn set_int_solution_callback<F>(&mut self, mut func : F) where F : 'static+FnMut(&mut Self), T : 'static+ModelWithIntSolutionCallback {
        // NOTE: We cheat here. 
        // 1. We pass self as a pointer to bypass the whole lifetime issue. This
        // is acceptable because we KNOW self will outlive the underlying Task.
        // 2. The constuction is acceptable because we know that the callback is called from inside
        //    the `.solve()` call, which holds a mutable reference to model.
        let modelp :  *const ModelAPI<T> = self;

        self.inner.set_solution_callback(move |pobj,xx,xc| {
            let model : & mut Self = unsafe { & mut (* (modelp as * mut Self)) };

            model.sol_itg.primal.status = SolutionStatus::Feasible;
            model.sol_itg.dual.status = SolutionStatus::Undefined;
            if model.sol_itg.primal.var.len() != xx.len() {
                model.sol_itg.primal.var = xx.to_vec();
            }
            else {
                model.sol_itg.primal.var.copy_from_slice(xx);
            }
            if model.sol_itg.primal.con.len() != xc.len() {
                model.sol_itg.primal.con = xc.to_vec();
            }
            else {
                model.sol_itg.primal.con.copy_from_slice(xc);
            }

            model.sol_itg.primal.obj = pobj;

            func(model);
        });
    }

    pub fn set_control_callback<F>(&mut self, func : F) where F : 'static+FnMut() -> ControlFlow<(),()>, T : ModelWithControlCallback {
        self.inner.set_callback(func);
    }

    /// Set the objective.
    ///
    /// Arguments:
    /// - `name` Optional objective name
    /// - `sense` Objective sense
    /// - `expr` Objective expression, this must contain exactly one
    ///   element. The shape is otherwise ignored.
    pub fn try_objective<I>(& mut self, name : Option<&str>, sense : Sense, e : I) -> Result<(),String> where I : IntoExpr<0> 
    {
        e.into_expr().eval_finalize(&mut self.rs, &mut self.ws, &mut self.xs).map_err(|er| er.to_string())?;
        let (shape,_ptr,sp,subj,cof) = self.rs.pop_expr();
        if sp.is_some() {
            panic!("Internal: eval_finalize must produce a dense expression");
        }
        if shape.len() != 0 {
            return Err(format!("Invalid expression shape for objective: {:?}",shape));
        }
        self.inner.objective(name,sense,subj,cof)
    }


    /// Same as [ModekAPI::try_objective], but `panic`s on error.
    pub fn objective<I>(& mut self, name : Option<&str>, sense : Sense, e : I) where I : IntoExpr<0> 
    {
        self.try_objective(name, sense, e).unwrap();
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
            D : VarDomainTrait<T,Result = R>,
            T : Sized
    {
        dom.try_into_domain()?.create(& mut self.inner,name)
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
            D : VarDomainTrait<T,Result = R>,
            T : Sized
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
            D : ConstraintDomain<N,T>,
            T : Sized
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        dom.try_into_domain(shape)?.add_constraint(& mut self.inner,name,eshape,ptr,subj,cof)
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
            D : ConstraintDomain<N,T>,
            T : Sized
    {
        self.try_constraint(name, expr, dom).unwrap()
    }
    
    /// Update the expression of a constraint in the Model.
    pub fn try_update<const N : usize, E : IntoExpr<N>>(&mut self, item : &Constraint<N>, expr : E) -> Result<(),String>
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        self.inner.update(item.idxs.as_slice(),eshape,ptr,subj,cof)
    }

    /// Same as [ModelAPI::try_update], but `panic`s on error.
    pub fn update<const N : usize, E : IntoExpr<N>>(&mut self, item : &Constraint<N>, e : E)
    {
        self.try_update(item,e).unwrap()
    }
    

    /// Add a disjunctive constraint to the model. A disjunctive constraint is a logical constraint
    /// of the form
    /// $$
    /// \\begin{array}{c}
    ///     A_1x+b_1 \\in K_1 \\\\
    ///     \\mathrm{or}      \\\\
    ///     \\vdots           \\\\
    ///     \\mathrm{or}      \\\\
    ///     A_nx+b_n \\in K_n 
    /// \\end{array}
    /// $$
    /// where each \\(K_\\cdot\\) is a single cone or a product of cones. 
    ///
    ///
    /// Another example is an indicator constraint like an indicator constraint
    /// $$ 
    ///     z\\in\\{0,1\\},\\ z=1 \\Rightarrow Ax+b\\in K_n 
    /// $$
    /// implemented as 
    /// $$
    ///     z\\in\\{0,1\\},\\ z = 0\\ \\vee\\ z = 1\\wedge Ax+b\\in K_n
    /// $$
    ///
    /// # Arguments
    /// - `name` Name of the disjunction
    /// - `terms` Structure defining the terms of the disjunction.
    ///
    /// # Example: Simple disjunction
    /// A Simple logical disjunctions like 
    /// $$
    ///     a^Tx+b = 3 \\vee\\ b^Tx+c = 1.0
    /// $$
    /// can be implemented as:
    /// ```rust
    /// use mosekcomodel::*;
    /// let mut model = Model::new(None);
    /// let x = model.variable(Some("x"), 5);
    /// let y = model.variable(Some("y"), 3);
    /// let a = vec![1.0,2.0,3.0,4.0,5.0];
    /// let b = vec![5.0,4.0,3.0];
    /// model.disjunction(None, 
    ///                   constraint(x.dot(a), equal_to(3.0))
    ///                     .or(constraint(y.dot(b), equal_to(1.0))));
    /// ```
    ///
    /// # Example: Indicator constraint
    ///
    /// A logical constraint for a binary variable \\(z\\in\\{0,1\\}\\) of the form
    /// $$
    ///     z = 1 \\Rightarrow a^T x+b = 1
    /// $$
    //  is implemented as 
    //  $$
    //      z = 0\\ \\vee\\ \\left[ z=1\\ \\wedge\\ a^Tx+b=1 \\right]
    //  $$
    /// ```rust
    /// use mosekcomodel::*;
    /// let mut model = Model::new(None);
    /// let a = vec![1.0,2.0,3.0,4.0,5.0];
    /// let x = model.variable(Some("x"), 5);
    /// let z = model.variable(Some("z"), nonnegative().integer());
    /// model.constraint(None,&z,less_than(1.0));
    /// model.disjunction(None,
    ///                   constraint(z.clone(),equal_to(0.0))
    ///                     .or(constraint(z, equal_to(1.0))
    ///                           .and(constraint(x.dot(a), equal_to(1.0)))));
    /// ```
    pub fn try_disjunction<D>(& mut self, name : Option<&str>, mut terms : D) -> Result<Disjunction,String> 
        where 
            D : disjunction::DisjunctionTrait, 
            T : DJCModelTrait 
    {
        let mut domains = Vec::new();
        let mut term_size = Vec::new();
        terms.eval(&mut domains, & mut term_size, &mut self.rs, &mut self.ws, &mut self.xs)?;

        let nexprs = term_size.iter().sum();
        let exprs : Vec<(&[usize],&[usize],&[usize],&[f64])> = 
            self.rs.pop_exprs(nexprs).iter().rev()
                .map(|(shape,ptr,sp,subj,cof)| { if sp.is_some() { panic!("Internal invalid: Sparse evaluated expression") } (*shape,*ptr,*subj,*cof) })
                .collect();

        self.inner.disjunction(name,exprs.as_slice(), domains.as_slice(), term_size.as_slice())
    }
   

    /// Same as [ModelAPI::try_disjunction], but `panic`s on errors.
    pub fn disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Disjunction where D : disjunction::DisjunctionTrait, T : DJCModelTrait {
        self.try_disjunction(name, terms).unwrap()
    }

    /// Write problem to a file. The file is written by the underlying solver task, so no
    /// structural information will be written.
    ///
    /// # Arguments
    /// - `filename` The filename extension determines the file format to use. If the
    ///   file extension is not recognized, the MPS format is used.
    ///
    pub fn try_write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        self.inner.write_problem(filename)
    }

    pub fn write_problem<P:AsRef<Path>>(&self, filename : P) {
        self.try_write_problem(filename).unwrap();
    }

    /// Solve the problem and extract the solution.
    ///
    /// This will fail if the optimizer fails with an error. Not producing a solution (stalling or
    /// otherwise failing), producing a non-optimal solution or a certificate of infeasibility 
    /// is *not* an error.
    pub fn try_solve(& mut self) -> Result<(),String> {
        self.sol_bas.primal.status = SolutionStatus::Undefined;
        self.sol_bas.dual.status = SolutionStatus::Undefined;
        self.sol_itr.primal.status = SolutionStatus::Undefined;
        self.sol_itr.dual.status = SolutionStatus::Undefined;
        self.sol_itg.primal.status = SolutionStatus::Undefined;
        self.sol_itg.dual.status = SolutionStatus::Undefined;

        self.inner.solve(&mut self.sol_itr, &mut self.sol_bas, & mut self.sol_itg)
    }

    /// Same as [ModelAPI::try_solve], but panics on any error.
    pub fn solve(&mut self) { self.try_solve().unwrap(); }

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
        self.select_sol(solid)
            .map(|sol| (sol.primal.status,sol.dual.status))
            .unwrap_or((SolutionStatus::Undefined,SolutionStatus::Undefined))
    }

    /// Get primal objective value, if available.
    ///
    /// The primal objective is only available if the primal solution is defined.
    pub fn primal_objective_value(&self, solid : SolutionType) -> Option<f64> {
        self.select_sol(solid)    
            .map(|sol| sol.primal.obj)
    }
    
    /// Get dual objective value, if available.
    ///
    /// The dual objective is only available if the dual solution is defined.
    pub fn dual_objective_value(&self, solid : SolutionType) -> Option<f64> {
        self.select_sol(solid)
            .map(|sol| sol.dual.obj)
    }





    pub(crate) fn primal_var_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.primal.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.primal.var.len() { panic!("Variable indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.var.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }

    pub(crate) fn dual_var_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.dual.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.dual.var.len() { panic!("Variable indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.dual.var.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }
    fn primal_con_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.primal.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.primal.con.len() { panic!("Constraint indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.con.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }
    fn dual_con_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.dual.status  {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.dual.con.len() { panic!("Constraint indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.con.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
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
    pub fn primal_solution<const N : usize, I:ModelItem<N,T>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> {
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
    pub fn sparse_primal_solution<const N : usize, I:ModelItem<N,T>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String> {
        item.sparse_primal(self,solid)
    }

    /// Get dual solution values for an item
    ///
    /// Returns: If solution item is defined, return the solution, otherwise a n error message.
    pub fn dual_solution<const N : usize, I:ModelItem<N,T>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> {
        item.dual(self,solid)
    }

    /// Get primal solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn primal_solution_into<const N : usize, I:ModelItem<N,T>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> {
        item.primal_into(self,solid,res)
    }

    /// Get dual solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn dual_solution_into<const N : usize, I:ModelItem<N,T>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> {
        item.primal_into(self,solid,res)
    }

    
    /// Evaluate an expression in the (primal) solution. 
    ///
    /// # Arguments
    /// - `solid` The solution in which to evaluate the expression.
    /// - `expr` The expression to evaluate.
    pub fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String> where E : IntoExpr<N> {
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


    pub fn try_set_param<V : SolverParameterValue<T>>(&mut self, parname : &str, parval : V) -> Result<(),String> {
        self.inner.set_param(parname, parval)
    }
    pub fn set_param<V : SolverParameterValue<T>>(&mut self, parname : &str, parval : V) {
        self.try_set_param(parname, parval).unwrap();
    }
}

















//======================================================
// ModelItem
//======================================================

/// The `ModelItem` represents either a variable or a constraint belonging to a [Model]. It is used
/// by the [Model] object when accessing solution assist overloading and determine which solution part to access.
pub trait ModelItem<const N : usize,M> where M : BaseModelTrait {
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
    fn dual_into(&self,  m : &ModelAPI<M>,  solid : SolutionType,   res : & mut [f64]) -> Result<usize,String>;
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

impl Disjunction {
    pub fn new(index : i64) -> Disjunction { Disjunction{index }}
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

impl<const N : usize,M> ModelItem<N,M> for Variable<N> where M : BaseModelTrait {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn shape(&self) -> [usize; N] { self.shape }
    
    fn sparse_primal(&self,m : &ModelAPI<M>,solid : SolutionType) -> Result<(Vec<f64>,Vec<[usize;N]>),String> {
        let mut nnz = vec![0.0; self.numnonzeros()];
        let dflt = [0usize; N];
        let mut idx : Vec<[usize;N]> = vec![dflt;self.numnonzeros()];
        self.sparse_primal_into(m,solid,nnz.as_mut_slice(),idx.as_mut_slice())?;
        Ok((nnz,idx))
    }

    fn primal_into(&self,m : &ModelAPI<M>,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; });
            }
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &ModelAPI<M>,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; })
            }
            Ok(sz)
        }
    }
}

pub trait SolverParameterValue<M : BaseModelTrait> {
    fn set(self,parname : &str, model : & mut M) -> Result<(),String>;
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





/// Split an evaluated expression into linear and semidefinite parts

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
