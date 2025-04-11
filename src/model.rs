//!
#[doc = include_str!("../js/mathjax.tag")]

use itertools::{merge_join_by, EitherOrBoth};
use itertools::{iproduct, izip};
use std::collections::HashMap;
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

struct Solution {
    primal : SolutionPart,
    dual   : SolutionPart
}

impl Solution {
    fn new() -> Solution { Solution{primal : SolutionPart::new(0,0) , dual : SolutionPart::new(0,0)  } }
}


/// Represents something that can be used as a domain for a variable.
pub trait VarDomainTrait {
    type Result; 
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result;
}

/// Implement ConicDomain as a variable domain
impl<const N : usize> VarDomainTrait for ConicDomain<N> {
    type Result = Variable<N>;
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result {
        m.conic_variable(name,self)
    }
}
/// Implement a fixed-size integer array as domain for variable, meaning unbounded with the array
/// as shape.
impl<const N : usize> VarDomainTrait for &[usize;N] {
    type Result = Variable<N>;
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result {
        m.free_variable(name,self)
    }
}

/// Implement LinearDomain as variable domain
impl<const N : usize> VarDomainTrait for LinearDomain<N> {
    type Result = Variable<N>;
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result {
        m.linear_variable(name,self)
    }
}

/// Implement integer as domain for variable, producing a vector variable if the given size.
impl VarDomainTrait for usize {
    type Result = Variable<1>;
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result {
        m.free_variable(name,&[self])
    }
}
/// Implements PSD domain for variables.
impl<const N : usize> VarDomainTrait for PSDDomain<N> {
    type Result = Variable<N>;
    fn create(self, m : & mut Model, name : Option<&str>) -> Self::Result {
        m.psd_variable(name,self)
    }
}

//======================================================
// Model
//======================================================

/// The `Model` object encapsulates an optimization problem and a
/// mapping from the structured API to the internal Task items.
///
/// Variables and constraints are created through the `Model` object and belong to exactly that
/// model.
///
/// # Example
///
/// A Basic example setting up a model and adding variables and constraints:
/// ```rust
/// use mosekcomodel::*;
///
/// // Create a model with a name
/// let mut model = Model::new(Some("MyModel"));
/// // Create a scalar unbounded variable
/// let x = model.variable(Some("x"), unbounded());
/// // Create a conic variable consisting of 4 quadratic cones of size 3
/// let y = model.variable(Some("y"), in_quadratic_cone().with_shape(&[4,3]));
/// // Create a binary variable
/// let z = model.ranged_variable(Some("z"),in_range(0.0, 1.0).integer()).0;
/// 
/// // Create a scalar constraint
/// _ = model.constraint(Some("C1"), x.add(y.index([0,0])), equal_to(5.0));
/// ```
#[doc = include_str!("../js/mathjax.tag")]
pub struct Model {
    /// The MOSEK task
    task : mosek::TaskCB,
    /// Vector of scalar variable atoms
    vars : Vec<VarAtom>,
    /// Vector of scalar constraint atoms
    cons : Vec<ConAtom>,

    /// Remote opt server host and access token
    optserver_host : Option<(String,Option<String>)>,

    /// Basis solution
    sol_bas : Solution,
    /// Interior solution
    sol_itr : Solution,
    /// Integer solution
    sol_itg : Solution,

    /// Workstacks for evaluating expressions
    rs : WorkStack,
    ws : WorkStack,
    xs : WorkStack
}

pub trait BaseModelTrait {
    fn try_linear_variable<const N : usize>(&mut self, name : Option<&str>,dom : LinearDomain<N>) -> Result<Variable<N>,String>;
    fn try_linear_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearDomain<N>) -> Result<Constraint<N>,String>;
}
pub trait ConicModelTrait {
    fn try_conic_variable<const N : usize>(&mut self, name : Option<&str>,dom : ConicDomain<N>) -> Result<Variable<N>,String>;
    fn try_conic_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : ConicDomain<N>) -> Result<Constraint<N>,String>;
}
pub trait PSDModelTrait {
    fn try_psd_variable<const N : usize>(&mut self, name : Option<&str>, dom : PSDDomain<N>) -> Variable<N>;
    fn try_psd_constraint<const N : usize>(& mut self, name : Option<&str>, dom : PSDDomain<N>) -> Result<Constraint<N>,String>;
}

//======================================================
// ModelItem
//======================================================

/// The `ModelItem` represents either a variable or a constraint belonging to a [Model]. It is used
/// by the [Model] object when accessing solution assist overloading and determine which solution part to access.
pub trait ModelItem<const N : usize> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn shape(&self) -> [usize;N];
    //fn numnonzeros(&self) -> usize;
    fn sparse_primal(&self,m : &Model,solid : SolutionType) -> Result<(Vec<f64>,Vec<[usize;N]>),String> {
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
    fn primal(&self,m : &Model,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.primal_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn dual(&self,m : &Model,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.dual_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String>;
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String>;
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

impl<const N : usize> ModelItem<N> for Constraint <N> {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn shape(&self) -> [usize; N] { self.shape }
    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
}



pub trait SolverParameterValue {
    fn set(self,parname : &str, model : & mut Model);
}

impl SolverParameterValue for f64 {
    fn set(self, parname : &str,model : & mut Model) { model.set_double_parameter(parname,self) }
}

impl SolverParameterValue for i32 {
    fn set(self, parname : &str,model : & mut Model) { model.set_int_parameter(parname,self) }
}

impl SolverParameterValue for &str {
    fn set(self, parname : &str,model : & mut Model) { model.set_str_parameter(parname,self) }
}


//======================================================
// Model
//======================================================

impl Model {
    /// Create new Model object.
    ///
    /// # Arguments
    /// - `name` An optional name
    /// # Returns
    /// An empty model.
    /// # Example
    /// ```rust
    /// use mosekcomodel::*;
    /// let mut model = Model::new(Some("SuperModel"));
    /// ```
    pub fn new(name : Option<&str>) -> Model {
        let mut task = mosek::Task::new().unwrap().with_callbacks();
        if let Some(name) = name { task.put_task_name(name).unwrap() };
        task.put_int_param(mosek::Iparam::PTF_WRITE_SOLUTIONS, 1).unwrap();
        Model{
            task,
            vars    : vec![VarAtom::Linear(-1,WhichLinearBound::Both)],
            cons    : Vec::new(),

            optserver_host : None,

            sol_bas : Solution::new(),
            sol_itr : Solution::new(),
            sol_itg : Solution::new(),
            rs      : WorkStack::new(0),
            ws      : WorkStack::new(0),
            xs      : WorkStack::new(0)
        }
    }

    /// Attach a log printer callback to the Model. This will receive messages from the solver
    /// while solving and during a few other calls like file reading/writing. 
    ///
    /// # Arguments
    /// - `func` A function that will be called with strings from the log. Individual lines may be
    ///   written in multiple chunks to there is no guarantee that the strings will end with a
    ///   newline.
    pub fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str) {
        self.task.put_stream_callback(mosek::Streamtype::LOG,func).unwrap();
    }

    /// Attach a solution callback function. This is called for each new integer solution 
    pub fn set_solution_callback<F>(&mut self, mut func : F) where F : 'static+FnMut(&Model) {
        // NOTE: We cheat here. We pass self as a pointer to bypass the whole lifetime issue. This
        // is acceptable because we KNOW self will outlive the underlying Task.
        let modelp :  *const Model = self;

        self.task.put_intsolcallback(move |xx| {
            let model : & mut Model = unsafe { & mut (* (modelp as * mut Model)) };
            model.sol_bas.primal.status = SolutionStatus::Undefined;
            model.sol_bas.dual.status = SolutionStatus::Undefined;
            model.sol_itr.primal.status = SolutionStatus::Undefined;
            model.sol_itr.dual.status = SolutionStatus::Undefined;
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

    /// Write problem to a file. The file is written by the underlying solver task, so no
    /// structural information will be written.
    ///
    /// # Arguments
    /// - `filename` The filename extension determines the file format to use. If the
    ///   file extension is not recognized, the MPS format is used.
    ///
    pub fn write_problem<P>(&self, filename : P) where P : AsRef<Path> {
        let path = filename.as_ref();
        self.task.write_data(path.to_str().unwrap()).unwrap();
    }

    //======================================================
    // Variable interface

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
    pub fn try_variable<I,D>(& mut self, name : Option<&str>, dom : I) -> Result<D::Result,String>
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait,
    {
        Ok(dom.try_into_domain()?.create(self,name))
    }

    /// Add a Variable. See [Model::try_variable].
    ///
    /// # Returns
    /// An `N`-dimensional variable object is returned. The `Variable` object may be dense or
    /// sparse, where "sparse" means that all entries outside the sparsity pattern are fixed to 0.
    ///
    /// Panics on any error.
    pub fn variable<I,D>(& mut self, name : Option<&str>, dom : I) -> D::Result
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait,
    {
        dom.try_into_domain().unwrap().create(self,name)
    }

    fn var_names<const N : usize>(& mut self, name : &str, first : i32, shape : &[usize;N], sp : Option<&[usize]>) {
        let mut buf = name.to_string();
        let baselen = buf.len();

        if let Some(sp) = sp {
            SparseIndexIterator::new(shape,sp)
                .enumerate()
                .for_each(|(j,index)| {
                    buf.truncate(baselen);
                    index.append_to_string(& mut buf);
                    //println!("name is now: {}",buf);
                    self.task.put_var_name(first + j as i32,buf.as_str()).unwrap();
                });
        }
        else {
            IndexIterator::new(shape)
                .enumerate()
                .for_each(|(j,index)| {
                    buf.truncate(baselen);
                    index.append_to_string(&mut buf);
                    self.task.put_var_name(first + j as i32,buf.as_str()).unwrap();
                });
        }
    }

    fn con_names<const N : usize>(task : & mut mosek::TaskCB, name : &str, first : i32, shape : &[usize; N]) {
        let mut buf = name.to_string();
        let baselen = buf.len();
        //println!("---------con_names '{}' shape = {:?}",name,shape);
        IndexIterator::new(shape)
            .enumerate()
            .for_each(|(j,index)| {
                buf.truncate(baselen);
                index.append_to_string(&mut buf);
                //println!("    name = {}",buf);
                task.put_con_name(first + j as i32,buf.as_str()).unwrap();
            });
    }


    /// Create a ranged variable
    ///
    /// # Arguments
    /// - `name` Optional variable name
    /// - `dom` variable domain range, see [in_range].
    ///
    /// # Returns
    /// Ok success, return a pair if variables. When used as variables or for getting the primal
    /// solution values, they are identical, but when getting the dual solution values, the first
    /// of the pair will fetch the dual bound corresponding to the lower bound, and the second the
    /// dual values for the upper bound.
    /// On error, return a string with the reason.
    pub fn try_ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> Result<(Variable<N>,Variable<N>),String> 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>
    {
        let domain = dom.into_range()?;
        let vari = self.task.get_num_var().unwrap();
        let n : usize = domain.shape.iter().product();
        let nelm = domain.sparsity.as_ref().map(|v| v.len()).unwrap_or(n);
        let varend : i32 = ((vari as usize) + nelm).try_into().unwrap();
        let firstvar = self.vars.len();
        self.vars.reserve(nelm*2);

        (vari..vari+nelm as i32).for_each(|j| self.vars.push(VarAtom::Linear(j,WhichLinearBound::Lower)));
        (vari..vari+nelm as i32).for_each(|j| self.vars.push(VarAtom::Linear(j,WhichLinearBound::Upper)));
        self.task.append_vars(nelm as i32).unwrap();
        if let Some(name) = name {
            self.var_names(name,vari,&domain.shape,None)
        }

        if domain.is_integer {
            self.task.put_var_type_list((vari..varend).collect::<Vec<i32>>().as_slice(), vec![mosek::Variabletype::TYPE_INT;nelm].as_slice()).unwrap();
        }
        
        self.task.put_var_bound_slice(vari,varend,vec![mosek::Boundkey::RA;n].as_slice(),domain.lower.as_slice(),domain.upper.as_slice()).unwrap();
        Ok((Variable::new((firstvar..firstvar+nelm).collect(),        domain.sparsity.clone(), &domain.shape),
            Variable::new((firstvar+nelm..firstvar+nelm*2).collect(), domain.sparsity, &domain.shape)))
    }


    /// Create a ranged variable. See [Model::try_ranged_variable].
    ///
    /// It will panic on any error.
    pub fn ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> (Variable<N>,Variable<N>) 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>
    {
        self.try_ranged_variable(name, dom).unwrap()
    }

    fn linear_variable<const N : usize>(&mut self, name : Option<&str>,dom : LinearDomain<N>) -> Variable<N> {
        let (dt,b,shape_,sp,isint) = dom.extract();
        let mut shape = [0usize; N]; shape.clone_from_slice(&shape_);

        let n = b.len();
        let vari = self.task.get_num_var().unwrap();
        let varend : i32 = ((vari as usize)+n).try_into().unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        if isint {
            self.task.put_var_type_list((vari..varend).collect::<Vec<i32>>().as_slice(), vec![mosek::Variabletype::TYPE_INT; n].as_slice()).unwrap();
        }
        //println!("linear_variable n = {},curnumvar = {}",n,vari);
        if let Some(name) = name {
            if let Some(ref sp) = sp {
                self.var_names(name,vari,&shape,Some(sp.as_slice()))
            }
            else {
                self.var_names(name,vari,&shape,None)
            }
        }
        self.vars.reserve(n);

        let firstvar = self.vars.len();
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j,WhichLinearBound::Both)));

        match dt {
            LinearDomainType::Free        => self.task.put_var_bound_slice_const(vari,vari+n as i32,mosek::Boundkey::FR,0.0,0.0).unwrap(),
            LinearDomainType::Zero        => {
                let bk = vec![mosek::Boundkey::FX; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),b.as_slice(),b.as_slice()).unwrap();
            },
            LinearDomainType::NonNegative => {
                let bk = vec![mosek::Boundkey::LO; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),b.as_slice(),b.as_slice()).unwrap();
            },
            LinearDomainType::NonPositive => {
                let bk = vec![mosek::Boundkey::UP; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),b.as_slice(),b.as_slice()).unwrap()
            }
        }

        Variable::new((firstvar..firstvar+n).collect(),
                      sp,
                      &shape)
    }

    fn free_variable<const N : usize>(&mut self, name : Option<&str>, shape : &[usize;N]) -> Variable<N> {
        let vari = self.task.get_num_var().unwrap();
        let n : usize = shape.iter().product();
        let varend : i32 = ((vari as usize) + n).try_into().unwrap();
        let firstvar = self.vars.len();
        self.vars.reserve(n);
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j,WhichLinearBound::Both)));
        self.task.append_vars(n as i32).unwrap();
        if let Some(name) = name {
            self.var_names(name,vari,shape,None)
        }
        self.task.put_var_bound_slice_const(vari,varend,mosek::Boundkey::FR,0.0,0.0).unwrap();
        Variable::new((firstvar..firstvar+n).collect(),
                      None,
                      shape)
    }

    fn psd_variable<const N : usize>(&mut self, name : Option<&str>, dom : PSDDomain<N>) -> Variable<N> {
        let (shape,(conedim0,conedim1)) = dom.dissolve();
        if conedim0 == conedim1 || conedim0 >= N || conedim1 >= N { panic!("Invalid cone dimensions") };
        if shape[conedim0] != shape[conedim1] { panic!("Mismatching cone dimensions") };

        let (cdim0,cdim1) = if conedim0 < conedim1 { (conedim0,conedim1) } else { (conedim1,conedim0) };

        let d0 = shape[0..cdim0].iter().product();
        let d1 = shape[cdim0];
        let d2 = shape[cdim0+1..cdim1].iter().product();
        let d3 = shape[cdim1];
        let d4 = shape[cdim1+1..].iter().product();

        let numcone = d0*d2*d4;
        let conesz  = d1*(d1+1)/2;

        let barvar0 = self.task.get_num_barvar().unwrap();
        self.task.append_barvars(vec![d1 as i32; numcone].as_slice()).unwrap();
        self.vars.reserve(numcone*conesz);
        let varidx0 = self.vars.len();
        for k in 0..numcone {
            for j in 0..d1 {
                for i in j..d1 {
                    self.vars.push(VarAtom::BarElm(barvar0 + k as i32, i*(i+1)/2+j));
                }
            }
        }

        if let Some(name) = name {
            //let mut xstrides = [0usize; N];
            let mut xshape = [0usize; N];
            xshape.iter_mut().zip(shape[0..cdim0].iter().chain(shape[cdim0+1..cdim1].iter()).chain(shape[cdim1+1..].iter())).for_each(|(t,&s)| *t = s);
            //xstrides.iter_mut().zip(xshape[0..N-2].iter()).rev().fold(1|v,(t,*s)| { *t = v; s * v});
            let mut idx = [1usize; N];
            for barj in barvar0..barvar0+numcone as i32 {
                let name = format!("{}{:?}",name,&idx[0..N-2]);
                self.task.put_barvar_name(barj, name.as_str()).unwrap();
                idx[0..N-2].iter_mut().zip(xshape).rev().fold(1,|carry,(i,d)| { *i += carry; if *i > d { *i = 1; 1 } else { 0 } } );
            }
        }

        let idxs : Vec<usize> = if conedim0 < conedim1 {
            //println!("Conedims {},{}",conedim0,conedim1);
            iproduct!(0..d0,0..d1,0..d2,0..d3,0..d4).map(|(i0,i1,i2,i3,i4)| {
                let (i1,i3) = if i3 > i1 { (i3,i1) } else { (i1,i3) };

                let baridx = i0 * d2 * d4 + i2 * d4 + i4;

                let ofs    = d1*i3+i1-i3*(i3+1)/2;
                //println!("d = {}, (i,j) = ({},{}) -> {}",d1,i1,i3,ofs);

                varidx0+baridx*conesz+ofs
            }).collect()
        }
        else {
            //println!("Conedims {},{}",conedim0,conedim1);
            iproduct!(0..d0,0..d1,0..d2,0..d3,0..d4).map(|(i0,i3,i2,i1,i4)| {
                let (i1,i3) = if i3 > i1 { (i3,i1) } else { (i1,i3) };

                let baridx = i0 * d2 * d4 + i2 * d4 + i4;
                let ofs    = d1*i3+i1 - i3*(i3+1)/2;

                varidx0+baridx*conesz+ofs
            }).collect()
        };

        //println!("PSD variable indexes = {:?}",idxs);
        Variable::new(idxs,
                      None,
                      &shape)
    }

    fn conic_variable<const N : usize>(&mut self, name : Option<&str>, dom : ConicDomain<N>) -> Variable<N> {
        let (ct,offset,shape,conedim,is_integer) = dom.dissolve();
        let n    = shape.iter().product();
        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();
        let vari = self.task.get_num_var().unwrap();

        let asubi : Vec<i64> = (acci..acci+n as i64).collect();
        let asubj : Vec<i32> = (vari..vari+n as i32).collect();
        let acof  : Vec<f64> = vec![1.0; n];

        let d0 : usize = shape[0..conedim].iter().product();
        let d1 : usize = shape[conedim];
        let d2 : usize = shape[conedim+1..].iter().product();
        let conesize = d1;
        let numcone  = d0*d2;

        let domidx = match ct {
            ConicDomainType::SVecPSDCone           => self.task.append_svec_psd_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::QuadraticCone         => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone  => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::GeometricMeanCone     => self.task.append_primal_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::DualGeometricMeanCone => self.task.append_dual_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::ExponentialCone       => self.task.append_primal_exp_cone_domain().unwrap(),
            ConicDomainType::DualExponentialCone   => self.task.append_dual_exp_cone_domain().unwrap(),
            ConicDomainType::PrimalPowerCone(ref alpha) => self.task.append_primal_power_cone_domain(conesize.try_into().unwrap(),alpha.as_slice()).unwrap(),
            ConicDomainType::DualPowerCone(ref alpha) => self.task.append_dual_power_cone_domain(conesize.try_into().unwrap(),alpha.as_slice()).unwrap(),
            ConicDomainType::Zero                  => self.task.append_rzero_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::Free                  => self.task.append_r_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::NonPositive           => self.task.append_rplus_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::NonNegative           => self.task.append_rminus_domain(conesize.try_into().unwrap()).unwrap(),
        };

        self.task.append_afes(n as i64).unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.task.put_var_bound_slice_const(vari, vari+n as i32, mosek::Boundkey::FR, 0.0, 0.0).unwrap();
        if is_integer {
            self.task.put_var_type_list((vari..vari+n as i32).collect::<Vec<i32>>().as_slice(), vec![mosek::Variabletype::TYPE_INT; n].as_slice()).unwrap();
        }
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),n as i64,afei,offset.as_slice()).unwrap();
        self.task.put_afe_f_entry_list(asubi.as_slice(),asubj.as_slice(),acof.as_slice()).unwrap();

        if let Some(name) = name {
            self.var_names(name,vari,&shape,None);
            let mut xshape = [0usize; N];
            xshape[0..conedim].copy_from_slice(&shape[0..conedim]);
            if conedim < N-1 {
                xshape[conedim..N-1].copy_from_slice(&shape[conedim+1..N]);
            }
            let mut idx = [1usize; N];
            for i in acci..acci+numcone as i64 {
                let n = format!("{}{:?}",name,&idx[0..N-1]);
                self.task.put_acc_name(i, n.as_str()).unwrap();
                idx.iter_mut().zip(xshape.iter()).rev().fold(1,|carry,(t,&d)| { *t += carry; if *t > d { *t = 1; 1 } else { 0 } });
            }
        }

        let firstvar = self.vars.len();
        self.vars.reserve(n);
        self.cons.reserve(n);

        iproduct!(0..d0,0..d1,0..d2).enumerate()
            .for_each(|(i,(i0,i1,i2))| {
                self.vars.push(VarAtom::ConicElm(vari+i as i32,self.cons.len()));
                self.cons.push(ConAtom::ConicElm{acci : acci+(i0*d2+i2) as i64, afei: afei+i as i64,accoffset : i1})
            } );

        Variable::new((firstvar..firstvar+n).collect(), None, &shape)
    }


    /// Add a ranged constraint. 
    ///
    /// # Arguments
    /// - `name` Optional variable name
    /// - `expr` Constraint expression
    /// - `dom` variable domain range, see [in_range].
    ///
    /// # Returns
    /// Ok success, return a pair of constraints. When used for getting the primal
    /// solution values, they are identical, but when getting the dual solution values, the first
    /// of the pair will fetch the dual bound corresponding to the lower bound, and the second the
    /// dual values for the upper bound.
    /// On error, return a string with the reason.
    pub fn try_ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> Result<(Constraint<N>,Constraint<N>),String> 
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>
    {
        expr.into_expr().eval_finalize(&mut self.rs, &mut self.ws, &mut self.xs).map_err(|e| e.to_string())?;
        let (eshape,ptr,_,subj,cof) = self.rs.pop_expr();
        let nelm = *ptr.last().unwrap();
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);
        let domain = dom.into_range(shape)?.dense();
      
        if domain.is_integer {
            return Err("Constraint cannt be integer".to_string());
        }
    
        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            return Err("Expression is invalid: Variable subscript out of bound for this Model".to_string());
        }

        let coni = self.task.get_num_con().unwrap();
        self.task.append_cons(i32::try_from(nelm).unwrap()).unwrap();

        if let Some(name) = name {
            Self::con_names(& mut self.task,name,coni,&shape);
        }

        self.cons.reserve(nelm*2);
        let firstcon = self.cons.len();
        (coni..coni+nelm as i32).zip(domain.lower.iter()).for_each(|(i,&c)| self.cons.push(ConAtom::Linear(i,c,mosek::Boundkey::RA,WhichLinearBound::Lower)));
        (coni..coni+nelm as i32).zip(domain.upper.iter()).for_each(|(i,&c)| self.cons.push(ConAtom::Linear(i,c,mosek::Boundkey::RA,WhichLinearBound::Upper)));

        self.cons.reserve(nelm);

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());

        if !asubj.is_empty() {
            self.task.put_a_row_slice(
                coni,coni+nelm as i32,
                &aptr[0..aptr.len()-1],
                &aptr[1..],
                asubj.as_slice(),
                acof.as_slice()).unwrap();
        }

        let lower : Vec<f64> = domain.lower.iter().zip(afix.iter()).map(|(&ofs,&b)| ofs-b).collect();
        let upper : Vec<f64> = domain.upper.iter().zip(afix.iter()).map(|(&ofs,&b)| ofs-b).collect();
        self.task.put_con_bound_slice(coni,
                                      coni+nelm as i32,
                                      vec![mosek::Boundkey::RA; nelm].as_slice(),
                                      lower.as_slice(),
                                      upper.as_slice()).unwrap();

        if ! abarsubi.is_empty() {
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

                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_bara_ij(coni+i as i32, j,&[matidx],&[1.0]).unwrap();
            }
        }

        Ok((Constraint{ idxs : (firstcon..firstcon+nelm).collect(),        shape },
            Constraint{ idxs : (firstcon+nelm..firstcon+2*nelm).collect(), shape }))
    }

    pub fn ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> (Constraint<N>,Constraint<N>)
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>
    {
        self.try_ranged_constraint(name,expr,dom).unwrap()
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
    pub fn try_constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Result<Constraint<N>,String>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,_,_,_,_) = self.rs.peek_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        dom.try_into_domain(shape)?.add_constraint(self,name)
    }

    /// Add a constraint. See [Model::try_constraint].
    ///
    /// # Returns
    /// - On success, return a N-dimensional constraint object that can be used to access
    ///   solution values.
    /// - On any failure: Panic.
    pub fn constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Constraint<N>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>
    {
        self.try_constraint(name, expr, dom).unwrap()
    }

    pub(crate) fn psd_constraint<const N : usize>(& mut self, name : Option<&str>, dom : PSDDomain<N>) -> Result<Constraint<N>,String> {
        let (shape,(conedim0,conedim1)) = dom.dissolve();
        // validate domain
       
        let conearrshape : Vec<usize> = shape.iter().enumerate().filter(|v| v.0 != conedim0 && v.0 != conedim1).map(|v| v.1).cloned().collect();
        let numcone : usize = conearrshape.iter().product();
        let conesize = shape[conedim0] * (shape[conedim0]+1) / 2;
        
        // Pop expression and validate 
        let (expr_shape,ptr,expr_sp,subj,cof) = self.rs.pop_expr();
        let nelm = ptr.len()-1;
        let nnz  = ptr.last().unwrap();
        
        // Check that expression shape matches domain shape
        if expr_shape.iter().zip(shape.iter()).any(|v| v.0 != v.1) { panic!("Mismatching shapes of expression {:?} and domain {:?}",expr_shape,&shape); }
        if expr_sp.is_some() { panic!("Constraint expression cannot be sparse") };

        if shape.iter().product::<usize>() != nelm { panic!("Mismatching expression and shape"); }
        if let Some(&j) = subj.iter().max() {
            if j >= self.vars.len() {
                panic!("Invalid subj index in evaluated expression");
            }
        }

        let strides = shape.to_strides();

        // build transpose permutation
        let mut tperm : Vec<usize> = (0..nelm).collect();
        tperm.sort_by_key(|&i| {
            let mut idx = strides.to_index(i);
            idx.swap(conedim0,conedim1);
            strides.to_linear(&idx)
        });

        let rnelm = conesize * numcone;

        let (urest,rcof) = self.xs.alloc(nnz*2+rnelm+1,nnz*2);
        let (rptr,rsubj) = urest.split_at_mut(rnelm+1);
        
        //println!("---- \n\tptr = {:?}\n\tsubj = {:?}\n\tcof = {:?}",ptr,subj,cof);

        //----------------------------------------
        // Compute number of non-zeros per element of the lower triangular part if 1/2 (E+E')
        //
        rptr[0] = 0;
        for ((idx,&p0b,&p0e,&p1b,&p1e),rp) in 
            izip!(shape.index_iterator(),
                  ptr.iter(),
                  ptr[1..].iter(),
                  ptr.permute_by(tperm.as_slice()),
                  ptr[1..].permute_by(tperm.as_slice()))
                .filter(|(index,_,_,_,_)| index[conedim0] >= index[conedim1])
                .zip(rptr[1..].iter_mut())
        {
            if idx[conedim0] == idx[conedim1] {
                *rp = p0e-p0b;
            }
            else {
                // count merged nonzeros
                *rp = merge_join_by(subj[p0b..p0e].iter(),subj[p1b..p1e].iter(), |i,j| i.cmp(j) ).count();
            }
        }
        rptr.iter_mut().fold(0,|p,ptr| { *ptr += p; *ptr });

        //----------------------------------------
        // Compute nonzeros of the lower triangular part if 1/2 (E+E')
        izip!(shape.index_iterator(),
              ptr.iter(),
              ptr[1..].iter(),
              ptr.permute_by(tperm.as_slice()),
              ptr[1..].permute_by(tperm.as_slice()))
            .filter(|(index,_,_,_,_)| index[conedim0] >= index[conedim1])
            .zip( rptr.iter().zip(rptr[1..].iter()))
            .for_each(| ((index,&p0b,&p0e,&p1b,&p1e),(&rpb,&rpe)) | {
                if index[conedim0] == index[conedim1] {
                    rsubj[rpb..rpe].copy_from_slice(&subj[p0b..p0e]);
                    rcof[rpb..rpe].copy_from_slice(&cof[p0b..p0e]);
                }
                else {
                    // count merged nonzeros
                    for (ii,rj,rc) in izip!(merge_join_by(subj[p0b..p0e].iter().zip(cof[p0b..p0e].iter()),
                                                          subj[p1b..p1e].iter().zip(cof[p0b..p0e].iter()), 
                                                          |i,j| i.0.cmp(j.0)),
                                            rsubj[rpb..rpe].iter_mut(),
                                            rcof[rpb..rpe].iter_mut()) {
                        match ii {
                            EitherOrBoth::Left((&j,&c)) => { *rj = j; *rc = 0.5 * c; },
                            EitherOrBoth::Right((&j,&c)) => { *rj = j; *rc = 0.5 * c; },
                            EitherOrBoth::Both((&j,&c0),(_,&c1)) => { *rj = j; *rc = 0.5*(c0 + c1); } 
                        }
                    }
                }

            });
        let rsubj = &rsubj[..*rptr.last().unwrap()];
        let rcof  = &rcof[..*rptr.last().unwrap()];
       
        //println!("psd_constraint. 1/2 E+E':\n\tptr : {:?}\n\tsubj : {:?}\n\t - {:?}",rptr,rsubj,rptr.iter().zip(rptr[1..].iter()).map(|(&b,&e)| &rsubj[b..e]).collect::<Vec<&[usize]>>());

        // now rptr, subj, cof contains the full 1/2(E'+E)
        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(rptr,rsubj,rcof,self.vars.as_slice());

        let conedim = shape[conedim0];
        let nelm : usize = conesize*numcone;

        let barvar0 = self.task.get_num_barvar().unwrap();
            
        let acc0 = self.task.get_num_acc().unwrap();
        let afe0 = self.task.get_num_afe().unwrap();
        self.task.append_barvars(vec![conedim.try_into().unwrap(); numcone].as_slice()).unwrap();
        let dom = self.task.append_rzero_domain(rnelm as i64).unwrap();
        
        self.task.append_afes(rnelm as i64).unwrap();

        // Input linear non-zeros and bounds
        let afeidxs : Vec<i64> = (afe0..afe0+rnelm as i64).collect();
        let rownumnz : Vec<i32> = aptr.iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| i32::try_from(p1-p0).unwrap()).collect();
        
        self.task.put_afe_f_row_list(&afeidxs, &rownumnz, &aptr, &asubj, &acof).unwrap();

        let dim : i32 = shape[conedim0].try_into().unwrap();
        let mxs : Vec<i64> = (0..dim).flat_map(|i| std::iter::repeat(i).zip(0..i+1))
            .map(|(i,j)| self.task.append_sparse_sym_mat(dim,&[i],&[j],&[1.0]).unwrap())
            .collect::<Vec<i64>>();

        self.task.append_acc_seq(dom, afe0, &afix).unwrap();
        //self.task.put_con_bound_slice(con0,con0+i32::try_from(rnelm).unwrap(),&vec![mosek::Boundkey::FX; nelm],&afix,&afix).unwrap();

        if ! abarsubi.is_empty() {
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

                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_afe_barf_entry(afe0+i as i64, j,&[matidx],&[1.0]).unwrap();
            }
        }

        
        // put PSD slack variable terms and constraint mappings


        let mut xstride = [0usize;N];
        izip!(0..N,xstride.iter_mut(),shape.iter()).rev()
            .fold(1usize, |c,(i,s,&d)| 
                  if i == conedim0 || i == conedim1 {
                      *s = 0;
                      c
                  }
                  else {
                      *s = c; 
                      c * d
                  });
        self.cons.reserve(nelm);
        let firstcon = self.cons.len();
        shape.index_iterator()
            .filter(|index| index[conedim0] >= index[conedim1])
            .zip(0..rnelm as i64)
            .for_each(| (index,coni) | {
                let barvari : i32 = barvar0 + i32::try_from(xstride.iter().zip(index.iter()).map(|(&a,&b)| a * b).sum::<usize>()).unwrap();
                let ii = index[conedim0];
                let jj = index[conedim1];
                let mi = mxs[ii*(ii+1)/2+jj];
                self.task.put_afe_barf_entry(afe0+coni,barvari,&[mi], &[-1.0]).unwrap();
                self.cons.push(ConAtom::BarElm{acci : acc0, accoffset : coni, afei : afe0+coni, barj : barvari, offset : ii*(ii+1)/2+jj});
            });

        if let Some(name) = name {
            self.task.put_acc_name(acc0, name).unwrap();
//            shape.index_iterator()
//                .filter(|index| index[conedim0] >= index[conedim1])
//                .zip(con0..con0+nelmi32)
//                .for_each(| (index,coni) | {
//                    self.task.put_con_name(coni,format!("{}{:?}",name,index).as_str()).unwrap();
//                });
//            let mut xshape = [1usize;N]; (0..).zip(shape.iter()).filter_map(|(i,s)| if i == conedim0 || i == conedim1 { None } else { Some(s) }).zip(xshape.iter_mut()).for_each(|(a,b)| *b = *a);
//            xshape.index_iterator()
//                .zip(barvar0..barvar0+i32::try_from(numcone).unwrap())
//                .for_each(| (index,barvari) | {
//                    self.task.put_barvar_name(barvari,format!("{}{:?}",name,&index[..N-2]).as_str()).unwrap();
//                });
        }

        
        // compute the mapping


        let mut xstrides = [0usize;N]; izip!(0..N,xstrides.iter_mut(),shape.iter()).rev()
            .fold(1usize,|c,(i,s,&d)| {
                if i == conedim0 { *s = c; c*d*(d+1)/2 }
                else if i == conedim1 { *s = 0; c }
                else { *s = c; c*d }
            });

        let mut idxs = vec![0usize; shape.iter().product()];
        idxs.iter_mut().zip(shape.index_iterator())
            .filter(|(_,index)| index[conedim0] >= index[conedim1])
            .zip(firstcon..firstcon+rnelm)
            .for_each(|((ix,_),coni)| { *ix = coni; } );
        let idxs_ = idxs.clone();
        izip!(idxs.iter_mut(),
              idxs_.permute_by(&tperm),
              shape.index_iterator())
            .filter(|(_,_,index)| index[conedim0] < index[conedim1])
            .for_each(|(t,&s,_)| { *t = s; })
            ;

        Ok(Constraint{
            idxs,
            shape,
        })
    }


    pub(crate) fn linear_constraint<const N : usize>(& mut self,
                                          name : Option<&str>,
                                          dom  : LinearDomain<N>) -> Result<Constraint<N>,String> {
        let (dt,b,_,shape,_) =  dom.into_dense().dissolve();
        let (_,ptr,_,subj,cof) = self.rs.pop_expr();

        // let nnz = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            return Err("Expression is invalid: Variable subscript out of bound for this Model".to_string());
        }

        let coni = self.task.get_num_con().unwrap();
        self.task.append_cons(nelm.try_into().unwrap()).unwrap();

        if let Some(name) = name {
            Self::con_names(& mut self.task,name,coni,&shape);
        }
        
        let bk = match dt {
            LinearDomainType::NonNegative => mosek::Boundkey::LO,
            LinearDomainType::NonPositive => mosek::Boundkey::UP,
            LinearDomainType::Zero        => mosek::Boundkey::FX,
            LinearDomainType::Free        => mosek::Boundkey::FR
        };

        self.cons.reserve(nelm);
        let firstcon = self.cons.len();
        (coni..coni+nelm as i32).zip(b.iter()).for_each(|(i,&c)| self.cons.push(ConAtom::Linear(i,c,bk,WhichLinearBound::Both)));

        self.cons.reserve(nelm);

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());

        if !asubj.is_empty() {
            self.task.put_a_row_slice(
                coni,coni+nelm as i32,
                &aptr[0..aptr.len()-1],
                &aptr[1..],
                asubj.as_slice(),
                acof.as_slice()).unwrap();
        }

        let rhs : Vec<f64> = b.iter().zip(afix.iter()).map(|(&ofs,&b)| ofs-b).collect();
        self.task.put_con_bound_slice(coni,
                                      coni+nelm as i32,
                                      vec![bk; nelm].as_slice(),
                                      rhs.as_slice(),
                                      rhs.as_slice()).unwrap();

        if ! abarsubi.is_empty() {
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

                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_bara_ij(coni+i as i32, j,&[matidx],&[1.0]).unwrap();
            }
        }

        Ok(Constraint{
            idxs : (firstcon..firstcon+nelm).collect(),
            shape,
        })
    }

    pub(crate) fn conic_constraint<const N : usize>(& mut self,
                        name : Option<&str>,
                        dom  : ConicDomain<N>) -> Result<Constraint<N>,String> {
        let (dt,offset,shape,conedim,_) = dom.dissolve();
        let (_,ptr,_,subj,cof) = self.rs.pop_expr();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            return Err("Expression is invalid: Variable subscript out of bound for this Model".to_string());
        }

        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());
        let conesize = shape[conedim];
        let numcone  = shape.iter().product::<usize>() / conesize;

        let domidx = match dt {
            ConicDomainType::NonNegative           => self.task.append_rplus_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::NonPositive           => self.task.append_rminus_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::Free                  => self.task.append_r_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::Zero                  => self.task.append_rzero_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::SVecPSDCone           => self.task.append_svec_psd_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::QuadraticCone         => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone  => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::GeometricMeanCone     => self.task.append_primal_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::DualGeometricMeanCone => self.task.append_dual_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::ExponentialCone       => self.task.append_primal_exp_cone_domain().unwrap(),
            ConicDomainType::DualExponentialCone   => self.task.append_dual_exp_cone_domain().unwrap(),
            ConicDomainType::PrimalPowerCone(ref alpha) => self.task.append_primal_power_cone_domain(conesize.try_into().unwrap(), alpha.as_slice()).unwrap(),
            ConicDomainType::DualPowerCone(ref alpha) => self.task.append_dual_power_cone_domain(conesize.try_into().unwrap(), alpha.as_slice()).unwrap(),
        };

        self.task.append_afes(nelm as i64).unwrap();
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),
                                  nelm as i64,
                                  afei,
                                  offset.as_slice()).unwrap();
        let d0 : usize = shape[0..conedim].iter().product();
        let d1 : usize = shape[conedim];
        let d2 : usize = shape[conedim+1..].iter().product();
        let afeidxs : Vec<i64> = iproduct!(0..d0,0..d2,0..d1)
            .map(|(i0,i2,i1)| afei + (i0*d1*d2 + i1*d2 + i2) as i64)
            .collect();

        if let Some(name) = name {
            let _numcone = d0*d2;
            let mut xshape = [1usize; N]; 
            xshape[0..conedim].copy_from_slice(&shape[0..conedim]);
            if conedim < N-1 {
                xshape[conedim+1..N-1].copy_from_slice(&shape[conedim+1..N]);
            }
            let mut idx = [1usize; N];
            for i in acci..acci+(d0*d2) as i64 {                
                let n = format!("{}{:?}",name,&idx[0..N-1]);
                xshape.iter().zip(idx.iter_mut()).rev().fold(1,|carry,(&d,i)| { *i += carry; if *i > d { *i = 1; 1 } else { 0 } } );
                self.task.put_acc_name(i,n.as_str()).unwrap();
            } 
        }

        if asubj.len() > 0 {
            self.task.put_afe_f_row_list(afeidxs.as_slice(),
                                         aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                         &aptr[..nelm],
                                         asubj.as_slice(),
                                         acof.as_slice()).unwrap();
        }
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();
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

                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
            }
        }

        let coni = self.cons.len();
        self.cons.reserve(nelm);
        iproduct!(0..d0,0..d1,0..d2).enumerate() 
            .for_each(|(k,(i0,i1,i2))| self.cons.push(ConAtom::ConicElm{acci:acci+(i0*d2+i2) as i64, afei : afei+k as i64,accoffset : i1}));

        Ok(Constraint{
            idxs : (coni..coni+nelm).collect(),
            shape 
        })
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
    pub fn try_disjunction<D>(& mut self, name : Option<&str>, mut terms : D) -> Result<Disjunction,String> where D : disjunction::DisjunctionTrait {
        let nexprs = terms.eval(self.vars.len(), &mut self.rs, &mut self.ws, &mut self.xs)?;
        let mut exprs = self.rs.pop_exprs(nexprs);
        let mut term_ptr = Vec::new(); term_ptr.push(0);
        let mut element_dom = Vec::new();
        let mut element_ptr = Vec::new();element_ptr.push(0);
        let mut element_afei = Vec::new();
        let mut element_b = Vec::new();

        terms.append_disjunction_data(&mut self.task, self.vars.as_slice(), &mut exprs, &mut term_ptr, &mut element_dom, &mut element_ptr, &mut element_afei, &mut element_b);

        let term_size : Vec<i64> = term_ptr.iter().zip(term_ptr[1..].iter()).map(|(a,b)| (b-a) as i64).collect();
        let djci = self.task.get_num_djc().unwrap();
        self.task.append_djcs(1).unwrap();
        if let Some(name) = name { self.task.put_djc_name(djci,name).unwrap(); }
        self.task.put_djc(djci, 
                          element_dom.as_slice(),
                          element_afei.as_slice(),
                          element_b.as_slice(),
                          term_size.as_slice()).unwrap();
        
        Ok(Disjunction{ index : djci })
    }
    
    pub fn disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Disjunction where D : disjunction::DisjunctionTrait {
        self.try_disjunction(name, terms).unwrap()
    }

    fn update_internal(&mut self, idxs : &[usize]) {
        let (_,ptr,_,subj,cof) = self.rs.pop_expr();
        if let Some(maxidx) = idxs.iter().max() {
            if *maxidx >= self.cons.len() {
                panic!("Invalid constraint indexes: Out of bounds");
            }

            // check that there are not dups.
            let mut perm = (0..idxs.len()).collect::<Vec<usize>>();
            perm.sort_by_key(|&i| unsafe{ *idxs.get_unchecked(i) });
            if idxs.permute_by(&perm).zip(idxs.permute_by(&perm[1..])).any(|(&a,&b)| a == b) {
                panic!("Invalid constraint contains duplicatesindexes: Out of bounds");
            }

            let (nconic,nnzconic,nbar,nnzbar,nlin,nnzlin) = izip!(self.cons.permute_by(idxs),ptr.iter(),ptr[1..].iter())
                .fold((0,0,0,0,0,0),
                      | (nconic,nnzconic,nbar,nnzbar,nlin,nnzlin), (c,&p0,&p1) | 
                          match c {
                              ConAtom::ConicElm{..} => (nconic+1,nnzconic+p1-p0,nbar,nnzbar,nlin,nnzlin),
                              ConAtom::BarElm{..} => (nconic,nnzconic,nbar+1,nnzbar+p1-p0,nlin,nnzlin),
                              ConAtom::Linear(..) => (nconic,nnzconic,nbar,nnzbar,nlin+1,nnzlin+p1-p0),
                          });
           
            let mut conic_ptr  = Vec::new();
            let mut conic_subj = Vec::new();
            let mut conic_cof  = Vec::new();
            let mut conic_afe  = Vec::new();
            let mut lin_ptr    = Vec::new();
            let mut lin_subj   = Vec::new();
            let mut lin_cof    = Vec::new();
            let mut lin_subi   = Vec::new();
            let mut lin_rhs    = Vec::new();
            let mut lin_bk     = Vec::new();

            if nlin == 0 || nconic == 0 {
                if nlin == 0 {
                    conic_ptr.extend_from_slice(ptr);
                    conic_subj.extend_from_slice(subj);
                    conic_cof.extend_from_slice(cof);
                    conic_afe.reserve(nconic+nbar);
                }
                else {
                    lin_ptr.extend_from_slice(ptr);
                    lin_subj.extend_from_slice(subj);
                    lin_cof.extend_from_slice(cof);
                    lin_subi.reserve(nlin);
                    lin_rhs.reserve(nlin);
                    lin_bk.reserve(nlin);
                }
                for c in self.cons.permute_by(idxs) {
                    match c {
                      ConAtom::ConicElm{afei,..} => conic_afe.push(*afei),
                      ConAtom::BarElm{afei,..} => conic_afe.push(*afei),
                      ConAtom::Linear(coni,rhs,bk,_) => { lin_subi.push(*coni); lin_rhs.push(*rhs); lin_bk.push(*bk); },
                     }
                }
            }
            else {
                conic_ptr.reserve(nconic+nbar+1); conic_ptr.push(0usize);
                conic_subj.reserve(nnzconic+nnzbar);
                conic_cof.reserve(nnzconic+nnzbar);
                conic_afe.reserve(nnzconic+nnzbar);
                lin_ptr.reserve(nlin+1);   lin_ptr.push(0usize);
                lin_subj.reserve(nnzlin);
                lin_cof.reserve(nnzlin);
                lin_subi.reserve(nlin);
                lin_rhs.reserve(nlin);
                lin_bk.reserve(nlin);

                for (c,jj,cc) in izip!(self.cons.permute_by(idxs),
                                       subj.chunks_ptr(ptr),
                                       cof.chunks_ptr(ptr)) {
                    match c {
                      ConAtom::ConicElm{afei,..} => {
                          conic_ptr.push(jj.len());
                          conic_subj.extend_from_slice(jj);
                          conic_cof.extend_from_slice(cc);
                          conic_afe.push(*afei);
                      }
                      ConAtom::BarElm{afei,..}   => {
                          conic_ptr.push(jj.len());
                          conic_subj.extend_from_slice(jj);
                          conic_cof.extend_from_slice(cc);
                          conic_afe.push(*afei);
                      }
                      ConAtom::Linear(coni,rhs,bk,_)       => {
                          lin_ptr.push(jj.len());
                          lin_subj.extend_from_slice(jj);
                          lin_cof.extend_from_slice(cc);
                          lin_subi.push(*coni);
                          lin_rhs.push(*rhs);
                          lin_bk.push(*bk);
                      }
                    }
                }
                conic_ptr.iter_mut().fold(0,|c,p| { *p += c; *p });
                lin_ptr.iter_mut().fold(0,|c,p| { *p += c; *p });
            }
            if nlin > 0 {
                let (asubj,
                     acof,
                     aptr,
                     afix,
                     abarsubi,
                     abarsubj,
                     abarsubk,
                     abarsubl,
                     abarcof) = split_expr(&lin_ptr,&lin_subj,&lin_cof,self.vars.as_slice());

                if !asubj.is_empty() {
                    self.task.put_a_row_list(
                        lin_subi.as_slice(),
                        &aptr[0..aptr.len()-1],
                        &aptr[1..],
                        asubj.as_slice(),
                        acof.as_slice()).unwrap();
                }

                lin_rhs.iter_mut().zip(afix.iter()).for_each(|(r,&f)| *r -= f);
                self.task.put_con_bound_list(lin_subi.as_slice(),
                                             lin_bk.as_slice(),
                                             lin_rhs.as_slice(),
                                             lin_rhs.as_slice()).unwrap();

                if ! abarsubi.is_empty() {
                    izip!(0..,
                          abarsubi.iter(),
                          abarsubj.iter())
                        .chain(std::iter::once((abarsubi.len(),&i64::MAX,&i32::MAX)))
                        .scan((0usize,i64::MAX,i32::MAX),|(p0,previ,prevj),(k,i,j)|
                            if *previ != *i || *prevj != *j {
                                let oldp0 = *p0;
                                *p0 = k;
                                Some((*previ,*prevj,oldp0,k))
                            }
                            else {
                                Some((*i,*j,*p0,*p0))
                            })
                        .filter(|(_,_,p0,p1)| p0 != p1)
                        .for_each(|(_i,j,p0,p1)| {
                       
                        let subk = &abarsubk[p0..p1];
                        let subl = &abarsubl[p0..p1];
                        let cof  = &abarcof[p0..p1];
                        let afei = conic_afe[abarsubi[p0] as usize];

                        let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                        let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                        self.task.put_afe_barf_entry(afei, j,&[matidx],&[1.0]).unwrap();
                    });
                }
            }

            // change conic elements
            if nconic > 0 {
                let (asubj,
                     acof,
                     aptr,
                     afix,
                     abarsubi,
                     abarsubj,
                     abarsubk,
                     abarsubl,
                     abarcof) = split_expr(&conic_ptr,&conic_subj,&conic_cof,self.vars.as_slice());
                let nelm = aptr.len()-1;

                if ! asubj.is_empty() {
                    self.task.put_afe_f_row_list(conic_afe.as_slice(),
                                                 aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                                 &aptr[..nelm],
                                                 asubj.as_slice(),
                                                 acof.as_slice()).unwrap();
                }
                self.task.put_afe_g_list(conic_afe.as_slice(),afix.as_slice()).unwrap();
                if ! abarsubi.is_empty() {
                    self.task.empty_afe_barf_row_list(conic_afe.as_slice()).unwrap();
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

                        let afei = conic_afe[i as usize];
                        p0 = p;

                        let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                        let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                        self.task.put_afe_barf_entry(afei,j,&[matidx],&[1.0]).unwrap();
                    }
                }
            }
        }
    }

    /// Update the expression of a constraint in the Model.
    pub fn update<const N : usize, E>(&mut self, item : &Constraint<N>, e : E) where E : expr::IntoExpr<N> {
        e.into_expr().eval_finalize(&mut self.rs, &mut self.ws, &mut self.xs).unwrap();
        {
            let (shape,_,_,_,_) = self.rs.peek_expr();
            if shape.iter().zip(item.shape.iter()).any(|(&a,&b)| a != b) {
                panic!("Shape of constraint ({:?}) does not match shape of expression ({:?})",item.shape,shape);
            }
        }
        self.update_internal(&item.idxs);
    }

    //======================================================
    // Objective


    fn set_objective(& mut self, name : Option<&str>, sense : Sense) {
        let (_shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ptr.len()-1 > 1 { panic!("Objective expressions may only contain one element"); }

        if let Some(name) = name { self.task.put_obj_name(name).unwrap(); }
        else { self.task.put_obj_name("").unwrap(); }
        let (csubj,
             ccof,
             _cptr,
             cfix,
             _barsubi,
             barsubj,
             barsubk,
             barsubl,
             barcof) = split_expr(ptr,subj,cof,self.vars.as_slice());

        let numvar : usize = self.task.get_num_var().unwrap().try_into().unwrap();

        if let Some(&j) = csubj.iter().min() { if j < 0 { panic!("Invalid expression") } }
        if let Some(&j) = csubj.iter().max() { if j as usize >= numvar { panic!("Invalid expression") } }

        let mut c = vec![0.0; numvar];
        csubj.iter().zip(ccof.iter()).for_each(|(&j,&v)| unsafe{ * c.get_unchecked_mut(j as usize) = v });
        let csubj : Vec<i32> = (0i32..numvar as i32).collect();
        self.task.put_c_list(csubj.as_slice(),
                             c.as_slice()).unwrap();
        self.task.put_cfix(cfix[0]).unwrap();

        self.task.put_barc_block_triplet(barsubj.as_slice(),barsubk.as_slice(),barsubl.as_slice(),barcof.as_slice()).unwrap();

        match sense {
            Sense::Minimize => self.task.put_obj_sense(mosek::Objsense::MINIMIZE).unwrap(),
            Sense::Maximize => self.task.put_obj_sense(mosek::Objsense::MAXIMIZE).unwrap()
        }
    }

    /// Set the objective.
    ///
    /// Arguments:
    /// - `name` Optional objective name
    /// - `sense` Objective sense
    /// - `expr` Objective expression, this must contain exactly one
    ///   element. The shape is otherwise ignored.
    pub fn objective<E : expr::IntoExpr<0>>(& mut self,
                                             name  : Option<&str>,
                                             sense : Sense,
                                             expr  : E) {
        expr.into().eval_finalize(& mut self.rs,& mut self.ws, & mut self.xs).unwrap();
        self.set_objective(name,sense);
    }

    //======================================================
    // Optimize

    /// Set a parameter in the underlying task object. 
    ///
    /// # Arguments
    /// - `parname` The name is the full name as listed in the MOSEK C manual, that is `MSK_IPAR_...`.
    /// - `parval` A parameter value. This can be an integer, a floating point value, string
    ///   value or a string representing an integer or float value, or a string representing a
    ///   symbolic constant value (again, the full value as listed in the MOSEK C manual).
    pub fn set_parameter<T>(& mut self, parname : &str, parval : T) 
        where T : SolverParameterValue {
        parval.set(parname,self);
    }

    /// Set a double parameter in the underlying task object.
    ///
    /// # Arguments
    /// - `parname` The name is the full name as listed in the MOSEK C manual, that is `MSK_DPAR_...`.
    /// - `parval` Parameter value 
    pub fn set_double_parameter(&mut self, parname : &str, parval : f64) {
        self.task.put_na_dou_param(parname, parval).unwrap();
    }

    /// Set a integer parameter in the underlying task object.
    ///
    /// # Arguments
    /// - `parname` The name is the full name as listed in the MOSEK C manual, that is `MSK_IPAR_...`.
    /// - `parval` Parameter value 
    pub fn set_int_parameter(&mut self, parname : &str, parval : i32) {
        self.task.put_na_int_param(parname, parval).unwrap();
    }

    /// Set a double parameter in the underlying task object.
    ///
    /// # Arguments
    /// - `parname` The name is the full name as listed in the MOSEK C manual, that is `MSK_SPAR_...`.
    /// - `parval` Parameter value 
    pub fn set_str_parameter(&mut self, parname : &str, parval : &str) {
        self.task.put_na_str_param(parname, parval).unwrap();
    }

    pub fn put_optserver(&mut self, hostname : &str, access_token : Option<&str>) {
        self.optserver_host = Some((hostname.to_string(),access_token.map(|v| v.to_string())));
    }

    pub fn clear_optserver(&mut self) {
        self.optserver_host = None;
    }

    /// Solve the problem and extract the solution.
    ///
    /// This will fail if the optimizer fails with an error. Not producing a solution (stalling or
    /// otherwise failing), producing a non-optimal solution or a certificate of infeasibility 
    /// is *not* an error.
    pub fn solve(& mut self) {
        self.task.put_int_param(mosek::Iparam::REMOVE_UNUSED_SOLUTIONS, 1).unwrap();
        if let Some((hostname,accesstoken)) = self.optserver_host.as_ref() {
            self.task.optimize_rmt(hostname.as_str(), accesstoken.as_ref().map(|v| v.as_str()).unwrap_or("")).unwrap();
        }
        else {
            self.task.optimize().unwrap();
        }

        let numvar = self.task.get_num_var().unwrap() as usize;
        let numcon = self.task.get_num_con().unwrap() as usize;
        let numacc = self.task.get_num_acc().unwrap() as usize;
        let numaccelm = self.task.get_acc_n_tot().unwrap() as usize;
        let numbarvar = self.task.get_num_barvar().unwrap() as usize;
        let numbarvarelm : usize = (0..numbarvar).map(|j| self.task.get_len_barvar_j(j as i32).unwrap() as usize).sum();

        let mut xx = vec![0.0; numvar];
        let mut slx = vec![0.0; numvar];
        let mut sux = vec![0.0; numvar];
        let mut xc = vec![0.0; numcon];
        let mut y = vec![0.0; numcon];
        let mut slc = vec![0.0; numcon];
        let mut suc = vec![0.0; numcon];
        let mut accx = vec![0.0; numaccelm];
        let mut doty = vec![0.0; numaccelm];
        let mut barx = vec![0.0; numbarvarelm];
        let mut bars = vec![0.0; numbarvarelm];

        let dimbarvar : Vec<usize> = (0..numbarvar).map(|j| self.task.get_dim_barvar_j(j as i32).unwrap() as usize).collect();
        let accptr    : Vec<usize> = once(0usize).chain((0..numacc)
                                                        .map(|i| self.task.get_acc_n(i as i64).unwrap() as usize)
                                                        .scan(0,|p,n| { *p += n; Some(*p) })).collect();
        let barvarptr : Vec<usize> = once(0usize).chain((0..numbarvar)
                                                        .map(|j| self.task.get_len_barvar_j(j as i32).unwrap() as usize)
                                                        .scan(0,|p,n| { *p += n; Some(*p) })).collect();

        // extract solutions
        for &whichsol in [mosek::Soltype::BAS,
                          mosek::Soltype::ITR,
                          mosek::Soltype::ITG].iter() {
            let sol = match whichsol {
                mosek::Soltype::BAS => & mut self.sol_bas,
                mosek::Soltype::ITR => & mut self.sol_itr,
                mosek::Soltype::ITG => & mut self.sol_itg,
                _ => & mut self.sol_itr
            };
            if ! self.task.solution_def(whichsol).unwrap() {
                sol.primal.status = SolutionStatus::Undefined;
                sol.dual.status   = SolutionStatus::Undefined;
            }
            else {
                let (psta,dsta) = split_sol_sta(whichsol,self.task.get_sol_sta(whichsol).unwrap());
                sol.primal.status = psta;
                sol.dual.status   = dsta;

                if let SolutionStatus::Undefined = psta {}
                else {
                    sol.primal.obj = self.task.get_primal_obj(whichsol).unwrap_or(0.0);
                    sol.primal.resize(self.vars.len(),self.cons.len());
                    self.task.get_xx(whichsol,xx.as_mut_slice()).unwrap();
                    self.task.get_xc(whichsol,xc.as_mut_slice()).unwrap();
                    if numbarvar > 0 { self.task.get_barx_slice(whichsol,0,numbarvar as i32,barx.len() as i64,barx.as_mut_slice()).unwrap(); }
                    if numacc > 0 { self.task.evaluate_accs(whichsol,accx.as_mut_slice()).unwrap(); }

                    self.vars[1..].iter().zip(sol.primal.var[1..].iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            VarAtom::Linear(j,_) => xx[j as usize],
                            VarAtom::BarElm(j,ofs) => barx[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                            VarAtom::ConicElm(j,_coni) => xx[j as usize]
                        };
                    });
                    self.cons.iter().zip(sol.primal.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm{acci,accoffset,..}=> { 
                                accx[accptr[acci as usize]+accoffset]
                            },
                            ConAtom::Linear(i,..) => xc[i as usize],
                            ConAtom::BarElm{barj:j,offset:ofs,..} => barx[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                        };
                    });
                }

                if let SolutionStatus::Undefined = dsta {}
                else {
                    sol.dual.obj = self.task.get_dual_obj(whichsol).unwrap();
                    sol.dual.resize(self.vars.len(),self.cons.len());
                    self.task.get_slx(whichsol,slx.as_mut_slice()).unwrap();
                    self.task.get_sux(whichsol,sux.as_mut_slice()).unwrap();
                    self.task.get_slc(whichsol,slc.as_mut_slice()).unwrap();
                    self.task.get_suc(whichsol,suc.as_mut_slice()).unwrap();
                    self.task.get_y(whichsol,y.as_mut_slice()).unwrap();
                    if numbarvar > 0 { self.task.get_bars_slice(whichsol,0,numbarvar as i32,bars.len() as i64,bars.as_mut_slice()).unwrap(); }
                    if numacc > 0 { self.task.get_acc_dot_y_s(whichsol,doty.as_mut_slice()).unwrap(); }

                    self.vars[1..].iter().zip(sol.dual.var.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            VarAtom::Linear(j,which) => 
                                match which {
                                    WhichLinearBound::Both  => slx[j as usize] - sux[j as usize],
                                    WhichLinearBound::Lower => slx[j as usize],
                                    WhichLinearBound::Upper => - sux[j as usize],
                                },
                            VarAtom::BarElm(j,ofs) => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                            VarAtom::ConicElm(_j,coni) => {
                                match self.cons[coni] {
                                    ConAtom::ConicElm{acci,accoffset:ofs,..} => doty[accptr[acci as usize]+ofs],
                                    ConAtom::Linear(i,_,_,which) => 
                                        match which {
                                            WhichLinearBound::Both  => y[i as usize],
                                            WhichLinearBound::Lower => slc[i as usize],
                                            WhichLinearBound::Upper => - suc[i as usize],
                                        },
                                    ConAtom::BarElm{barj:j,offset:ofs,..} => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                                }
                            }
                        };
                    });
                    self.cons.iter().zip(sol.dual.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm{acci,accoffset:ofs,..} => doty[accptr[acci as usize]+ofs],
                            ConAtom::Linear(i,_,_,which) => 
                                match which {
                                    WhichLinearBound::Both  => y[i as usize],
                                    WhichLinearBound::Lower => slc[i as usize],
                                    WhichLinearBound::Upper => -suc[i as usize],
                                },
                            ConAtom::BarElm{barj:j,offset:ofs,..} => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                        };
                    });
                }
            }
        }
    }
    //======================================================
    // Solutions

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
    pub(crate) fn primal_con_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
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
    pub(crate) fn dual_con_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
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
    pub fn primal_objective(&self, solid : SolutionType) -> Option<f64> { 
        if let Some(sol) = self.select_sol(solid) {
            Some(sol.primal.obj)
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
    pub fn primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.primal(self,solid) }
    
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
    pub fn sparse_primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String> { item.sparse_primal(self,solid) }

    /// Get dual solution values for an item
    ///
    /// Returns: If solution item is defined, return the solution, otherwise a n error message.
    pub fn dual_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.dual(self,solid) }

    /// Get primal solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn primal_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }

    /// Get dual solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn dual_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }

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

    /// Evaluate an expression in the (primal) solution. 
    ///
    /// # Arguments
    /// - `solid` The solution in which to evaluate the expression.
    /// - `expr` The expression to evaluate.
    pub fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String>
        where 
            E : IntoExpr<N>     
{
        expr.into_expr().eval(& mut self.rs,&mut self.ws,&mut self.xs).map_err(|e| format!("{:?}",e))?;
        let mut shape = [0usize; N];
        let (val,sp) = self.evaluate_primal_internal(solid, &mut shape)?;
        self.rs.clear();
        NDArray::new(shape,sp,val)
    }
} // impl Model


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
