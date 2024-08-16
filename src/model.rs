use itertools::{merge_join_by, EitherOrBoth};
use itertools::{iproduct, izip};
use std::{iter::once, path::Path};
use crate::expr;
use crate::utils::*;
use crate::domain::*;
use crate::variable::*;
use crate::WorkStack;

/// Objective sense
#[derive(Clone,Copy)]
pub enum Sense {
    Maximize,
    Minimize
}

#[derive(Clone,Copy,Debug)]
enum VarAtom {
    // Task variable index
    Linear(i32),
    // Task bar element (barj,k,l)
    BarElm(i32,usize),
    // Conic variable (j,offset)
    ConicElm(i32,usize)
}
#[allow(unused)]
#[derive(Clone,Copy,Debug)]
enum ConAtom {
    // Conic constraint element (acci, offset)
    ConicElm(i64,usize), 
    BarElm(i32,i32,usize),
    Linear(i32)
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
    con    : Vec<f64>
}

impl SolutionPart {
    fn new(numvar : usize, numcon : usize) -> SolutionPart { SolutionPart{status : SolutionStatus::Unknown, var : vec![0.0; numvar], con : vec![0.0; numcon] } }
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

//======================================================
// Domain
//======================================================

/// Represents something that can be used as a domain for a constraint.
pub trait ConDomainTrait<const N : usize> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N>;
}
/// Represents something that can be used as a domain for a variable.
pub trait VarDomainTrait<const N : usize> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<N>;
}

/// Implement LinearDomain as variable domain
impl<const N : usize> VarDomainTrait<N> for LinearDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<N> {
        m.linear_variable(name,self)
    }
}
/// Implement LinearDomain as constraint domain
impl<const N : usize> ConDomainTrait<N> for LinearDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.linear_constraint(name,self)
    }

}

/// Implement ConicDomain as a variable domain
impl<const N : usize> VarDomainTrait<N> for ConicDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<N> {
        m.conic_variable(name,self)
    }
}
/// Implement ConicDomain as a constraint domain
impl<const N : usize> ConDomainTrait<N> for ConicDomain<N> {
    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.conic_constraint(name,self)
    }
}

/// Implement a fixed-size integer array as domain for variable, meaning unbounded with the array
/// as shape.
impl<const N : usize> VarDomainTrait<N> for &[usize;N] {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<N> {
        m.free_variable(name,self)
    }
}

/// Implement a fixed-size integer array as domain for constraint, meaning unbounded with the array
/// as shape.
impl<const N : usize> ConDomainTrait<N> for &[usize;N] {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:LinearDomainOfsType::Scalar(0.0),
                                shape:*self,
                                sp:None,
                                is_integer: false})
    }
}

/// Implement integer as domain for variable, producing a vector variable if the given size.
impl VarDomainTrait<1> for usize {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<1> {
        m.free_variable(name,&[self])
    }
}
/// Implement integer as domain for constraint, producing a vector variable if the given size.
impl ConDomainTrait<1> for usize {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<1> {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:LinearDomainOfsType::Scalar(0.0),
                                shape:[self],
                                sp:None,
                                is_integer:false})
    }
}
/// Implements PSD domain for variables.
impl<const N : usize> VarDomainTrait<N> for PSDDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable<N> {
        m.psd_variable(name,self)
    }
}

impl<const N : usize> ConDomainTrait<N> for PSDDomain<N> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint<N> {
        m.psd_constraint(name,self)
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
pub struct Model {
    /// The MOSEK task
    task : mosek::TaskCB,
    /// Vector of scalar variable atoms
    vars : Vec<VarAtom>,
    /// Vector of scalar constraint atoms
    cons : Vec<ConAtom>,

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

/// A Constraint object is a wrapper around an array of constraint
/// indexes and a shape. Note that constraint objects are never sparse.
#[derive(Clone)]
pub struct Constraint<const N : usize> {
    idxs     : Vec<usize>,
    shape    : [usize; N]
}

impl<const N : usize> Constraint<N> {
    pub fn index<I>(&self, idx : I) -> I::Output where I : ModelItemIndex<Self>, Self:Sized {
        idx.index(self)
    }
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
    /// Create new Model object
    ///
    /// # Arguments
    /// - `name` An optional name
    /// # Returns
    /// An empty model.
    pub fn new(name : Option<&str>) -> Model {
        let mut task = mosek::Task::new().unwrap().with_callbacks();
        if let Some(name) = name { task.put_task_name(name).unwrap() };
        task.put_int_param(mosek::Iparam::PTF_WRITE_SOLUTIONS, 1).unwrap();
        Model{
            task,
            vars    : vec![VarAtom::Linear(-1)],
            cons    : Vec::new(),
            sol_bas : Solution::new(),
            sol_itr : Solution::new(),
            sol_itg : Solution::new(),
            rs      : WorkStack::new(0),
            ws      : WorkStack::new(0),
            xs      : WorkStack::new(0)
        }
    }

    /// Attach a log printer callback to the Model.
    ///
    /// # Arguments
    /// - `func` A function that will be called with strings from the log. Individual lines may be
    ///   written in multiple chunks to there is no guarantee that the strings will end with a
    ///   newline.
    pub fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str) {
        self.task.put_stream_callback(mosek::Streamtype::LOG,func).unwrap();
    }

    /// Write problem to a file. The filename extension determines the file format to use. If the
    /// file extension is not recognized, the MPS format is used.
    pub fn write_problem<P>(&self, filename : P) where P : AsRef<Path> {
        let path = filename.as_ref();
        self.task.write_data(path.to_str().unwrap()).unwrap();
    }

    //======================================================
    // Variable interface

    /// Add a Variable
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `dom` The domain of the variable. This defines the bound
    ///   type, shape and sparsity of the variable. For sparse
    ///   variables, elements outside of the sparsity pattern are
    ///   treated as variables fixed to 0.0.
    pub fn variable<const N : usize, D>(& mut self, name : Option<&str>, dom : D) -> Variable<N> 
        where 
            D : VarDomainTrait<N> 
    {
        dom.create(self,name)
    }

    fn var_names<const N : usize>(& mut self, name : &str, first : i32, shape : &[usize;N], sp : Option<&[usize]>) {
        let mut buf = name.to_string();
        let baselen = buf.len();

        if let Some(sp) = sp {
            SparseIndexIterator::new(shape,sp)
                .enumerate()
                .for_each(|(j,index)| {
                    buf.truncate(baselen);
                    append_name_index(& mut buf,&index);
                    //println!("name is now: {}",buf);
                    self.task.put_var_name(first + j as i32,buf.as_str()).unwrap();
                });
        }
        else {
            IndexIterator::new(shape)
                .enumerate()
                .for_each(|(j,index)| {
                    buf.truncate(baselen);
                    append_name_index(&mut buf, &index);
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
                append_name_index(&mut buf, &index);
                //println!("    name = {}",buf);
                task.put_con_name(first + j as i32,buf.as_str()).unwrap();
            });
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
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j)));

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
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j)));
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
        let nd = dom.shape.len();
        let (conedim0,conedim1) = dom.conedims;
        if conedim0 == conedim1 || conedim0 >= nd || conedim1 >= nd { panic!("Invalid cone dimensions") };
        if dom.shape[conedim0] != dom.shape[conedim1] { panic!("Mismatching cone dimensions") };

        let (cdim0,cdim1) = if conedim0 < conedim1 { (conedim0,conedim1) } else { (conedim1,conedim0) };

        let d0 = dom.shape[0..cdim0].iter().product();
        let d1 = dom.shape[cdim0];
        let d2 = dom.shape[cdim0+1..cdim1].iter().product();
        let d3 = dom.shape[cdim1];
        let d4 = dom.shape[cdim1+1..].iter().product();

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
            xshape.iter_mut().zip(dom.shape[0..cdim0].iter().chain(dom.shape[cdim0+1..cdim1].iter()).chain(dom.shape[cdim1+1..].iter())).for_each(|(t,&s)| *t = s);
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
                      &dom.shape)
    }

    fn conic_variable<const N : usize>(&mut self, name : Option<&str>, dom : ConicDomain<N>) -> Variable<N> {
        let n    = dom.shape.iter().product();
        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();
        let vari = self.task.get_num_var().unwrap();

        let asubi : Vec<i64> = (acci..acci+n as i64).collect();
        let asubj : Vec<i32> = (vari..vari+n as i32).collect();
        let acof  : Vec<f64> = vec![1.0; n];

        let d0 : usize = dom.shape[0..dom.conedim].iter().product();
        let d1 : usize = dom.shape[dom.conedim];
        let d2 : usize = dom.shape[dom.conedim+1..].iter().product();
        let conesize = d1;
        let numcone  = d0*d2;

        let domidx = match dom.dt {
            ConicDomainType::SVecPSDCone           => self.task.append_svec_psd_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::QuadraticCone         => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone  => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::GeometricMeanCone     => self.task.append_primal_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::DualGeometricMeanCone => self.task.append_dual_geo_mean_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::ExponentialCone       => self.task.append_primal_exp_cone_domain().unwrap(),
            ConicDomainType::DualExponentialCone   => self.task.append_dual_exp_cone_domain().unwrap(),
            ConicDomainType::PrimalPowerCone(ref alpha) => self.task.append_primal_power_cone_domain(conesize.try_into().unwrap(),alpha.as_slice()).unwrap(),
            ConicDomainType::DualPowerCone(ref alpha) => self.task.append_dual_power_cone_domain(conesize.try_into().unwrap(),alpha.as_slice()).unwrap(),
        };

        self.task.append_afes(n as i64).unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.task.put_var_bound_slice_const(vari, vari+n as i32, mosek::Boundkey::FR, 0.0, 0.0).unwrap();
        if dom.is_integer {
            self.task.put_var_type_list((vari..vari+n as i32).collect::<Vec<i32>>().as_slice(), vec![mosek::Variabletype::TYPE_INT; n].as_slice()).unwrap();
        }
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),n as i64,afei,dom.ofs.as_slice()).unwrap();
        self.task.put_afe_f_entry_list(asubi.as_slice(),asubj.as_slice(),acof.as_slice()).unwrap();

        if let Some(name) = name {
            self.var_names(name,vari,&dom.shape,None);
            let mut xshape = [0usize; N];
            xshape[0..dom.conedim].copy_from_slice(&dom.shape[0..dom.conedim]);
            if dom.conedim < N-1 {
                xshape[dom.conedim..N-1].copy_from_slice(&dom.shape[dom.conedim+1..N]);
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
                self.cons.push(ConAtom::ConicElm(acci+(i0*d2+i2) as i64,i1))
            } );

        Variable::new((firstvar..firstvar+n).collect(),
                      None,
                      &dom.shape)
    }


    // fn parametrized_conic_variable(&mut self, _name : Option<&str>, size : usize, num : usize, ct : ParamConicDomainType, alpha : &[f64], ofs : Option<&[f64]>) -> Vec<usize> {
    //     let n = size * num;
    //     let (firstidx,firsttaskidx) = self.alloc_linear_var(n);
    //     let lasttaskidx = firsttaskidx + n as i32;

    //     let firstafeidx = self.task.get_num_afe().unwrap();
    //     self.task.append_afes(size.try_into().unwrap()).unwrap();
    //     let lastafeidx = firstafeidx + n as i64;
    //     let afeidxs : Vec<i64> = (firstafeidx..lastafeidx).collect();
    //     let varidxs : Vec<i32> = (firsttaskidx..lasttaskidx).collect();
    //     self.task.put_afe_f_entry_list(afeidxs.as_slice(),
    //                               varidxs.as_slice(),
    //                               vec![1.0; n].as_slice()).unwrap();
    //     self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
    //     let firstaccidx = self.task.get_num_acc().unwrap();
    //     let dom = match ct {
    //         ParamConicDomainType::PrimalPowerCone => self.task.append_primal_power_cone_domain(size.try_into().unwrap(), alpha).unwrap(),
    //         ParamConicDomainType::DualPowerCone   => self.task.append_dual_power_cone_domain(size.try_into().unwrap(), alpha).unwrap()
    //     };

    //     match ofs {
    //         None => self.task.append_accs_seq(vec![dom; num].as_slice(),
    //                                           n as i64,
    //                                           firstafeidx,
    //                                           vec![0.0; n].as_slice()).unwrap(),
    //         Some(offset) => self.task.append_accs_seq(vec![dom; num].as_slice(),
    //                                                   n as i64,
    //                                                   firstafeidx,
    //                                                   offset).unwrap()
    //     }

    //     (firstidx..firstidx+size).collect()
    // }

    //======================================================
    // Constraint interface


    /// Add a constraint
    ///
    /// Note that even if the domain or the expression are sparse, a constraint will always be full.
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `expr` Constraint expression. Note that the shape of the expression and the domain must match exactly.
    /// - `dom`  The domain of the constraint. This defines the bound type and shape.
    pub fn constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr : &E, dom : D) -> Constraint<N> 
        where
            E : expr::ExprTrait<N>, 
            D : ConDomainTrait<N> 
    {
        expr.eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs);
        dom.create(self,name)
    }

    fn psd_constraint<const N : usize>(& mut self, name : Option<&str>, dom : PSDDomain<N>) -> Constraint<N> {
        // validate domain
        let (conedim0,conedim1) = if dom.conedims.0 < dom.conedims.1 { (dom.conedims.0,dom.conedims.1) } else { (dom.conedims.1,dom.conedims.0) } ;
        if dom.conedims.0 >= dom.conedims.1 { panic!("Invalid cone dimension specification"); }
        if dom.conedims.1 >= N { panic!("Invalid cone dimension specification"); }
        if dom.shape[conedim0] != dom.shape[conedim1] { panic!("Invalid cone shape"); }
       
        let conearrshape : Vec<usize> = dom.shape.iter().enumerate().filter(|v| v.0 != conedim0 && v.0 != conedim1).map(|v| v.1).cloned().collect();
        let numcone : usize = conearrshape.iter().product();
        let conesize = dom.shape[conedim0] * (dom.shape[conedim0]+1) / 2;
        
        // Pop expression and validate 
        let (expr_shape,ptr,expr_sp,subj,cof) = self.rs.pop_expr();
        let nelm = ptr.len()-1;
        let nelmi32 : i32 = nelm.try_into().unwrap();
        let nnz  = ptr.last().unwrap();
        
        let shape = dom.shape;
        // Check that expression shape matches domain shape
        if expr_shape.iter().zip(shape.iter()).any(|v| v.0 != v.1) { panic!("Mismatching shapes of expression {:?} and domain {:?}",expr_shape,&shape); }
        if expr_sp.is_some() { panic!("Constraint expression cannot be sparse") };

        if shape.iter().product::<usize>() != nelm { panic!("Mismatching expression and shape"); }
        if let Some(&j) = subj.iter().max() {
            if j >= self.vars.len() {
                panic!("Invalid subj index in evaluated expression");
            }
        }

        let mut strides = [0usize; N]; strides.iter_mut().zip(shape.iter()).rev().fold(1,|v,(s,&d)| { *s = v; v * d });

        // build transpose permutation
        let mut tperm : Vec<usize> = (0..nelm).collect();
        tperm.sort_by_key(|&i| {
            let mut idx = [0usize; N];
            izip!(idx.iter_mut(),strides.iter(),shape.iter()).for_each(|(idx,&s,&d)| *idx = (i / s) % d);
            idx.swap(conedim0,conedim1);
            idx.iter().zip(strides.iter()).map(|v| v.0 * v.1).sum::<usize>()
        });

        let rnelm = conesize * numcone;

        let (urest,rcof) = self.xs.alloc(nnz*2+rnelm+1,nnz*2);
        let (rptr,rsubj) = urest.split_at_mut(rnelm+1);
        
        //----------------------------------------
        // Compute number of non-zeros per element of the lower triangular part if 1/2 (E+E')
        //
        rptr[0] = 0;
        for ((idx,&p0b,&p0e,&p1b,&p1e),rp) in 
            izip!(shape.index_iterator(),
                  ptr.iter(),
                  ptr[1..].iter(),
                  perm_iter(tperm.as_slice(),ptr),
                  perm_iter(tperm.as_slice(),&ptr[1..]))
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
              perm_iter(tperm.as_slice(),ptr),
              perm_iter(tperm.as_slice(),&ptr[1..]))
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

        let conedim = dom.shape[dom.conedims.0];
        let nelm : usize = conesize*numcone;

        let barvar0 = self.task.get_num_barvar().unwrap();
        let con0    = self.task.get_num_con().unwrap();
        self.task.append_barvars(vec![conedim.try_into().unwrap(); numcone].as_slice()).unwrap();
        self.task.append_cons(rnelm.try_into().unwrap()).unwrap();


        // Input linear non-zeros and bounds
        

        self.task.put_a_row_slice(con0,con0+i32::try_from(rnelm).unwrap(),
                                  &aptr[0..rnelm],
                                  &aptr[1..],
                                  asubj.as_slice(),
                                  acof.as_slice()).unwrap();
        let dim : i32 = shape[conedim0].try_into().unwrap();
        let mxs : Vec<i64> = (0..dim).flat_map(|i| std::iter::repeat(i).zip(0..i+1))
            .map(|(i,j)| self.task.append_sparse_sym_mat(dim,&[i],&[j],&[1.0]).unwrap())
            .collect::<Vec<i64>>();

        self.task.put_con_bound_slice(con0,con0+i32::try_from(rnelm).unwrap(),&vec![mosek::Boundkey::FX; nelm],&afix,&afix).unwrap();

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
                self.task.put_bara_ij(con0+i as i32, j,&[matidx],&[1.0]).unwrap();
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
            .zip(con0..con0+nelmi32)
            .for_each(| (index,coni) | {
                let barvari : i32 = barvar0 + i32::try_from(xstride.iter().zip(index.iter()).map(|(&a,&b)| a * b).sum::<usize>()).unwrap();
                let ii = index[conedim0];
                let jj = index[conedim1];
                let mi = mxs[ii*(ii+1)/2+jj];
                self.task.put_bara_ij(coni,barvari,&[mi], &[-1.0]).unwrap();
                self.cons.push(ConAtom::BarElm(coni,barvari, ii*(ii+1)/2+jj));
            });

        if let Some(name) = name {
            shape.index_iterator()
                .filter(|index| index[conedim0] >= index[conedim1])
                .zip(con0..con0+nelmi32)
                .for_each(| (index,coni) | {
                    self.task.put_con_name(coni,format!("{}{:?}",name,index).as_str()).unwrap();
                });
            let mut xshape = [1usize;N]; (0..).zip(shape.iter()).filter_map(|(i,s)| if i == conedim0 || i == conedim1 { None } else { Some(s) }).zip(xshape.iter_mut()).for_each(|(a,b)| *b = *a);
            xshape.index_iterator()
                .zip(barvar0..barvar0+i32::try_from(numcone).unwrap())
                .for_each(| (index,barvari) | {
                    self.task.put_barvar_name(barvari,format!("{}{:?}",name,&index[..N-2]).as_str()).unwrap();
                });
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
              perm_iter(&tperm,&idxs_),
              shape.index_iterator())
            .filter(|(_,_,index)| index[conedim0] < index[conedim1])
            .for_each(|(t,&s,_)| { *t = s; })
            ;

        Constraint{
            idxs,
            shape,
        }
    }


    fn linear_constraint<const N : usize>(& mut self,
                                          name : Option<&str>,
                                          dom  : LinearDomain<N>) -> Constraint<N> {
        let (dt,b,dshape,_,_) = dom.extract();

        let (shape_,ptr,_sp,subj,cof) = self.rs.pop_expr();
        let mut shape = [0usize; N]; shape.clone_from_slice(&shape_);
        if shape.len() != dshape.len() || shape.iter().zip(dshape.iter()).any(|(&a,&b)| a != b) {
            panic!("Mismatching shapes of expression {:?} and domain {:?}",shape,dshape);
        }
        // let nnz = subj.len();
        let nelm = ptr.len()-1;
        if shape.iter().product::<usize>() != nelm {
            panic!("Mismatching expression and shape");
        }

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }

        let coni = self.task.get_num_con().unwrap();
        self.task.append_cons(nelm.try_into().unwrap()).unwrap();

        if let Some(name) = name {
            Self::con_names(& mut self.task,name,coni,&shape);
        }

        self.cons.reserve(nelm);
        let firstcon = self.cons.len();
        (coni..coni+nelm as i32).for_each(|i| self.cons.push(ConAtom::Linear(i)));

        let bk = match dt {
            LinearDomainType::NonNegative => mosek::Boundkey::LO,
            LinearDomainType::NonPositive => mosek::Boundkey::UP,
            LinearDomainType::Zero        => mosek::Boundkey::FX,
            LinearDomainType::Free        => mosek::Boundkey::FR
        };

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

        Constraint{
            idxs : (firstcon..firstcon+nelm).collect(),
            shape : dshape,
        }
    }

    fn conic_constraint<const N : usize>(& mut self,
                        name : Option<&str>,
                        dom  : ConicDomain<N>) -> Constraint<N> {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain: {:?} vs {:?}",shape,dom.shape);
        }
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }
        if ! shape.iter().zip(dom.shape.iter()).all(|(&d0,&d1)| d0==d1 ) {
            panic!("Mismatching domain/expression shapes: {:?} vs {:?}",shape,dom.shape);
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
        let conesize = shape[dom.conedim];
        let numcone  = shape.iter().product::<usize>() / conesize;

        let domidx = match dom.dt {
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
                                  dom.ofs.as_slice()).unwrap();
        let d0 : usize = shape[0..dom.conedim].iter().product();
        let d1 : usize = shape[dom.conedim];
        let d2 : usize = shape[dom.conedim+1..].iter().product();
        let afeidxs : Vec<i64> = iproduct!(0..d0,0..d2,0..d1)
            .map(|(i0,i2,i1)| afei + (i0*d1*d2 + i1*d2 + i2) as i64)
            .collect();

        if let Some(name) = name {
            let _numcone = d0*d2;
            let mut xshape = [1usize; N]; 
            xshape[0..dom.conedim].copy_from_slice(&shape[0..dom.conedim]);
            if dom.conedim < N-1 {
                xshape[dom.conedim+1..N-1].copy_from_slice(&shape[dom.conedim+1..N]);
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
        iproduct!(0..d0,0..d1,0..d2)
            .for_each(|(i0,i1,i2)| self.cons.push(ConAtom::ConicElm(acci+(i0*d2+i2) as i64,i1)));

        Constraint{
            idxs : (coni..coni+nelm).collect(),
            shape : dom.shape
        }
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

    /// Set the objective
    ///
    /// Arguments:
    /// - `name` Optional objective name
    /// - `sense` Objective sense
    /// - `expr` Objective expression, this must contain exactly one
    ///   element. The shape is otherwise ignored.
    pub fn objective<E : expr::ExprTrait<0>>(& mut self,
                                             name  : Option<&str>,
                                             sense : Sense,
                                             expr  : & E) {
        expr.eval_finalize(& mut self.rs,& mut self.ws, & mut self.xs);
        self.set_objective(name,sense);
    }

    //======================================================
    // Optimize


    pub fn set_parameter<T>(& mut self, parname : &str, parval : T) 
        where T : SolverParameterValue {
        parval.set(parname,self);
    }

    pub fn set_double_parameter(&mut self, parname : &str, parval : f64) {
        self.task.put_na_dou_param(parname, parval).unwrap();
    }
    pub fn set_int_parameter(&mut self, parname : &str, parval : i32) {
        self.task.put_na_int_param(parname, parval).unwrap();
    }
    pub fn set_str_parameter(&mut self, parname : &str, parval : &str) {
        self.task.put_na_str_param(parname, parval).unwrap();
    }

    /// Solve the problem and extract the solution.
    pub fn solve(& mut self) {
        self.task.put_int_param(mosek::Iparam::REMOVE_UNUSED_SOLUTIONS, 1).unwrap();
        self.task.optimize().unwrap();

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
                                                        .fold_map(0,|&p,n| n+p)).collect();
        let barvarptr : Vec<usize> = once(0usize).chain((0..numbarvar)
                                                        .map(|j| self.task.get_len_barvar_j(j as i32).unwrap() as usize)
                                                        .fold_map(0,|&p,n| n+p)).collect();

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
                let (psta,dsta) = split_sol_sta(self.task.get_sol_sta(whichsol).unwrap());
                sol.primal.status = psta;
                sol.dual.status   = dsta;

                if let SolutionStatus::Undefined = psta {}
                else {
                    sol.primal.resize(self.vars.len(),self.cons.len());
                    self.task.get_xx(whichsol,xx.as_mut_slice()).unwrap();
                    self.task.get_xc(whichsol,xc.as_mut_slice()).unwrap();
                    if numbarvar > 0 { self.task.get_barx_slice(whichsol,0,numbarvar as i32,barx.len() as i64,barx.as_mut_slice()).unwrap(); }
                    if numacc > 0 { self.task.evaluate_accs(whichsol,accx.as_mut_slice()).unwrap(); }

                    self.vars[1..].iter().zip(sol.primal.var[1..].iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            VarAtom::Linear(j) => xx[j as usize],
                            VarAtom::BarElm(j,ofs) => barx[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                            VarAtom::ConicElm(j,_coni) => xx[j as usize]
                        };
                    });
                    //println!("{}:{}: cons = {:?}",file!(),line!(),&self.cons);
                    //println!("{}:{}: numacc = {:?}",file!(),line!(),numacc);
                    //println!("{}:{}: accptr = {:?}",file!(),line!(),accptr);
                    self.cons.iter().zip(sol.primal.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm(acci,ofs) => { 
                                //println!("{}:{}: acci = {}",file!(),line!(),acci);
                                //println!("{}:{}: accptr[{}] = {}",file!(),line!(),acci,accptr[acci as usize]);
                                //println!("{}:{}: accx = {:?}",file!(),line!(),accx);
                                accx[accptr[acci as usize]+ofs]
                            },
                            ConAtom::Linear(i) => xc[i as usize],
                            ConAtom::BarElm(_,j,ofs) => barx[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                        };
                    });
                }

                if let SolutionStatus::Undefined = dsta {}
                else {
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
                            VarAtom::Linear(j) => slx[j as usize] - sux[j as usize],
                            VarAtom::BarElm(j,ofs) => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                            VarAtom::ConicElm(_j,coni) => {
                                match self.cons[coni] {
                                    ConAtom::ConicElm(acci,ofs) => doty[accptr[acci as usize]+ofs],
                                    ConAtom::Linear(i) => y[i as usize],
                                    ConAtom::BarElm(_,j,ofs) => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                                }
                            }
                        };
                    });
                    self.cons.iter().zip(sol.dual.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm(acci,ofs) => doty[accptr[acci as usize]+ofs],
                            ConAtom::Linear(i) => y[i as usize],
                            ConAtom::BarElm(_,j,ofs) => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
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

    /// Get solution status for the given solution
    pub fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus) {
        if let Some(sol) = self.select_sol(solid) {
            (sol.primal.status,sol.dual.status)
        }
        else {
            (SolutionStatus::Undefined,SolutionStatus::Undefined)
        }
    }

    /// Get primal solution values for an item
    ///
    /// Returns: If solution item is defined, return the solution, otherwise a n error message.
    pub fn primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.primal(self,solid) }
    
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
                    VarAtom::Linear(j) => {
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

fn split_sol_sta(solsta : i32) -> (SolutionStatus,SolutionStatus) {
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

    use crate::utils::*;
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
        let _v4 = m.variable(None, in_quadratic_cone(5));
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

        let c = m.constraint(Some("c"), &x.sub(dense([3,2,3],vec![0.0,-1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0,-9.0,-10.0,-11.0,-12.0,-13.0,-14.0,-15.0,-16.0,-17.0])), in_psd_cones(&[3,2,3],0,2));
        m.objective(Some("obj"), Sense::Minimize, &z);

        m.solve();

        m.write_problem("test_psd_con.ptf");
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
}
