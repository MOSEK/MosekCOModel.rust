//#![feature(specialization)]
extern crate mosek;
extern crate itertools;

mod utils;
pub mod variable;
pub mod matrix;
pub mod expr;
use expr::workstack::WorkStack;
use itertools::{iproduct};
use std::iter::once;

pub use expr::{ExprTrait,ExprLeftMultipliable,ExprRightMultipliable};
pub use variable::{Variable};

use utils::*;

/////////////////////////////////////////////////////////////////////
// Model, constraint and variables

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
#[derive(Clone,Copy)]
enum ConAtom {
    ConicElm(i64,usize),
    Linear(i32)
}

#[derive(Clone,Copy)]
pub enum SolutionType {
    Default,
    Basic,
    Interior,
    Integer
}


#[derive(Clone,Copy,Debug)]
pub enum SolutionStatus {
    Optimal,
    Feasible,
    CertInfeas,
    CertIllposed,
    Unknown,
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

////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////

/// The Model object encapsulates an optimization problem and a
/// mapping from the structured API to the internal Task items.
pub struct Model {
    /// The MOSEK task
    task : mosek::Task,
    vars      : Vec<VarAtom>,
    cons      : Vec<ConAtom>,

    sol_bas : Solution,
    sol_itr : Solution,
    sol_itg : Solution,

    /// Workstacks for evaluating expressions
    rs : WorkStack,
    ws : WorkStack,
    xs : WorkStack
}

////////////////////////////////////////////////////////////
// ModelItem
////////////////////////////////////////////////////////////
pub trait ModelItem {
    fn len(&self) -> usize;
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

////////////////////////////////////////////////////////////
// Variable and Constraint
////////////////////////////////////////////////////////////

pub trait ModelItemIndex<I> {
    type Output;
    fn index(&self, index : I) -> Self::Output;
}


// impl std::ops::Index<&[usize]> for Variable { ... }
// impl std::ops::Index<&[std::ops::Range]> for Variable { ... }

/// A Constraint object is a wrapper around an array of constraint
/// indexes and a shape. Note that constraint objects are never sparse.
#[derive(Clone)]
pub struct Constraint {
    idxs     : Vec<usize>,
    shape    : Vec<usize>
}

impl ModelItem for Constraint {
    fn len(&self) -> usize { return self.shape.iter().product(); }
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

/////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////
// Domain definitions
pub enum LinearDomainType {
    NonNegative,
    NonPositive,
    Zero,
    Free
}

pub enum ConicDomainType {
    QuadraticCone,
    RotatedQuadraticCone
}
pub enum ParamConicDomainType {
    PrimalPowerCone,
    DualPowerCone
}

pub trait ConDomainTrait {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint;
}
pub trait VarDomainTrait {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable;
}


pub struct LinearDomain {
    dt    : LinearDomainType,
    ofs   : Vec<f64>,
    shape : Vec<usize>,
    sp    : Option<Vec<usize>>
}

pub struct ConicDomain {
    dt      : ConicDomainType,
    ofs     : Vec<f64>,
    shape   : Vec<usize>,
    conedim : usize
}

pub struct PSDDomain {
    shape    : Vec<usize>,
    conedims : (usize,usize)
}

impl VarDomainTrait for ConicDomain {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.conic_variable(name,self)
    }
}
impl ConDomainTrait for ConicDomain {
    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.conic_constraint(name,self)
    }
}

impl VarDomainTrait for &[usize] {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,self)
    }
}
impl ConDomainTrait for &[usize] {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:vec![0.0; self.iter().product::<usize>()],
                                shape:self.to_vec(),
                                sp:None})
    }
}

impl VarDomainTrait for Vec<usize> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,self.as_slice())
    }
}
impl ConDomainTrait for Vec<usize> {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint {
         m.linear_constraint(name,
                             LinearDomain{
                                 dt:LinearDomainType::Free,
                                 ofs:vec![0.0; self.iter().product::<usize>()],
                                 shape:self,
                                 sp:None})
    }
}
impl VarDomainTrait for usize {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,&[self])
    }
}
impl ConDomainTrait for usize {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:vec![0.0; self],
                                shape:vec![self],
                                sp:None})
    }
}
impl VarDomainTrait for PSDDomain {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.psd_variable(name,self)
    }
}


impl LinearDomain {
    pub fn with_shape(self,shape : Vec<usize>) -> LinearDomain {
        match self.sp {
            Some(ref sp) => if ! sp.last().map_or_else(|| true,|&v| v < shape.iter().product()) {
                panic!("Shaped does not match sparsity");
            },
            None => if self.ofs.len() != shape.iter().product()  {
                panic!("Shaped does not fit expression");
            }
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : shape,
            sp    : self.sp
        }
    }

    pub fn with_sparsity(self,sp : Vec<usize>) -> LinearDomain {
        if sp.len() > 1 {
            if ! sp[..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                panic!("Sparsity pattern is not sorted");
            }
        }
        if ! sp.last().map_or_else(|| true, |&v| v < self.shape.iter().product()) {
                panic!("Sparsity pattern does not fit in shape");
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : self.shape,
            sp    : Some(sp)
        }
    }

    pub fn with_shape_and_sparsity(self,shape : Vec<usize>, sp : Vec<usize>) -> LinearDomain {
        if sp.len() > 1 {
            if ! sp[..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                panic!("Sparsity pattern is not sorted");
            }
        }
        if ! sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
                panic!("Sparsity pattern does not fit in shape");
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : shape,
            sp    : Some(sp)
        }
    }
}

impl VarDomainTrait for LinearDomain {
    fn create(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,self)
    }
}
impl ConDomainTrait for LinearDomain {
    fn create(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,self)
    }

}

pub trait OffsetTrait {
    fn greater_than(self) -> LinearDomain;
    fn less_than(self)    -> LinearDomain;
    fn equal_to(self)     -> LinearDomain;
}

impl OffsetTrait for f64 {
    fn greater_than(self) -> LinearDomain { LinearDomain{ dt : LinearDomainType::NonNegative, ofs:vec![self], shape:vec![], sp : None } }
    fn less_than(self)    -> LinearDomain { LinearDomain{ dt : LinearDomainType::NonPositive,  ofs:vec![self], shape:vec![], sp : None } }
    fn equal_to(self)     -> LinearDomain { LinearDomain{ dt : LinearDomainType::Zero,         ofs:vec![self], shape:vec![], sp : None } }
}

impl OffsetTrait for Vec<f64> {
    fn greater_than(self) -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:self, shape:vec![n], sp : None } }
    fn less_than(self)    -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:self, shape:vec![n], sp : None } }
    fn equal_to(self)     -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero, ofs:self, shape:vec![n], sp : None } }
}

impl OffsetTrait for &[f64] {
    fn greater_than(self) -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:self.to_vec(), shape:vec![n], sp : None } }
    fn less_than(self)    -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:self.to_vec(), shape:vec![n], sp : None } }
    fn equal_to(self)     -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero, ofs:self.to_vec(), shape:vec![n], sp : None } }
}

////////////////////////////////////////////////////////////
// Domain constructors
////////////////////////////////////////////////////////////

pub fn greater_than<T : OffsetTrait>(v : T) -> LinearDomain { v.greater_than() }
pub fn less_than<T : OffsetTrait>(v : T) -> LinearDomain { v.less_than() }
pub fn equal_to<T : OffsetTrait>(v : T) -> LinearDomain { v.equal_to() }
pub fn in_quadratic_cone(dim : usize) -> ConicDomain { ConicDomain{dt:ConicDomainType::QuadraticCone,ofs:vec![0.0; dim],shape:vec![dim],conedim:0} }
pub fn in_rotated_quadratic_cone(dim : usize) -> ConicDomain { ConicDomain{dt:ConicDomainType::RotatedQuadraticCone,ofs:vec![0.0; dim],shape:vec![dim],conedim:0} }
pub fn in_quadratic_cones(shape : Vec<usize>, conedim : usize) -> ConicDomain {
    if conedim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicDomain{dt:ConicDomainType::QuadraticCone,
                ofs : vec![0.0; shape.iter().product()],
                shape:shape,
                conedim:conedim}
}
pub fn in_rotated_quadratic_cones(shape : Vec<usize>, conedim : usize) -> ConicDomain {
    if conedim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicDomain{dt      : ConicDomainType::RotatedQuadraticCone,
                ofs     : vec![0.0; shape.iter().product()],
                shape   : shape,
                conedim : conedim}
}
pub fn in_psd_cone(dim : usize) -> PSDDomain {
    PSDDomain{
        shape : vec![dim,dim],
        conedims : (0,1)
    }
}
pub fn in_psd_cones(shape : Vec<usize>, conedim1 : usize, conedim2 : usize) -> PSDDomain {
    if conedim1 == conedim2 || conedim1 >= shape.len() || conedim2 >= shape.len() {
        panic!("Invalid shape or cone dimensions");
    }
    if shape[conedim1] != shape[conedim2] {
        panic!("Mismatching cone dimensions");
    }
    PSDDomain{
        shape    : shape,
        conedims : (conedim1,conedim2)
    }
}

////////////////////////////////////////////////////////////
// Model
////////////////////////////////////////////////////////////

impl Model {
    /// Create new Model object
    ///
    /// Arguments:
    /// - `name` An optional name
    pub fn new(name : Option<&str>) -> Model {
        let mut task = mosek::Task::new().unwrap();
        match name {
            Some(name) => task.put_task_name(name).unwrap(),
            None => {}
        }
        Model{
            task    : task,
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

    /// Write problem to a file
    pub fn write_problem(&self, filename : &str) {
        self.task.write_data(filename).unwrap();
    }

    ////////////////////////////////////////////////////////////
    // Variable interface

    /// Add a Variable
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `dom` The domain of the variable. This defines the bound
    ///   type, shape and sparsity of the variable. For sparse
    ///   variables, elements outside of the sparsity pattern are
    ///   treated as variables fixed to 0.0.
    pub fn variable<D : VarDomainTrait>(& mut self, name : Option<&str>, dom : D) -> Variable {
        dom.create(self,name)
    }


    fn var_names(& mut self, name : &str, first : i32, shape : &[usize], sp : Option<&[usize]>) {
        let mut buf = name.to_string();
        let baselen = buf.len();
        utils::for_each_index(shape,
                              sp,
                              |j,idx:&[usize]| {
                                  buf.truncate(baselen);
                                  buf.push('[');
                                  if let Some(&i) = idx.first() {
                                      for c in i.digits_10() { buf.push(c); }
                                      for &i in idx[1..].iter() {
                                          buf.push(',');
                                          for c in i.digits_10() {
                                              buf.push(c);
                                          }
                                      }
                                  }
                                  buf.push(']');
                                  self.task.put_var_name(first + j as i32,buf.as_str()).unwrap();
                              });
    }

    fn con_names(task : & mut mosek::Task, name : &str, first : i32, shape : &[usize]) {
        let mut buf = name.to_string();
        let baselen = buf.len();
        utils::for_each_index(shape,
                              None,
                              |j,idx:&[usize]| {
                                  buf.truncate(baselen);
                                  buf.push('[');
                                  if let Some(&i) = idx.first() {
                                      for c in i.digits_10() { buf.push(c); }
                                      for &i in idx[1..].iter() {
                                          buf.push(',');
                                          for c in i.digits_10() {
                                              buf.push(c);
                                          }
                                      }
                                  }
                                  buf.push(']');
                                  task.put_con_name(first + j as i32,buf.as_str()).unwrap();
                              });
    }

    fn linear_variable(&mut self, name : Option<&str>,dom : LinearDomain) -> Variable {
        let n = dom.ofs.len();
        let vari = self.task.get_num_var().unwrap();
        let varend : i32 = ((vari as usize)+n).try_into().unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        if let Some(name) = name {
            if let Some(ref sp) = dom.sp {
                self.var_names(name,vari,dom.shape.as_slice(),Some(sp.as_slice()))
            }
            else {
                self.var_names(name,vari,dom.shape.as_slice(),None)
            }
        }
        self.vars.reserve(n);

        let firstvar = self.vars.len();
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j)));

        match dom.dt {
            LinearDomainType::Free        => self.task.put_var_bound_slice_const(vari,vari+n as i32,mosek::Boundkey::FR,0.0,0.0).unwrap(),
            LinearDomainType::Zero        => {
                let bk = vec![mosek::Boundkey::FX; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap();
            },
            LinearDomainType::NonNegative => {
                let bk = vec![mosek::Boundkey::LO; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap();
            },
            LinearDomainType::NonPositive => {
                let bk = vec![mosek::Boundkey::UP; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap()
            }
        }

        Variable::new((firstvar..firstvar+n).collect(),
                      dom.sp,
                      dom.shape)
    }

    fn free_variable(&mut self, name : Option<&str>, shape : &[usize]) -> Variable {
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
                      shape.to_vec())
    }

    fn psd_variable(&mut self, _name : Option<&str>, dom : PSDDomain) -> Variable {
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
        for k in 0..numcone {
            for j in 0..d1 {
                for i in j..d1 {
                    self.vars.push(VarAtom::BarElm(barvar0 + k as i32, i*(i+1)/2+j))
                }
            }
        }

        let idxs : Vec<usize> = if conedim0 < conedim1 {
            iproduct!(0..d0,0..d1,0..d2,0..d3,0..d4).map(|(i0,i1,i2,i3,i4)| {
                let (i1,i3) = if i3 < i1 { (i3,i1) } else { (i1,i3) };

                let baridx = i0 * d2 * d4 + i2 * d4 + i4;
                let ofs    = d1*i3-d3*(d3-1)/2-d3+i1;

                baridx*conesz+ofs
            }).collect()
        }
        else {
            iproduct!(0..d0,0..d1,0..d2,0..d3,0..d4).map(|(i0,i3,i2,i1,i4)| {
                let (i1,i3) = if i3 < i1 { (i3,i1) } else { (i1,i3) };

                let baridx = i0 * d2 * d4 + i2 * d4 + i4;
                let ofs    = d1*i3-d3*(d3-1)/2-d3+i1;

                baridx*conesz+ofs
            }).collect()
        };

        Variable::new(idxs,
                       None,
                      dom.shape)
    }

    fn conic_variable(&mut self, _name : Option<&str>, dom : ConicDomain) -> Variable {
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
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
        };

        self.task.append_afes(n as i64).unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),n as i64,afei,dom.ofs.as_slice()).unwrap();
        self.task.put_afe_f_entry_list(asubi.as_slice(),asubj.as_slice(),acof.as_slice()).unwrap();

        // if let Some(name) = name {
        //     if let Some(ref sp) = dom.sp {
        //         self.var_names(name,vari,dom.shape,Some(sp.as_slice()))
        //     }
        //     else {
        //         self.var_names(name,vari,dom.shape,None)
        //     }
        // }

        let firstvar = self.vars.len();
        self.vars.reserve(n);
        self.cons.reserve(n);

        iproduct!(0..d0,0..d1,0..d2).enumerate()
            .for_each(|(i,(i0,i1,i2))| {
                self.vars.push(VarAtom::ConicElm(vari+i as i32,self.cons.len()));
                self.cons.push(ConAtom::ConicElm(acci+i1 as i64,i0*d2+i2))
            } );

        Variable::new((firstvar..firstvar+n).collect(),
                      None,
                      dom.shape)
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

    ////////////////////////////////////////////////////////////
    // Constraint interface


    /// Add a constraint
    ///
    /// Note that even if the domain or the expression are sparse, a constraint will always be full.
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `expr` Constraint expression. Note that the shape of the expression and the domain must match exactly.
    /// - `dom`  The domain of the constraint. This defines the bound type and shape.
    pub fn constraint<E : expr::ExprTrait, D : ConDomainTrait>(& mut self, name : Option<&str>, expr : &E, dom : D) -> Constraint {
        expr.eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs);
        dom.create(self,name)
    }

    fn linear_constraint(& mut self,
                         name : Option<&str>,
                         dom  : LinearDomain) -> Constraint {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        println!("{}:{}: dom.shape = {:?}",file!(),line!(),dom.shape);
        if dom.shape.len() != shape.len() || ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain");
        }
        // let nnz = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }

        let coni = self.task.get_num_con().unwrap();
        self.task.append_cons(nelm.try_into().unwrap()).unwrap();

        if let Some(name) = name {
            Self::con_names(& mut self.task,name,coni,dom.shape.as_slice())
        }

        self.cons.reserve(nelm);
        let firstcon = self.cons.len();
        (coni..coni+nelm as i32).for_each(|i| self.cons.push(ConAtom::Linear(i)));

        let bk = match dom.dt {
            LinearDomainType::NonNegative => mosek::Boundkey::LO,
            LinearDomainType::NonPositive => mosek::Boundkey::UP,
            LinearDomainType::Zero        => mosek::Boundkey::FX,
            LinearDomainType::Free        => mosek::Boundkey::FR
        };


        
        // let acci = self.task.get_num_acc().unwrap();
        // let afei = self.task.get_num_afe().unwrap();

        // self.task.append_afes(nelm as i64).unwrap();
        // let domidx = match dom.dt {
        //     LinearDomainType::NonNegative => self.task.append_rplus_domain(nelm as i64).unwrap(),
        //     LinearDomainType::NonPositive => self.task.append_rminus_domain(nelm as i64).unwrap(),
        //     LinearDomainType::Zero        => self.task.append_rzero_domain(nelm as i64).unwrap(),
        //     LinearDomainType::Free        => self.task.append_r_domain(nelm as i64).unwrap(),
        // };

        // match dom.sp {
        //     None => self.task.append_acc_seq(domidx, afei,dom.ofs.as_slice()).unwrap(),
        //     Some(sp) => {
        //         let mut ofs = vec![0.0; nelm];
        //         if sp.len() != dom.ofs.len() { panic!("Broken sparsity pattern") };
        //         if let Some(&v) = sp.iter().max() { if v >= nelm { panic!("Broken sparsity pattern"); } }
        //         sp.iter().zip(dom.ofs.iter()).for_each(|(&ix,&c)| unsafe { *ofs.get_unchecked_mut(ix) = c; } );
        //         self.task.append_acc_seq(domidx, afei,ofs.as_slice()).unwrap();
        //     }
        // }

        self.cons.reserve(nelm);
        // (0..nelm).for_each(|i| self.cons.push(ConAtom::ConicElm(acci,i)));

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());
        // let abarsubi : Vec<i64> = abarsubi.iter().map(|&i| i + afei).collect();

        // let afeidxs : Vec<i64> = (afei..afei+nelm as i64).collect();
        if asubj.len() > 0 {
            self.task.put_a_row_slice(
                coni,coni+nelm as i32,
                &aptr[0..aptr.len()-1],
                &aptr[1..],
                asubj.as_slice(),
                acof.as_slice()).unwrap();

            // self.task.put_afe_f_row_list(
            //     afeidxs.as_slice(),
            //     aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0).try_into().unwrap()).collect::<Vec<i32>>().as_slice(),
            //     &aptr[..nelm],
            //     asubj.as_slice(),
            //     acof.as_slice()).unwrap();
        }
        // self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();

        let rhs : Vec<f64> = dom.ofs.iter().zip(afix.iter()).map(|(&ofs,&b)| ofs-b).collect();
        println!("{}:{}: coni = {}:{}, dom.ofs : {}, afix : {}",file!(),line!(),coni,coni+nelm as i32,dom.ofs.len(), afix.len());
        self.task.put_con_bound_slice(coni,
                                      coni+nelm as i32,
                                      vec![bk; nelm].as_slice(),
                                      rhs.as_slice(),
                                      rhs.as_slice()).unwrap();

        if abarsubi.len() > 0 {
            for (i,j,subk,subl,cof) in utils::ijkl_slice_iterator(abarsubi.as_slice(),
                                                                  abarsubj.as_slice(),
                                                                  abarsubk.as_slice(),
                                                                  abarsubl.as_slice(),
                                                                  abarcof.as_slice()) {
                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                //self.task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
                self.task.put_bara_ij(coni+i as i32, j,&[matidx],&[1.0]).unwrap();
            }
        }

        Constraint{
            idxs : (firstcon..firstcon+nelm).collect(),
            shape : dom.shape
        }
    }

    fn conic_constraint(& mut self,
                        _name : Option<&str>,
                        dom  : ConicDomain) -> Constraint {
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
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
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

        // if let Some(name) = name {
        //     let accshape = shape[0..dom.conedim].iter().join(shape[dom.conedim+1..].iter()).collect();
        //     //iproduct!(0..d0,0..d2).for_each(|(i0,i2)| self.task.put_acc_name(format!("{}[{},*,{}]")).unwrap() )
        //     }
        // }

        if asubj.len() > 0 {
            self.task.put_afe_f_row_list(afeidxs.as_slice(),
                                         aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                         &aptr[..nelm],
                                         asubj.as_slice(),
                                         acof.as_slice()).unwrap();
        }
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();
        if abarsubi.len() > 0 {
            for (i,j,subk,subl,cof) in utils::ijkl_slice_iterator(abarsubi.as_slice(),
                                                                  abarsubj.as_slice(),
                                                                  abarsubk.as_slice(),
                                                                  abarsubl.as_slice(),
                                                                  abarcof.as_slice()) {
                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
            }
        }

        let coni = self.cons.len();
        self.cons.reserve(nelm);
        iproduct!(0..d0,0..d1,0..d2)
            .for_each(|(i0,i1,i2)| self.cons.push(ConAtom::ConicElm(acci+i1 as i64,i0*d2+i2)));

        Constraint{
            idxs : (coni..coni+nelm).collect(),
            shape : dom.shape
        }
    }

    ////////////////////////////////////////////////////////////
    // Objective


    fn set_objective(& mut self, name : Option<&str>, sense : Sense) {
        let (_shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ptr.len()-1 > 1 { panic!("Objective expressions may only contain one element"); }

        println!("Objective: ptr = {:?}, subj = {:?}, cof = {:?}",ptr,subj,cof);

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
    pub fn objective<E : expr::ExprTrait>(& mut self,
                                          name  : Option<&str>,
                                          sense : Sense,
                                          expr  : & E) {
        expr.eval_finalize(& mut self.rs,& mut self.ws, & mut self.xs);
        self.set_objective(name,sense);
    }

    ////////////////////////////////////////////////////////////
    // Optimize

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
                    self.cons.iter().zip(sol.primal.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm(acci,ofs) => accx[accptr[acci as usize]+ofs],
                            ConAtom::Linear(i) => xc[i as usize]
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
                                    ConAtom::Linear(i) => y[i as usize]
                                }
                            }
                        };
                    });
                    self.cons.iter().zip(sol.dual.con.iter_mut()).for_each(|(&v,r)| {
                        *r = match v {
                            ConAtom::ConicElm(acci,ofs) => doty[accptr[acci as usize]+ofs],
                            ConAtom::Linear(i) => y[i as usize]
                        };
                    });
                }
            }
        }
    }
    ////////////////////////////////////////////////////////////
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
    fn primal_var_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
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

    fn dual_var_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
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
    pub fn primal_solution<I:ModelItem>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.primal(self,solid) }

    /// Get dual solution values for an item
    ///
    /// Returns: If solution item is defined, return the solution, otherwise a n error message.
    pub fn dual_solution<I:ModelItem>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.dual(self,solid) }

    /// Get primal solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn primal_solution_into<I:ModelItem>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }

    /// Get dual solution values for an item
    ///
    /// Arguments:
    /// - `solid` Which solution
    /// - `item`  The item to get solution for
    /// - `res`   Copy the solution values into this slice
    /// Returns: The number of values copied if solution is available, otherwise an error string.
    pub fn dual_solution_into<I:ModelItem>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }
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
            else {
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
                        barcof.push(c);
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

/////////////////////////////////////////////////////////////////////
// TEST

#[cfg(test)]
mod tests {
    use super::*;

    fn eq<T:std::cmp::Eq>(a : &[T], b : &[T]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(a,b)| *a == *b )
    }

    #[test]
    fn it_works() {
        let mut m = Model::new(Some("SuperModel"));
        let mut v1 = m.variable(None, greater_than(5.0));
        let mut v2 = m.variable(None, 10);
        let mut v3 = m.variable(None, vec![3,3]);
        let mut v4 = m.variable(None, in_quadratic_cone(5));
        let mut v5 = m.variable(None, greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(vec![2,2]));
        let mut v6 = m.variable(None, greater_than(vec![1.0,3.0]).with_shape_and_sparsity(vec![2,2],vec![0,3]));
    }

    #[test]
    fn variable_stack() {
        let mut m = Model::new(Some("SuperModel"));
        let mut v1 = m.variable(None, vec![3,2,1]);
        let mut v2 = m.variable(None, vec![3,2,1]);
        let mut v3 = m.variable(None, equal_to(vec![1.0,2.0,3.0,4.0]).with_shape_and_sparsity(vec![3,2,1],vec![0,2,3,5]));

        let mut w_0 = Variable::stack(0,&[&v1,&v2]);
        let mut w_1 = Variable::stack(1,&[&v1,&v2]);
        let mut w_2 = Variable::stack(2,&[&v1,&v2]);

        assert!(eq(w_0.shape(),&[6,2,1]));
        assert!(eq(w_0.idxs(),&[1,2,3,4,5,6,7,8,9,10,11,12]));
        assert!(eq(w_1.shape(),&[3,4,1]));
        assert!(eq(w_1.idxs(),&[1,2,7,8,3,4,9,10,5,6,11,12]));
        assert!(eq(w_2.shape(),&[3,2,2]));
        assert!(eq(w_2.idxs(),&[1,7,2,8,3,9,4,10,5,11,6,12]));

        let mut u_0 = Variable::stack(0,&[&v1,&v3]);
        let mut u_1 = Variable::stack(1,&[&v1,&v3]);
        let mut u_2 = Variable::stack(2,&[&v1,&v3]);

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
}

//
// 1  4   14
// 2  5   15 16
// 3  6      17
//

//  1  2
//  3  4
//  5  6
// 13
// 14 15
//    16
