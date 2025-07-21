use crate::*;
use crate::model::{ModelWithLogCallback,ModelWithIntSolutionCallback,VectorConeModelTrait,PSDModelTrait};
use crate::domain::*;
use crate::utils::iter::{ChunksByIterExt, PermuteByEx, PermuteByMutEx};
use std::f64;
use std::ops::ControlFlow;
use std::path::Path;
use itertools::{iproduct, izip, Either};
use model::ModelWithControlCallback;

#[derive(Clone,Copy)]
enum Item {
    Linear{index:usize},
    RangedUpper{index:usize},
    RangedLower{index:usize},
    Conic{index:usize},
}

impl Item {
    fn index(&self) -> usize { 
        match self {
            Item::Conic { index } => *index,
            Item::Linear { index } => *index,
            Item::RangedUpper { index } => *index,
            Item::RangedLower { index } => *index
        }
    } 
}

#[derive(Clone)]
enum ConeType {
    QuadraticCone,
    RoteatedQuadraticCone,
    SVecPSDCone,
    GeometricMeanCone,
    PowerCone(Vec<f64>),
    ExponentialCone
}

#[derive(Clone,Copy)]
enum Element {
    Linear{lb:f64,ub:f64},
    Conic{coneidx:usize,offset:usize,b:f64},
}

/// Simple model object.
#[derive(Default)]
pub struct Backend {
    name : Option<String>,

    var_elt       : Vec<Element>, // Either lb,ub,int or index,coneidx,offset
    var_int       : Vec<bool>,
    cones         : Vec<ConeType>,

    vars          : Vec<Item>,

    a_ptr         : Vec<[usize;2]>,
    a_subj        : Vec<usize>,
    a_cof         : Vec<f64>,
    con_elt       : Vec<Element>,

    con_a_row     : Vec<usize>, // index into a_ptr
    cons          : Vec<Item>,

    sense_max     : bool,
    c_subj        : Vec<usize>,
    c_cof         : Vec<f64>,
}

impl BaseModelTrait for Backend {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name         : name.map(|v| v.to_string()),
            .. Default::default()
        }
    }
    fn free_variable<const N : usize>
        (&mut self,
         _name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();
        let first = self.var_elt.len();
        let last  = first + n;

        self.var_elt.resize(last,Element::Linear { lb: f64::NEG_INFINITY, ub: f64::INFINITY });
        self.var_int.resize(last,false);

        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last {
            self.vars.push(Item::Linear{index:i});
        }

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), None, shape))
    }

    fn linear_variable<const N : usize,R>
        (&mut self, 
         _name : Option<&str>,
         dom  : LinearDomain<N>) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result,String>    
        where 
            Self : Sized
    {
        let (dt,b,sp,shape,is_integer) = dom.dissolve();
        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_int.len();
        let last  = first + n;


        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last { self.vars.push(Item::Linear{index:i}) }
        match dt {
            LinearDomainType::Zero => {
                self.var_elt.resize(last,Element::Linear { lb: 0.0, ub: 0.0 });
            },
            LinearDomainType::Free => {
                self.var_elt.resize(last,Element::Linear { lb: f64::NEG_INFINITY, ub: f64::INFINITY});
            },
            LinearDomainType::NonNegative => {
                self.var_elt.reserve(last);
                for lb in b {
                    self.var_elt.push(Element::Linear { lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.var_elt.reserve(last);
                for ub in b {
                    self.var_elt.push(Element::Linear { lb : f64::NEG_INFINITY, ub });
                }
            },
        }
        self.var_int.resize(last,is_integer);

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, _name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (shape,bl,bu,sp,is_integer) = dom.dissolve();

        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_int.len();
        let last  = first + n;

        let ptr0 = self.vars.len();
        let ptr1 = self.vars.len()+n;
        let ptr2 = self.vars.len()+2*n;
        self.vars.reserve(n*2);
        for i in first..last { self.vars.push(Item::RangedLower{index:i}) }
        for i in first..last { self.vars.push(Item::RangedUpper{index:i}) }

        self.var_elt.reserve(last);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.var_elt.push(Element::Linear { lb, ub })
        }
        self.var_int.resize(last,is_integer);

        Ok((Variable::new((ptr0..ptr1).collect::<Vec<usize>>(), sp.clone(), &shape),
            Variable::new((ptr1..ptr2).collect::<Vec<usize>>(), sp, &shape)))
    }

    fn linear_constraint<const N : usize>
        (& mut self, 
         name  : Option<&str>,
         dom   : LinearDomain<N>,
         _eshape : &[usize], 
         ptr   : &[usize], 
         subj  : &[usize], 
         cof   : &[f64]) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
    {
        let (dt,b,sp,shape,_is_integer) = dom.dissolve();

        assert_eq!(b.len(),ptr.len()-1); 
        let nrow = b.len();

        let a_row0 = self.a_ptr.len();
        let con_row0 = self.con_a_row.len();
        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        {
            for (b,n) in ptr.iter().zip(ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
                self.a_ptr.push([b,n]);
            }
        }

        let con0 = self.cons.len();
        self.a_subj.extend_from_slice(subj);
        self.a_cof.extend_from_slice(cof);
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }
        self.cons.reserve(n); for i in con_row0..con_row0+n { self.cons.push(Item::Linear { index: i }) }
        
        match dt {
            LinearDomainType::Zero => {
                self.con_elt.reserve(con_row0+nrow);
                for b in b {
                    self.con_elt.push(Element::Linear { lb: b, ub: b });
                }
            },
            LinearDomainType::Free => { 
                self.con_elt.resize(con_row0+nrow,Element::Linear { lb: f64::NEG_INFINITY, ub: f64::INFINITY });
            },
            LinearDomainType::NonNegative => {
                self.con_elt.reserve(con_row0+nrow);
                for lb in b {
                    self.con_elt.push(Element::Linear { lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.con_elt.reserve(con_row0+nrow);
                for ub in b {
                    self.con_elt.push(Element::Linear { lb : f64::NEG_INFINITY, ub });
                }
            },
        }

        Ok(Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape))
    }

    fn ranged_constraint<const N : usize>
        (& mut self, 
         name : Option<&str>, 
         dom  : LinearRangeDomain<N>,
         _eshape : &[usize], 
         ptr : &[usize], 
         subj : &[usize], 
         cof : &[f64]) -> Result<<LinearRangeDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
    {
        let (shape,bl,bu,_,_) = dom.dissolve();

        let a_row0 = self.a_ptr.len()-1;
        let con_row0 = self.con_a_row.len();

        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        for (b,n) in izip!(ptr.iter(),ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
            self.a_ptr.push([b,n]);
        }

        self.a_subj.extend_from_slice(subj);
        self.a_cof.extend_from_slice(cof);
        self.con_elt.reserve(con_row0+n);
        for (&lb,&ub) in (bl.iter().zip(bu.iter())) {
            self.con_elt.push(Element::Linear { lb, ub });
        }

        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }

        let con0 = self.cons.len();
        self.cons.reserve(n*2);
        for i in con_row0..con_row0+n { self.cons.push(Item::RangedLower { index: i }); }
        for i in con_row0..con_row0+n { self.cons.push(Item::RangedUpper { index: i }); }

        Ok((Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape),
            Constraint::new((con0+n..con0+2*n).collect::<Vec<usize>>(), &shape)))
    }

    fn update(& mut self, idxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>
    {
        if shape.iter().product::<usize>() != idxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = idxs.iter().max() {
            if i >= self.cons.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

        for (subj,cof,i) in izip!(subj.chunks_ptr(ptr),cof.chunks_ptr(ptr),idxs.iter().map(|&i| self.cons[i].index())) {
            let n = subj.len();

            let ai = self.con_a_row[i];

            let entry = self.a_ptr[ai];
            if entry[1] >= n {
                self.a_subj[entry[0]..entry[0]+n].copy_from_slice(subj);
                self.a_cof[entry[0]..entry[0]+n].copy_from_slice(cof);
                self.a_ptr[i][1] = n;
            }
            else {
                self.con_elt.push(self.con_elt[ai]);
                self.a_ptr[ai][1] = 0;
                self.con_a_row[i] = self.a_ptr.len();
                self.a_ptr.push([self.a_subj.len(),n]);
                self.a_subj.extend_from_slice(subj);
                self.a_cof.extend_from_slice(cof);
            }
        }
        Ok(())
    }

    fn write_problem<P>(&self, _filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        Err("Writing problem not supported".to_string())
    }

    fn solve(& mut self, _sol_bas : & mut Solution, _sol_itr : &mut Solution, _sol_itg : &mut Solution) -> Result<(),String>
    {
        unimplemented!("Dummy Backend does not implement solve")
    }

    fn objective(&mut self, _name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.sense_max = match sense { Sense::Maximize => true, Sense::Minimize => false };
        self.c_subj.resize(subj.len(),0); self.c_subj.copy_from_slice(subj);
        self.c_cof.resize(cof.len(),0.0); self.c_cof.copy_from_slice(cof);
        Ok(())
    }

    fn set_parameter<V>(&mut self, _parname : V::Key, _parval : V) -> Result<(),String> where V : SolverParameterValue<Self>,Self: Sized
    {
        Err("Parameters not supported".to_string())
    }
}



impl ModelWithLogCallback for Backend {
    /// Attach a log printer callback to the Backend. This will receive messages from the solver
    /// while solving and during a few other calls like file reading/writing. 
    ///
    /// # Arguments
    /// - `func` A function that will be called with strings from the log. Individual lines may be
    ///   written in multiple chunks to there is no guarantee that the strings will end with a
    ///   newline.
    fn set_log_handler<F>(& mut self, _func : F) where F : 'static+Fn(&str) {
        // do nothing
    }
}

impl ModelWithIntSolutionCallback for Backend {
    /// Attach a solution callback function. This is called for each new integer solution 
    fn set_solution_callback<F>(&mut self, mut _func : F) where F : 'static+FnMut(f64,&[f64],&[f64]) {
        // do nothing
    }
}

impl ModelWithControlCallback for Backend {
    fn set_callback<F>(&mut self, mut _func : F) where F : 'static+FnMut() -> ControlFlow<(),()> {
        // do nothing
    }
}



trait VectorConeForDummy : VectorDomainTrait { fn to_conetype(self) -> ConeType; }
impl VectorConeForDummy for QuadraticCone {fn to_conetype(self) -> ConeType { match self { Self::Normal => ConeType::QuadraticCone, Self::Rotated => ConeType::RoteatedQuadraticCone }} }
impl VectorConeForDummy for SVecPSDCone { fn to_conetype(self) -> ConeType { ConeType::SVecPSDCone }  }
impl VectorConeForDummy for GeometricMeanCone { fn to_conetype(self) -> ConeType { ConeType::GeometricMeanCone }  }
impl VectorConeForDummy for PowerCone { fn to_conetype(self) -> ConeType { ConeType::PowerCone(self.0)}  }
impl VectorConeForDummy for ExponentialCone { fn to_conetype(self) -> ConeType { ConeType::ExponentialCone } }

impl<D> VectorConeModelTrait<D> for Backend where D : VectorConeForDummy+'static {
    fn conic_constraint<const N : usize>
       (& mut self, 
        name   : Option<&str>, 
        dom    : VectorDomain<N,D>,
        _shape : &[usize], 
        ptr    : &[usize], 
        subj   : &[usize], 
        cof    : &[f64]) -> Result<Constraint<N>,String> 
    {
        let (ct,offset,shape,conedim,_is_integer) = dom.dissolve();
        let dt = ct.to_conetype();

        let (d0,d,d1) = (shape[..conedim].iter().product(),shape[conedim],shape[conedim+1..].iter().product());
        let n = d0*d*d1;
        let ncones = d0*d1;

        let a_row0 = self.a_ptr.len()-1;

        let first = self.con_elt.len();
        let last  = first+n;

        let firstcon = self.cons.len();
        let lastcon = firstcon+n;

        let firstcone = self.cones.len();

        self.cones.reserve(ncones);
        for _ in 0..ncones { self.cones.push(dt.clone()) }

        self.con_elt.reserve(n);        

        self.a_ptr.reserve(n);
        for (b,n) in ptr.iter().zip(ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
            self.a_ptr.push([b,n]);
        }

        let con0 = self.cons.len();
        self.a_subj.extend_from_slice(subj);
        self.a_cof.extend_from_slice(cof);
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }

        for (i0,i,i1,&b) in iproduct!(0..d0,0..d,0..d1,offset.iter()) {
            self.con_elt.push(Element::Conic { coneidx: firstcone+i0*d1+i1, offset: i, b });
        }
                 
        self.cons.reserve(n);
        for index in first..last {
           self.cons.push(Item::Conic{index});
        }

        Ok(Constraint::new((firstcon..firstcon+n).collect::<Vec<usize>>(), &shape))
    }


    fn conic_variable<const N : usize>(&mut self, name : Option<&str>,dom : VectorDomain<N,D>) -> Result<Variable<N>,String> {
        let (ct,offset,shape,conedim,is_integer) = dom.dissolve();
        let dt = ct.to_conetype();

        let (d0,d,d1) = (shape[..conedim].iter().product(),shape[conedim],shape[conedim+1..].iter().product());
        let n = d0*d*d1;
        let ncones = d0*d1;

        let first = self.var_elt.len();
        let last  = first+n;

        let firstvar = self.vars.len();
        let lastvar = firstvar+n;
        
        let firstcone = self.cones.len();
        
        self.cones.reserve(ncones);
        for _ in 0..ncones { self.cones.push(dt.clone()) }

        self.var_int.resize(last,is_integer);
        self.var_elt.reserve(n);

        for (i0,i,i1,&b) in iproduct!(0..d0,0..d,0..d1,offset.iter()) {
            self.var_elt.push(Element::Conic { coneidx: firstcone+i0*d1+i1, offset: i, b });
        }
                 
        self.vars.reserve(n);
        for index in first..last {
           self.vars.push(Item::Conic{index});
        }

        Ok(Variable::new((firstvar..firstvar+n).collect::<Vec<usize>>(), None, &shape))
   }
}


