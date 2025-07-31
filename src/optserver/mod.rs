//! This module implements a backend that can offload to a mosek OptServer. 
//!
//! It works by 
use crate::*;
use crate::model::{DJCDomainTrait, DJCModelTrait, ModelWithIntSolutionCallback, ModelWithLogCallback, PSDModelTrait, VectorConeModelTrait};
use crate::domain::*;
use crate::utils::iter::{ChunksByIterExt, PermuteByEx, PermuteByMutEx};
use itertools::{iproduct, izip};
use std::fs::{write, File};
use std::net::TcpStream;
use std::path::Path;

mod http;
mod json;

pub type Model = ModelAPI<Backend<DefaultURIOpener>>;

#[derive(Clone,Copy)]
enum Item {
    Linear{index:usize},
    RangedUpper{index:usize},
    RangedLower{index:usize},
}

impl Item {
    fn index(&self) -> usize { 
        match self {
            Item::Linear { index } => *index,
            Item::RangedUpper { index } => *index,
            Item::RangedLower { index } => *index
        }
    } 
}

#[derive(Clone,Copy)]
struct Element {
    lb:f64,ub:f64
}



pub trait URIOpener {
    fn connect(&mut self, uri : String) -> std::io::Result<TcpStream>;
}

#[derive(Default)]
pub struct DefaultURIOpener {}

impl URIOpener for DefaultURIOpener {
    fn connect(&mut self, uri : String) -> std::io::Result<TcpStream> {
        todo!("Implement connect")
    }
}

/// Simple model object that supports input of linear, conic and disjunctive constraints. It only
/// stores data, it does not support solving or writing problems.
#[derive(Default)]
pub struct Backend<T> where T : URIOpener {
    name : Option<String>,

    var_elt       : Vec<Element>, // Either lb,ub,int or index,coneidx,offset
    var_int       : Vec<bool>,

    vars          : Vec<Item>,
    var_names     : Vec<Option<String>>,

    a_ptr         : Vec<[usize;2]>,
    a_subj        : Vec<usize>,
    a_cof         : Vec<f64>,

    con_elt       : Vec<Element>,
    con_a_row     : Vec<usize>, // index into a_ptr
    cons          : Vec<Item>,
    con_names     : Vec<Option<String>>,


    sense_max     : bool,
    c_subj        : Vec<usize>,
    c_cof         : Vec<f64>,

    url_opener    : T,
    address       : String
}

impl<T> BaseModelTrait for Backend<T> where T : URIOpener+Default {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name : name.map(|v| v.to_string()),
            var_elt    : Default::default(), 
            var_int    : Default::default(), 

            vars       : Default::default(), 
            var_names  : Default::default(),

            a_ptr      : Default::default(), 
            a_subj     : Default::default(), 
            a_cof      : Default::default(), 

            con_elt    : Default::default(), 
            con_a_row  : Default::default(), 
            cons       : Default::default(), 
            con_names  : Default::default(),

            sense_max  : Default::default(), 
            c_subj     : Default::default(), 
            c_cof      : Default::default(), 

            url_opener : Default::default(), 
            address    : Default::default(), 
        }
    }
    fn free_variable<const N : usize>
        (&mut self,
         name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();
        let first = self.var_elt.len();
        let last  = first + n;

        self.var_elt.resize(last,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY });
        self.var_int.resize(last,false);
        self.var_names.resize(last,None);

        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last {
            self.vars.push(Item::Linear{index:i});
        }

        if let Some(name) = name {
            (0..n).scan([0usize;N],|i,_| { 
                let r = format!("{}{:?}",name,i); 
                i.iter_mut().zip(shape.iter()).rev().fold(1,|c,(v,&d)| { *v += c; if *v >= d { *v = 0; 1 } else { 0 }}); 
                Some(r)
            })
                .zip(self.var_names[first..last].iter_mut())
                .for_each(|(n,vn)| *vn = Some(n));
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
                self.var_elt.resize(last,Element{ lb: 0.0, ub: 0.0 });
            },
            LinearDomainType::Free => {
                self.var_elt.resize(last,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY});
            },
            LinearDomainType::NonNegative => {
                self.var_elt.reserve(last);
                for lb in b {
                    self.var_elt.push(Element{ lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.var_elt.reserve(last);
                for ub in b {
                    self.var_elt.push(Element{ lb : f64::NEG_INFINITY, ub });
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
            self.var_elt.push(Element{ lb, ub })
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

        if let Some(sp) = &sp {
            assert_eq!(b.len(),sp.len()); 
        }
        else {
            assert_eq!(b.len(),ptr.len()-1); 
        }
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
                    self.con_elt.push(Element{ lb: b, ub: b });
                }
            },
            LinearDomainType::Free => { 
                self.con_elt.resize(con_row0+nrow,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY });
            },
            LinearDomainType::NonNegative => {
                self.con_elt.reserve(con_row0+nrow);
                for lb in b {
                    self.con_elt.push(Element{ lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.con_elt.reserve(con_row0+nrow);
                for ub in b {
                    self.con_elt.push(Element{ lb : f64::NEG_INFINITY, ub });
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
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.con_elt.push(Element{ lb, ub });
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

    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        let p = filename.as_ref();
        if let Some(ext) = p.extension().and_then(|ext| ext.to_str()) {
            if let std::cmp::Ordering::Equal = ext.cmp("ptf") {
                let mut f = File::create_new(p).map_err(|e| e.to_string())?;
                self.format_json_to(&mut f).map_err(|e| e.to_string())
            }
            else {
                Err("Writing problem not supported".to_string())
            }
        }
        else {
            Err("Writing problem not supported".to_string())
        }
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

fn bnd_to_bk(lb : f64, ub : f64) -> &'static str {
    match (lb.is_finite(),ub.is_finite()) {
        (false,false) => "fr",
        (false,true)  => "up",
        (true,false)  => "lo",
        (true,true)   => if lb < ub { "ra" } else { "fx" }
    }
}
impl<T> Backend<T> where T : URIOpener {
    /// JSON Task format writer.
    ///
    /// See https://docs.mosek.com/latest/capi/json-format.html
    fn format_json_to<S>(&self, strm : &mut S) -> std::io::Result<()> 
        where 
            S : std::io::Write 
    {
        use json::JSON;
        
        let mut top = json::Dict::new();
        top.append("$schema",JSON::String("http://mosek.com/json/schema#".to_string()));

        if let Some(name) = &self.name {
            top.append("Task/name", name.clone());
        }

        top.append(
            "Task/info",
            json::Dict::from(|taskinfo| {
                taskinfo.append("numvar",self.vars.len() as i64);
                taskinfo.append("numcon",self.con_elt.len() as i64);
            }));
            
        top.append(
            "Task/data",
            json::Dict::from(|taskdata| {
                taskdata.append("var",json::Dict::from(|d| {
                    d.append("bk",  JSON::List(self.var_elt.iter().map(|e| bnd_to_bk(e.lb,e.ub).into()).collect()));
                    d.append("bl",  JSON::List(self.var_elt.iter().map(|&e| JSON::Float(e.lb)).collect()));
                    d.append("bu",  JSON::List(self.var_elt.iter().map(|&e| JSON::Float(e.ub)).collect()));
                    d.append("type",JSON::List(self.var_int.iter().map(|&e| if e { "int".into() } else { "cont".into() }).collect()));
                }));
                taskdata.append("con",json::Dict::from(|d| {
                    d.append("bk",  JSON::List(self.con_elt.iter().map(|e| bnd_to_bk(e.lb,e.ub).into()).collect()));
                    d.append("bl",  JSON::List(self.con_elt.iter().map(|e| JSON::Float(e.lb)).collect()));
                    d.append("bu",  JSON::List(self.con_elt.iter().map(|e| JSON::Float(e.ub)).collect()));
                }));
                taskdata.append(
                    "objective",
                    json::Dict::from(|d| {
                        d.append("sense", if self.sense_max { "max" } else { "min" });
                        d.append("cfix",0.0f64);
                        d.append("c", json::Dict::from(|d2| {
                            d2.append("subj",JSON::List(self.c_subj.iter().map(|&i| JSON::Int(i as i64)).collect()));
                            d2.append("val", self.c_cof.as_slice());
                    }));
                }));
                taskdata.append(
                    "A", 
                    json::Dict::from(|d| {
                        d.append("subi",JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_subj[row[0]..row[0]+row[1]].iter()).map(|&i| JSON::Int(i as i64)).collect()));
                        d.append("subj",JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).enumerate().flat_map(|(i,row)| std::iter::repeat(i).take(row[1])).map(|i| JSON::Int(i as i64)).collect()));
                        d.append("val", JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_cof[row[0]..row[0]+row[1]].iter()).map(|&d| JSON::Float(d)).collect()));
                }));
        }));

        Ok(())
    }
}

