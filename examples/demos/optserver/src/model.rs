//! 
//! This project demonstrates how to implement an alternative solver backend for [MosekAPI]. In
//! this case, the backend is an OptServer instance communicating over HTTP.
//!

use std::fmt::Display;
use std::io::Write;
use std::path::Path;
use mosekcomodel::*;
use mosekcomodel::domain::LinearRangeDomain;
use itertools::izip;

use mosekcomodel::utils::iter::ChunksByIterExt;
use mosekcomodel::model::Solution;

enum ConeType {
    Unbounded,
    Fixed,
    Nonnegative,
    Nonpositive,
}

struct Block {
    ct         : ConeType,
    first      : usize,
    block_size : usize,
}

struct ConElement {
    block_i     : usize, // index into con_blocks
    block_entry : usize, // offset into the indexed block
}

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

fn fmt_json_list<I : IntoIterator>(dst : & mut String, v : I) where I::Item : Display {
    let mut it = v.into_iter();
    dst.push('[');
    if let Some(item) = it.next() {
    dst.push_str(format!("{}",item).as_str());
    for item in it {
            dst.push_str(format!(",{}",item).as_str());
        }
    }
    dst.push(']');
}

#[derive(Default)]
pub struct ModelOptserver {
    name : Option<String>,
    hostname : String,
    access_token  : Option<String>,

    var_range_lb  : Vec<f64>,
    var_range_ub  : Vec<f64>,
    var_range_int : Vec<bool>,

    vars          : Vec<Item>,

    a_ptr      : Vec<[usize;2]>,
    a_subj     : Vec<usize>,
    a_cof      : Vec<f64>,
    con_lb           : Vec<f64>,
    con_ub           : Vec<f64>,

    con_a_row     : Vec<usize>, // index into a_ptr
    cons          : Vec<Item>,

    sense_max : bool,
    c_subj : Vec<usize>,
    c_cof  : Vec<f64>,

    double_param : Vec<(String,f64)>,
    int_param    : Vec<(String,i32)>,

    //rs : WorkStack,
    //ws : WorkStack,
    //xs : WorkStack,
}

impl ModelOptserver {
    pub fn new(name : Option<&str>,  hostname : &str, access_token : Option<&str>) -> ModelOptserver {
        ModelOptserver {
            name         : name.map(|v| v.to_string()),
            hostname     : hostname.to_string(),
            access_token : access_token.map(|v| v.to_string()),

            ..Default::default()
        }
    }

    fn bnd2bk(bl : f64,bu : f64) -> &'static str {
        match (bl > f64::NEG_INFINITY, bu < f64::INFINITY) {
            (false,false) => "fr",
            (true,false) => "lo",
            (false,true) => "up",
            (true,true) => if bl < bu { "ra" } else { "fx" }
        }
    }
    fn format_to(&self, dst : &mut String) -> Result<(),String> {
        let annz : usize = self.con_a_row.iter().map(|&i| self.a_ptr[i][1] ).sum();


        dst.push_str("{\"$schema\":\"http://mosek.com/json/schema#\"");
        dst.push_str(format!(",\"Task/INFO\":{{numvar:{},numcon:{},numanz:{}}}",self.vars.len(),self.con_a_row.len(),annz).as_str());
        dst.push_str(",\"Task/data\":{");
        dst.push_str("\"var\":{");
        dst.push_str("\"bk\":"); 
        fmt_json_list(dst, 
                      self.var_range_lb.iter().zip(self.var_range_ub.iter())
                        .map(|(&bl,&bu)| Self::bnd2bk(bl,bu) ));
        dst.push_str(",\"bl\":"); fmt_json_list(dst, self.var_range_lb.as_slice());
        dst.push_str(",\"bu\":"); fmt_json_list(dst, self.var_range_ub.as_slice());
        if self.var_range_int.iter().any(|&v| v) {  
            dst.push_str(",\"type\":"); fmt_json_list(dst, self.var_range_int.iter().map(|&v| if v { "true" } else { "false" }));
        }
        dst.push_str("}"); // var
        

        dst.push_str(",\"con\":{");
        dst.push_str("\"bk\":"); 
        fmt_json_list(dst, 
                      self.con_lb.iter().zip(self.con_ub.iter())
                        .map(|(&bl,&bu)| Self::bnd2bk(bl,bu) ));
        dst.push_str(",\"bl\":"); fmt_json_list(dst, self.con_lb.as_slice());
        dst.push_str(",\"bu\":"); fmt_json_list(dst, self.con_ub.as_slice());
        dst.push_str("}"); // con

        dst.push_str(",\"obj\":{");
        dst.push_str(if self.sense_max { ",\"sense\":\"max\"" } else { ",\"sense\":\"min\""});
        dst.push_str(",\"c\":{");
        dst.push_str("\"subj\":"); fmt_json_list(dst,self.c_subj.iter().map(|&i| self.vars[i].index()));
        dst.push_str(",\"cof\":"); fmt_json_list(dst,self.c_cof.iter());
        dst.push_str("}"); // c

        dst.push_str(",\"A\":{");
        dst.push_str("\"subi\":");  fmt_json_list(dst,self.con_a_row.iter().enumerate().flat_map(|(i,&k)| std::iter::repeat(i).take(self.a_ptr[k][1])));
        dst.push_str(",\"subj\":"); fmt_json_list(dst,self.con_a_row.iter().flat_map(|&k| { let entry = self.a_ptr[k]; self.a_subj[entry[0]..entry[0]+entry[1]].iter() }));
        dst.push_str(",\"cof\":");  fmt_json_list(dst,self.con_a_row.iter().flat_map(|&k| { let entry = self.a_ptr[k]; self.a_cof[entry[0]..entry[0]+entry[1]].iter() }));
        dst.push_str("}"); // A
        dst.push_str("}"); // Task/data
        dst.push_str("}"); // $schema
        Ok(())  
    }

    fn write_jtask(&self,f : &mut std::fs::File) -> Result<usize,String> {
        let mut data = String::new();
        self.format_to(&mut data)?;
        f.write(data.as_ref()).map_err(|err| err.to_string())
    }
}




impl BaseModelTrait for ModelOptserver {
    fn new(name : Option<&str>) -> Self {
        ModelOptserver::new(name,"",None)
    }

    fn free_variable<const N : usize>
        (&mut self,
         _name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();
        let first = self.var_range_lb.len();
        let last  = first + n;

        self.var_range_lb.resize(last,f64::NEG_INFINITY);
        self.var_range_ub.resize(last,f64::INFINITY);
        self.var_range_int.resize(last,false);

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
        let first = self.var_range_lb.len();
        let last  = first + n;


        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last { self.vars.push(Item::Linear{index:i}) }
        match dt {
            LinearDomainType::Zero => {
                self.var_range_lb.resize(last,0.0);
                self.var_range_ub.resize(last,0.0);
            },
            LinearDomainType::Free => {
                self.var_range_lb.resize(last,f64::NEG_INFINITY);
                self.var_range_ub.resize(last,f64::INFINITY);
            },
            LinearDomainType::NonNegative => {
                self.var_range_lb.resize(last,0.0);
                self.var_range_lb[first..last].copy_from_slice(b.as_slice());
                self.var_range_ub.resize(last,f64::INFINITY);
            },
            LinearDomainType::NonPositive => {
                self.var_range_lb.resize(last,f64::NEG_INFINITY);
                self.var_range_ub.resize(last,0.0);
                self.var_range_ub[first..last].copy_from_slice(b.as_slice());
            },
        }
        self.var_range_int.resize(last,is_integer);

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, _name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (shape,bl,bu,sp,is_integer) = dom.dissolve();

        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_range_lb.len();
        let last  = first + n;

        let ptr0 = self.vars.len();
        let ptr1 = self.vars.len()+n;
        let ptr2 = self.vars.len()+2*n;
        self.vars.reserve(n*2);
        for i in first..last { self.vars.push(Item::RangedLower{index:i}) }
        for i in first..last { self.vars.push(Item::RangedUpper{index:i}) }
        self.var_range_lb.resize(last,0.0);
        self.var_range_ub.resize(last,0.0);
        self.var_range_int.resize(last,is_integer);

        self.var_range_lb[ptr0..ptr1].copy_from_slice(bl.as_slice());
        self.var_range_ub[ptr1..ptr2].copy_from_slice(bu.as_slice());

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

        let a_row0 = self.a_ptr.len()-1;
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
                self.con_lb.extend_from_slice(b.as_slice());
                self.con_ub.extend_from_slice(b.as_slice());
            },
            LinearDomainType::Free => { 
                self.con_lb.resize(con_row0+nrow,f64::NEG_INFINITY);
                self.con_ub.resize(con_row0+nrow,f64::INFINITY);
            },
            LinearDomainType::NonNegative => {
                self.con_lb.extend_from_slice(b.as_slice());
                self.con_ub.resize(con_row0+nrow,f64::INFINITY);
            },
            LinearDomainType::NonPositive => {
                self.con_lb.resize(con_row0+nrow,f64::NEG_INFINITY);
                self.con_ub.extend_from_slice(b.as_slice());
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
        self.con_lb.extend_from_slice(bl.as_slice());
        self.con_ub.extend_from_slice(bu.as_slice());

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
                self.a_ptr[ai][1] = 0;
                let lb = self.con_lb[ai];
                let ub = self.con_ub[ai];
                self.con_a_row[i] = self.a_ptr.len();
                self.con_lb.push(lb);
                self.con_ub.push(ub);
                self.a_ptr.push([self.a_subj.len(),n]);
                self.a_subj.extend_from_slice(subj);
                self.a_cof.extend_from_slice(cof);
            }
        }
        Ok(())
    }

    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        let path : &Path = filename.as_ref();
        if let Some(ext) = path.extension() {
            if ext.eq(".jtask") {
                let mut f = std::fs::File::create(filename).map_err(|err| err.to_string())?;
                _ = self.write_jtask(&mut f).map_err(|err| err.to_string())?;
                Ok(())
            }
            else {
                Err(format!("File type not supported: {:?}",ext))
            }
        }
        else {
            Err(format!("File type not supported for {:?}",path))
        }
    }

    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, solitg : &mut Solution) -> Result<(),String>
    {
        unimplemented!();
    }

    fn objective(&mut self, name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.c_subj.resize(subj.len(),0); self.c_subj.copy_from_slice(subj);
        self.c_cof.resize(cof.len(),0.0); self.c_cof.copy_from_slice(cof);
        Ok(())
    }

    fn set_parameter<V>(&mut self, parname : V::Key, parval : V) -> Result<(),String> where V : SolverParameterValue<Self>,Self: Sized
    {
        parval.set(parname,self)
    }
}

impl SolverParameterValue<ModelOptserver> for f64 {
    type Key = &'static str;
    fn set(self,parname : Self::Key, model : & mut ModelOptserver) -> Result<(),String> {
        model.double_param.push((parname.to_string(), self));
        Ok(())
    }
}

impl SolverParameterValue<ModelOptserver> for i32 {
    type Key = &'static str;
    fn set(self,parname : Self::Key, model : & mut ModelOptserver) -> Result<(),String> {
        model.int_param.push((parname.to_string(),self));
        Ok(())
    }
}


