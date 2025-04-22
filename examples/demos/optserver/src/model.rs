//! 
//! This project demonstrates how to implement an alternative solver backend for [MosekAPI]. In
//! this case, the backend is an OptServer instance communicating over HTTP.
//!

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

enum VarItem {
    Linear{index:usize},
    RangedUpper{index:usize},
    RangedLower{index:usize},
    Conic{index:usize,con_block_index: usize, con_block_offset: usize},
}
struct ConItem {
    block_i: usize, 
    block_entry: usize
}

#[derive(Default)]
pub struct ModelOptserver {
    name : Option<String>,
    hostname : String,
    access_token       : Option<String>,

    var_range_lb  : Vec<f64>,
    var_range_ub  : Vec<f64>,
    var_range_int : Vec<bool>,

    vars          : Vec<VarItem>,

    con_blocks    : Vec<Block>,

    a_ptr      : Vec<[usize;2]>,
    a_subj     : Vec<usize>,
    a_cof      : Vec<f64>,

    con_rhs          : Vec<f64>,
    con_a_row        : Vec<usize>, // index into a_ptr
    con_block_i      : Vec<usize>,
    con_block_offset : Vec<usize>,

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

    fn write_jtask(&self,f : &mut std::fs::File) -> std::io::Result<usize> {
        unimplemented!();
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
            self.vars.push(VarItem::Linear{index:i});
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
        for i in first..last { self.vars.push(VarItem::Linear{index:i}) }
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
        for i in first..last { self.vars.push(VarItem::RangedLower{index:i}) }
        for i in first..last { self.vars.push(VarItem::RangedUpper{index:i}) }
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

        let a_row0 = self.a_ptr.len()-1;
        let con_row0 = self.con_rhs.len();
        let block_i = self.con_blocks.len();

        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        {
            for (b,n) in ptr.iter().zip(ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
                self.a_ptr.push([b,n]);
            }
        }

        {
            let n0 = self.a_subj.len();
            self.a_subj.resize(n0+subj.len(),0);
            self.a_cof.resize(n0+cof.len(),0.0);
            self.a_subj[n0..].copy_from_slice(subj);
            self.a_cof[n0..].copy_from_slice(cof);
        }

        self.con_rhs.resize(con_row0+n,0.0);
        self.con_rhs[con_row0..].copy_from_slice(b.as_slice());
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }
        self.con_block_i.resize(con_row0+n, block_i);
        self.con_block_offset.reserve(n); for i in 0..n { self.con_block_offset.push(i); } 
        
        match dt {
            LinearDomainType::Zero => {
                self.con_blocks.push(Block{ ct : ConeType::Fixed,       first : con_row0, block_size : n });
            },
            LinearDomainType::Free => { 
                self.con_blocks.push(Block{ ct : ConeType::Unbounded,   first : con_row0, block_size : n });
            },
            LinearDomainType::NonNegative => {
                self.con_blocks.push(Block{ ct : ConeType::Nonnegative, first : con_row0, block_size : n });
            },
            LinearDomainType::NonPositive => {
                self.con_blocks.push(Block{ ct : ConeType::Nonpositive, first : con_row0, block_size : n });
            },
        }

        Ok(Constraint::new((con_row0..con_row0+n).collect::<Vec<usize>>(), &shape))
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
        let con_row0 = self.con_rhs.len();
        let block_i = self.con_blocks.len();

        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        for (b,n) in izip!(ptr.iter(),ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
            self.a_ptr.push([b,n]);
        }

        {
            let n0 = self.a_subj.len();
            self.a_subj.resize(n0+subj.len(),0);
            self.a_cof.resize(n0+cof.len(),0.0);
            self.a_subj[n0..].copy_from_slice(subj);
            self.a_cof[n0..].copy_from_slice(cof);
        }

        self.con_rhs.resize(con_row0+2*n,0.0);
        self.con_rhs[con_row0..con_row0+n].copy_from_slice(bl.as_slice());
        self.con_rhs[con_row0+n..con_row0+2*n].copy_from_slice(bu.as_slice());
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }
        self.con_block_i.resize(con_row0+n, block_i);
        self.con_block_i.resize(con_row0+n, block_i+1);
        self.con_block_offset.reserve(2*n); 
        for i in 0..n { self.con_block_offset.push(i); }
        for i in 0..n { self.con_block_offset.push(i); }

        self.con_blocks.push(Block{ ct : ConeType::Nonnegative, first : con_row0, block_size : n });
        self.con_blocks.push(Block{ ct : ConeType::Nonpositive, first : con_row0+n, block_size : n });

        Ok((Constraint::new((con_row0..con_row0+n).collect::<Vec<usize>>(), &shape),
            Constraint::new((con_row0+n..con_row0+2*n).collect::<Vec<usize>>(), &shape)))
    }

    fn update(& mut self, idxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>
    {
        if shape.iter().product::<usize>() != idxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = idxs.iter().max() {
            if i >= self.con_rhs.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

//        if let Some(sp) = sp {
//            let mut it = izip!(sp.iter(),subj.chunks_ptr(ptr),cof.chunks_ptr(ptr)).peekable();
//            for &i in idxs.iter() {
//                if let Some((_,subj,cof)) = it.peek().and_then(|v| if *(v.0) == i { Some(v) } else { None }) {
//                    let n = subj.len();
//                    let entry = self.a_ptr[i];
//                    if entry[1] >= n {
//                        self.a_subj[entry[0]..entry[0]+n].copy_from_slice(subj);
//                        self.a_cof[entry[0]..entry[0]+n].copy_from_slice(cof);
//                        self.a_ptr[i] = [entry[0],n];
//                    }
//                    else {
//                        let p0 = self.a_subj.len();
//                        self.a_subj.extend_from_slice(subj);
//                        self.a_cof.extend_from_slice(cof);
//                        self.a_ptr[i] = [p0,n];
//                    }
//                }
//                else {
//                    self.a_ptr[i] = [0,0];
//                }
//            }
//
//        }
//        else
        {
            for (subj,cof,&i) in izip!(subj.chunks_ptr(ptr),cof.chunks_ptr(ptr),idxs.iter()) {
                let n = subj.len();
                let entry = self.a_ptr[i];
                if entry[1] >= n {
                    self.a_subj[entry[0]..entry[0]+n].copy_from_slice(subj);
                    self.a_cof[entry[0]..entry[0]+n].copy_from_slice(cof);
                    self.a_ptr[i] = [entry[0],n];
                }
                else {
                    let p0 = self.a_subj.len();
                    self.a_subj.extend_from_slice(subj);
                    self.a_cof.extend_from_slice(cof);
                    self.a_ptr[i] = [p0,n];
                }
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


