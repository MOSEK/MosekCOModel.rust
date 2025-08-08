//! This module implements a backend that uses a MOSEK OptServer instance for solving, for example
//! [solve.mosek.com:30080](http://solve.mosek.com). 
//!
use mosekcomodel::*;
use mosekcomodel::model::ModelWithLogCallback;
use mosekcomodel::utils::iter::{ChunksByIterExt, PermuteByEx, PermuteByMutEx};
use itertools::izip;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

//mod http;
mod json;
mod bio;
mod pipe;

pub type Model = ModelAPI<Backend>;

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

/// Simple model object that supports input of linear, conic and disjunctive constraints. It only
/// stores data, it does not support solving or writing problems.
#[derive(Default)]
pub struct Backend {
    name : Option<String>,

    log_cb        : Option<Box<dyn Fn(&str)>>,
    sol_cb        : Option<Box<dyn FnMut(f64,&[f64],&[f64])>>,

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

    address       : Option<reqwest::Url>,
}

impl BaseModelTrait for Backend {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name : name.map(|v| v.to_string()),
            log_cb     : None,
            sol_cb     : None,
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

            address    : None,
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
            match ext {
                "json"|"jtask" => {
                    let mut f = File::create(p).map_err(|e| e.to_string())?;
                    self.write_jtask(&mut f).map_err(|e| e.to_string())
                },
                _ => Err("Writing problem not supported".to_string())
            }
        }
        else {
            Err("Writing problem not supported".to_string())
        }
    }


    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution) -> Result<(),String>
    {
        if let Some(address) = self.address {
        
            let mut client = reqwest::blocking::Client::new();

            let (mut req_r,mut req_w) = pipe::new();
            let (mut resp_r,mut resp_w) = pipe::new();

            let t = std::thread::spawn(move || {
                let resp = client.post(address)
                    .header("Content-Type", "application/x-mosek-b")
                    .header("Accept", "application/x-mosek-multiplex")
                    .header("X-Mosek-Callback", "values")
                    .header("X-Mosek-Stream", "log")
                    .body(reqwest::blocking::Body::new(req_r))
                    .send()
                    .map_err(|e| e.to_string())?
                    ;

                if ! resp.status().is_success() {
                    return Err(format!("OptServer responded with code {}",resp.status()));
                }

                for (k,v) in resp.headers().iter() {
                    if k.as_str().eq_ignore_ascii_case("Content-Type") {
                        if ! v.as_bytes().eq(b"application/x-mosek-multiplex") {
                            return Err(format!("Unsupported solution format: {}",std::str::from_utf8(v.as_bytes()).unwrap_or("<invalid utf-8>")));
                        }
                    }
                }
                resp.copy_to(&mut resp_w).map_err(|e| e.to_string())?;

                Ok(())
            });

            self.write_btask(&mut req_w)?;
                
            // We should now receive an application/x-mosek-multiplex stream ending with either a
            // fail or a result.
            // parse incoming stream 
            let mut buf = Vec::new();

            loop { // loop over messages
                // for each message loop over frames
                buf.clear();
                {
                    let mut mr = MessageReader::new(&mut resp);
                 
                    mr.read_to_end(&mut buf).map_err(|e| e.to_string())?;

                    let headers_end = subseq_location(buf.as_slice(),b"\n\n").ok_or_else(|| "Invalid response format B".to_string())?;
                    let mut lines = buf[..headers_end].chunk_by(|&a,_| a != b'\n');
                    
                    let head = lines.next().ok_or_else(|| "Invalid response format A".to_string())?.trim_ascii_end();
                    let mut headers = lines
                        .map(|s| s.trim_ascii_end())
                        .map(|s| { if let Some(p) = subseq_location(s, b":") { (&s[..p],&s[p+1..]) } else { (&s[..0],s) } });
                    let body = &buf[headers_end+2..];


                    let mut it = buf.chunk_by(|a,_| *a == b'\n');

                    match head {
                        b"log" => {
                            print!("{}",std::str::from_utf8(body).unwrap_or("<?>"));
                            if let Some(f) = &self.log_cb {
                               if let Ok(s) = std::str::from_utf8(body) {
                                   f(s)                                   
                               }
                            }
                        },
                        b"msg" => {},
                        b"wrn" => {},
                        b"err" => {},
                        b"cbmap" => {},
                        b"cbinfo" => {},
                        b"cbcode" => {},
                        b"cbwarn" => {},
                        b"ok" => {
                            let mut trm = None;
                            for (k,v) in headers {
                                match k {
                                    b"trm" => trm = Some(v),
                                    b"content-type" => {
                                        if v != b"application/x-mosek-b" {
                                            return Err(format!("Unexpected solution format: {}",std::str::from_utf8(v).unwrap_or("<?>")));
                                        }
                                    }
                                    _ => {},
                               }
                            }

                            return self.read_bsolution(sol_bas,sol_itr,sol_itg,body);
                        },
                        b"fail" => { 
                            let mut res = None;
                            for (k,v) in headers {
                                match k {
                                    b"res" => res = Some(v),
                                    _ => {},
                                }
                            }

                            let message = std::str::from_utf8(body).unwrap_or("");
                            return Err(format!("Solve failed ({}): {}",
                                    std::str::from_utf8(res.unwrap_or(b"?")).unwrap_or("?"),
                                    message));
                        },
                        _ => {},
                    }
                }
            }
        }
        else {
            return Err("No optserver address given".to_string());
        }
        t.join()?;

        Ok(())

    } 

    fn objective(&mut self, _name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.sense_max = match sense { Sense::Maximize => true, Sense::Minimize => false };
        self.c_subj.resize(subj.len(),0); self.c_subj.copy_from_slice(subj);
        self.c_cof.resize(cof.len(),0.0); self.c_cof.copy_from_slice(cof);
        Ok(())
    }
}

struct MessageReader<'a,R> where R : Read {
    eof : bool,
    frame_remains : usize,
    final_frame : bool,
    s : & 'a mut R

}
impl<'a,R> MessageReader<'a,R> where R : Read {
    fn new(s : &'a mut R) -> MessageReader<'a,R> { MessageReader { eof: false, frame_remains: 0, s , final_frame : false}}
    pub fn skip(&mut self) -> std::io::Result<()> {
        let mut buf = [0;4096];
        while 0 < self.read(&mut buf)? {}
        Ok(())
    }
}

impl<'a,R> Read for MessageReader<'a,R> where R : Read {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        //println!("MessageReader::read(), eof = {}",self.eof);
        if self.eof {
            Ok(0)
        }
        else {
            if self.frame_remains == 0 {
                if self.final_frame {
                    return Ok(0);
                }
                let mut buf = [0;2];
                self.s.read_exact(&mut buf)?;

                self.final_frame = buf[0] > 127;
                self.frame_remains = (((buf[0] & 0x7f) as usize) << 8) | (buf[1] as usize);
            }

            let n = buf.len().min(self.frame_remains);
            let nr = self.s.read(&mut buf[..n])?;
            self.frame_remains -= nr;
            Ok(nr)
        }
    }
}

pub struct SolverAddress(pub String);

impl SolverParameterValue<Backend> for SolverAddress {
    type Key = ();
    fn set(self,_parname : Self::Key, model : & mut Backend) -> Result<(),String> {
        model.address = Some(reqwest::Url::parse(self.0.as_str())
            .map_err(|e| "Invalid SolverAddress value".to_string())
            .and_then(|mut url| 
                if url.scheme() == "" {
                    url.set_scheme("http").unwrap();
                    Ok(url)
                }
                else if url.scheme().eq_ignore_ascii_case("http") || url.scheme().eq_ignore_ascii_case("https") { 
                    Ok(url) 
                } else {
                    Err(format!("Invalid url scheme: {}",url.as_str()))
                })?);
        Ok(())
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
impl Backend {
    fn copy_solution(&self, 
                     sol : &mut Solution,
                     psta : SolutionStatus,
                     dsta : SolutionStatus,
                     xx  : &[f64],
                     slx : &[f64],
                     sux : &[f64],
                     xc  : &[f64],
                     slc : &[f64],
                     suc : &[f64])
    {
        sol.primal.status = 
            if xx.len() != self.var_elt.len() || xc.len() != self.con_elt.len() { SolutionStatus::Undefined } else { psta };
        sol.dual.status = 
            if slx.len() != self.var_elt.len() ||
               sux.len() != self.var_elt.len() ||  
               slc.len() != self.con_elt.len() ||
               suc.len() != self.con_elt.len()  {
                SolutionStatus::Undefined
            }
            else {
                dsta
            };

        let numvar = self.vars.len();
        let numcon = self.cons.len();

        if let SolutionStatus::Undefined = sol.primal.status {}
        else {           
            sol.primal.var.resize(numvar,0.0);
            sol.primal.con.resize(numcon,0.0);
            for (v,tgt) in self.vars.iter().zip(sol.primal.var.iter_mut()) {
                *tgt = 
                    match v {
                        Item::Linear{index} => xx[*index],
                        Item::RangedUpper{index} => xx[*index],
                        Item::RangedLower{index} => xx[*index],
                    }
            }
            for (c,tgt) in self.cons.iter().zip(sol.primal.con.iter_mut()) {
                *tgt = 
                    match c {
                        Item::Linear{index} => xc[*index],
                        Item::RangedUpper{index} => xc[*index],
                        Item::RangedLower{index} => xc[*index],
                    }
            }
            
        }
        if let SolutionStatus::Undefined = sol.dual.status {} 
        else {
            sol.dual.var.resize(numvar,0.0);
            sol.dual.con.resize(numcon,0.0);
            for (v,tgt) in self.vars.iter().zip(sol.dual.var.iter_mut()) {
                *tgt = 
                    match v {
                        Item::Linear{index} => slx[*index]-sux[*index],
                        Item::RangedUpper{index} => -sux[*index],
                        Item::RangedLower{index} => slx[*index],
                    }
            }
            for (c,tgt) in self.cons.iter().zip(sol.dual.con.iter_mut()) {
                *tgt = 
                    match c {
                        Item::Linear{index} => slc[*index]-suc[*index],
                        Item::RangedUpper{index} => -suc[*index],
                        Item::RangedLower{index} => slc[*index],
                    }
            }
        }
    }

    
    /// MOSEK B fomat
    /// The order of entries are fixed, some may be left out. If the presence is conditional, the
    /// condition is mentioned in brackets after the format. A condition `{C}` means that the entry
    /// is present if and only if `C` is satisfied, while `{C, optional}` means that it _may_ be
    /// present if `C` is satisfied. If no condition is listed, the entry is mandatory.
    ///
    /// # Into section
    /// ```text
    /// INFO/MOSEKVER: III
    /// INFO/name: [B
    /// INFO/numvar: I
    /// INFO/numcon: I
    /// INFO/numcone: I
    /// INFO/numbarvar: I
    /// INFO/numdomain: L
    /// INFO/numafe: L
    /// INFO/numacc: L
    /// INFO/numdjc: L
    /// INFO/numsymmat: L
    /// INFO/atruncatetol: d
    /// ```
    ///
    /// # Data section 
    /// ```text
    /// data/symmat: [B[i[l[i[i[d  {numsymmat > 0}
    /// var/bound:   [B[d[d []     {numvar > 0}    -- bk,lb,ub
    /// data/c:      [B[dd                         -- sense, c, cfix
    /// data/barc:   I[i[l[l[d     {numbarvar > 0, optional} -- nnz, subi,numterm,alpha, if and only if numbarvar > 0
    /// con/bound:   [B[d[d        {numcon > 0}    --
    /// data/A:      [I[i[d                        -- rowlen, subj, valj    
    /// data/bara:   [i[i[l[l[d    {numbarvar > 0, optional}
    /// data/Fg:     [I[l[d[d      {numafe > 0} 
    /// data/barf:   [l[i[i[l[l[d  {numbarvar > 0, optional} -- subi,rowlen,subj,numterm,midx,alpha
    /// data/domain: [B[l[d        {numdomain > 0}
    /// data/acc:    [l[l[l[d      {numacc > 0}
    /// data/djc:    [l[l[l[l[l[d  {numdjc > 0}
    /// ```
    ///
    /// # Names section 
    /// All name lists are a list of `\0`-terminated strings, and entries are only present if they contain at least one name.
    /// ```text
    /// names/obj:    [B         - a single name
    /// names/var:    [B      {numvar > 0}
    /// names/barvar: [B      {numbarvar > 0}
    /// names/con:    [B      {numcon > 0}
    /// names/cone:   [B      {numcone > 0}
    /// names/domain: [B      {numdomain > 0}
    /// names/acc:    [B      {numacc > 0}
    /// names/djc:    [B      {numdjc > 0}
    /// ```
    ///
    /// # Solutions section
    /// ```text
    /// solution/basic/status [B[B       -- prosta, solsta
    /// solution/basic/var [B[D[D[D      {numvar > 0} -- stakey,level,slx,sux
    /// solution/basic/con [B[D[D[D[D    {numcon > 0} -- stakey,level,slc,suc,y
    /// solution/basic/acc [D            {numacc > 0} -- doty
    ///
    /// solution/interior/status [B[B    -- prosta, solsta
    /// solution/interior/var [B[D[D[D   {numvar > 0} -- stakey,level,slx,sux,snx
    /// solution/interior/barvar [D[D    {numbarvar > 0} -- barx,bars
    /// solution/interior/con [B[D[D[D[D {numcon > 0} -- stakey,level,slc,suc,y
    /// solution/interior/acc [D         {numacc > 0} -- doty
    ///
    /// solution/integer/status [B[B     -- prosta, solsta
    /// solution/integer/var [B[D[D[D    {numvar > 0} -- stakey,level
    /// solution/integer/barvar [D[D     {numbarvar > 0} -- barx,bars
    /// solution/integer/con [B[D[D[D[D  {numcon > 0} -- stakey,level
    /// ``` 
    /// 
    /// # Parameters section
    /// ```text
    /// parameter/double: [B[d   {optional}
    /// parameter/integer: [B[i  {optional}
    /// parameter/symbolic: [B[B {optional} # second element is a list of  of '\0'-terminated value strings
    /// ```
    ///
    /// # Bounds indicators
    /// ```text
    /// MSK_BK_FR: 'f'
    /// MSK_BK_FX: 'x'
    /// MSK_BK_LO: 'l'
    /// MSK_BK_UP: 'u'
    /// MSK_BK_RA: 'r'
    /// ```
    /// # Domain type indicators
    /// ```text
    /// MSK_DOMAIN_R:                    'R'
    /// MSK_DOMAIN_RMINUS:               '-'
    /// MSK_DOMAIN_RPLUS:                '+'
    /// MSK_DOMAIN_RZERO:                '0'
    /// MSK_DOMAIN_QUADRATIC_CONE:       'q'
    /// MSK_DOMAIN_RQUADRATIC_CONE:      'r'
    /// MSK_DOMAIN_PRIMAL_EXP_CONE:      'e'
    /// MSK_DOMAIN_DUAL_EXP_CONE:        'x'
    /// MSK_DOMAIN_INF_NORM_CONE:        'i'
    /// MSK_DOMAIN_ONE_NORM_CONE:        '1'
    /// MSK_DOMAIN_PRIMAL_GEO_MEAN_CONE: 'g'
    /// MSK_DOMAIN_DUAL_GEO_MEAN_CONE:   'G'
    /// MSK_DOMAIN_SVEC_PSD_CONE:        'V'
    /// MSK_DOMAIN_PRIMAL_POWER_CONE:    'p'
    /// MSK_DOMAIN_DUAL_POWER_CONE:      'o'
    /// ```
    pub fn write_btask<W>(&self,w : &mut W) -> std::io::Result<()> where W : Write {
        let mut w = bio::Ser::new(w)?;
        // INFO/MOSEKVER: III
        // INFO/name: [B
        // INFO/numvar: I
        // INFO/numcon: I
        // INFO/numcone: I
        // INFO/numbarvar: I
        // INFO/numdomain: L
        // INFO/numafe: L
        // INFO/numacc: L
        // INFO/numdjc: L
        // INFO/numsymmat: L
        // INFO/atruncatetol: d

        {
            let mut e = w.entry(b"INFO/MOSEKVER",b"III")?;
            e.write_value::<u32>(10)?;
            e.write_value::<u32>(0)?;
            e.write_value::<u32>(0)?;
        }
        w.entry(b"INFO/name",b"[B")?.write_array(self.name.map(|s| s.as_bytes()).unwrap_or(b""))?;     
        w.entry(b"INFO/numvar",b"I")?.write_value(self.var_elt.len() as u32)?;
        w.entry(b"INFO/numcon",b"I")?.write_value(self.con_elt.len() as u32)?;
        w.entry(b"INFO/numcone",b"I")?.write_value(0u32)?;
        w.entry(b"INFO/numbarvar",b"I")?.write_value(0u32)?;
        w.entry(b"INFO/numdomain",b"L")?.write_value(0u64)?;
        w.entry(b"INFO/numafe",b"L")?.write_value(0u64)?;
        w.entry(b"INFO/numacc",b"L")?.write_value(0u64)?;
        w.entry(b"INFO/numdjc",b"L")?.write_value(0u64)?;
        w.entry(b"INFO/numsymmat",b"L")?.write_value(0u64)?;
        w.entry(b"INFO/atruncatetol",b"d")?.write_value(0.0)?;

        // data/symmat: [B[i[l[i[i[d  {numsymmat > 0}
        // var/bound:   [B[d[d        {numvar > 0}    -- bk,lb,ub
        // data/c:      [B[dd                         -- sense, c, cfix
        // data/barc:   I[i[l[l[d     {numbarvar > 0, optional} -- nnz, subi,numterm,alpha, if and only if numbarvar > 0
        // con/bound:   [B[d[d        {numcon > 0}    --
        // data/A:      [I[i[d                        -- rowlen, subj, valj    
        // data/bara:   [i[i[l[l[d    {numbarvar > 0, optional}
        // data/Fg:     [I[l[d[d      {numafe > 0} 
        // data/barf:   [l[i[i[l[l[d  {numbarvar > 0, optional} -- subi,rowlen,subj,numterm,midx,alpha
        // data/domain: [B[l[d        {numdomain > 0}
        // data/acc:    [l[l[l[d      {numacc > 0}
        // data/djc:    [l[l[l[l[l[d  {numdjc > 0}
        if self.var_elt.len() > 0 {
            w.entry(b"var/bound", b"[B[d[d")?
                .write_array(self.var_elt.iter().map(bounds_to_bbk).collect::<Vec<u8>>().as_slice())?
                .write_array(self.var_elt.iter().map(|e| e.lb).collect::<Vec<f64>>().as_slice())?
                .write_array(self.var_elt.iter().map(|e| e.ub).collect::<Vec<f64>>().as_slice())?
                ;
        }
        {
            let mut c = vec![0.0; self.var_elt.len()];
            c.permute_by_mut(&self.c_subj.as_slice()).zip(self.c_cof.iter()).for_each(|(d,&s)| *d = s);
            w.entry(b"data/c",b"[B[d[d")?
                .write_array(if self.sense_max {b"maximize"} else {b"minimize"})?
                .write_array(c.as_slice())?
                .write_value(0.0)?
                ;
        }

        w.entry(b"con/bound",b"[B[d[d")?
            .write_array(self.con_elt.iter().map(bounds_to_bbk).collect::<Vec<u8>>().as_slice())?
            .write_array(self.con_elt.iter().map(|e| e.lb).collect::<Vec<f64>>().as_slice())?
            .write_array(self.con_elt.iter().map(|e| e.ub).collect::<Vec<f64>>().as_slice())?
            ;

        {
            let a_row_len : Vec<u32> = self.a_ptr.permute_by(self.con_a_row.as_slice()).map(|row| row[1] as u32).collect();
            let numanz = a_row_len.iter().sum();
            let a_subj : Vec<i32> = self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_subj[row[0]..row[0]+row[1]].iter()).collect();
            let a_cof  : Vec<f64> = self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_cof[row[0]..row[0]+row[1]].iter()).collect();
            
            w.entry(b"data/A",b"[I[i[d")?
                .write_array(a_row_len.as_slice())?
                .write_array(a_subj.as_slice())?
                .write_array(a_cof.as_slice())?
                ;
        }

        // names/obj:    [B                      - a single name
        // names/var:    [B      {numvar    > 0}
        // names/barvar: [B      {numbarvar > 0}
        // names/con:    [B      {numcon    > 0}
        // names/cone:   [B      {numcone   > 0}
        // names/domain: [B      {numdomain > 0}
        // names/acc:    [B      {numacc    > 0}
        // names/djc:    [B      {numdjc    > 0}
        
        w.entry(b"names/obj",b"[B")?
            .write_array(b"obj")?
            ;
        if self.var_elt.len() > 0 {
            let mut ew = w.entry(b"names/var",b"[B")?.stream_writer::<u8>()?;
            for n in self.var_names.iter() {
                ew.write(n.map(|n| n.as_bytes()).unwrap_or(b""))?;
                ew.write(&[0])?;
            }
            ew.close()?;
        }
        
        if self.con_elt.len() > 0 {
            let mut ew = w.entry(b"names/con",b"[B")?.stream_writer::<u8>()?;
            for n in self.con_names.iter() {
                ew.write(n.map(|n| n.as_bytes()).unwrap_or(b""))?;
                ew.write(&[0])?;
            }
            ew.close();
        }
    
        // parameter/double: [B[d
        // parameter/integer: [B[i
        // parameter/symbolic: [B[B # second element is a list of  of '\0'-terminated value strings
        
        // Do not send parameters

        Ok(())
    }


    fn bsol_array<'b>(data : &'b [u8]) -> Result<(usize,&'b[u8]),String> {
        let b0 = data.get(0).ok_or_else(|| "Invalid solution format".to_string())?;
        let nb = (b0 >> 5) as usize;
        if nb == 7 { return Err("Invalid solution format".to_string()); }
        let mut l = (nb & 0x1f) as usize;
        for &b in data.get(1..l+1).ok_or_else(|| "Invalid solution format".to_string())?.iter() {
            l = l << 8 + b as usize;
        }

        Ok((nb+l+1, &data[nb+1..nb+1+l]))
    }


    fn read_one_bsol<'a>(&self, sol : &mut Solution, bs : &mut bio::BAIO<'a>) -> Result<(),String> {
        let numvar = self.var_elt.len();
        let numcon = self.con_elt.len();
        let (prefix,whichsol,psta,dsta) =
        {
            let mut entry = bs.expect()?;
            let name = entry.name();
            let prefix = name.strip_suffix(b"/status").unwrap();
            let whichsol = prefix.strip_prefix(b"solution/").unwrap();

            if entry.fmt() != b"[B[B" { return Err("Invalid solution format".to_string()); }

            _ = entry.next()?.unwrap();

            let (psta,dsta) = {
                let (dt,solsta) = entry.next()?.unwrap();
                if let DataType::U8 = dt {
                    match solsta {
                        b"unknown" =>            (SolutionStatus::Unknown,SolutionStatus::Unknown),
                        b"optimal" =>            (SolutionStatus::Optimal,SolutionStatus::Optimal),
                        b"prim_feas" =>          (SolutionStatus::Feasible,SolutionStatus::Unknown),
                        b"dual_feas" =>          (SolutionStatus::Unknown,SolutionStatus::Feasible),
                        b"prim_and_dual_feas" => (SolutionStatus::Feasible,SolutionStatus::Feasible),
                        b"prim_infeas_cer" =>    (SolutionStatus::Undefined,SolutionStatus::CertInfeas),
                        b"dual_infeas_cer" =>    (SolutionStatus::CertInfeas,SolutionStatus::Undefined),
                        b"prim_illposed_cer" =>  (SolutionStatus::Undefined,SolutionStatus::CertIllposed),
                        b"dual_illposed_cer" =>  (SolutionStatus::CertIllposed,SolutionStatus::Undefined),
                        b"integer_optimal" =>    (SolutionStatus::Optimal,SolutionStatus::Undefined),
                        _ => return Err("Invalid solution format".to_string())
                    }
                } else { 
                    return Err("Invalid solution format".to_string())
                }
            };
            if let SolutionStatus::Undefined = psta { }
            else {
                sol.primal.var.resize(numvar,0.0);
                sol.primal.con.resize(numcon,0.0);
            }
            if let SolutionStatus::Undefined = dsta { }
            else {
                sol.dual.var.resize(numvar,0.0);
                sol.dual.con.resize(numcon,0.0);
            }

            sol.primal.status = psta;
            sol.dual.status = dsta;

            (prefix,whichsol,psta,dsta)
        };

        // expect ".../var"
        let (xx,sx) = {
            let mut entry = bs.expect()?;
            let name  = entry.name();
            let fmt   = entry.fmt();
            if !name.starts_with(prefix) || !name.ends_with(b"/var") { return Err("Invalid solution format".to_string()); }

            match fmt {
                b"[B[D" => { 
                    // primal solution only
                    _ = entry.next()?.unwrap(); // stakey
                    let (_,bxx) = entry.next()?.unwrap();
                    if bxx.len() != numvar*8 {
                        return Err("Invalid solution format".to_string());
                    }
                    let mut xx = vec![0.0; numvar];
                    unsafe {
                        xx.as_mut_slice().align_to_mut::<u8>().1.copy_from_slice(bxx);
                    }
                    (xx,None)
                },
                b"[B[D[D[D[D" | 
                b"[B[D[D[D" => {
                    // primal and linear dual solution
                    _ = entry.next()?.unwrap(); // stakey
                    let (_,bxx)  = entry.next()?.unwrap();
                    let (_,bslx) = entry.next()?.unwrap();
                    let (_,bsux) = entry.next()?.unwrap();
                    entry.finalize();
                    if bxx.len()  != numvar*8 ||
                       bslx.len() != numvar*8 || 
                       bsux.len() != numvar*8
                    {
                        return Err("Invalid solution format".to_string());
                    }
                    let mut xx  = vec![0.0; numvar];
                    let mut slx = vec![0.0; numvar];
                    let mut sux = vec![0.0; numvar];
                    unsafe {
                        xx.as_mut_slice().align_to_mut().1.copy_from_slice(bxx);
                        slx.as_mut_slice().align_to_mut().1.copy_from_slice(bslx);
                        sux.as_mut_slice().align_to_mut().1.copy_from_slice(bsux);
                    }
                    let dobjpart : f64 = izip!(slx.iter(),sux.iter(),self.var_elt.iter()).map(|(sl,su,e)| sl*e.lb-su*e.ub).sum(); 
                    (xx,Some((slx,sux,dobjpart)))
                },
                _ => return Err("Invalid solution format".to_string())
            }
        };


        // expect ".../con"
        let (xc,sc) = {
            let mut entry = bs.expect()?;
            let name = entry.name();
            let fmt = entry.fmt();
            if !name.starts_with(prefix) || !name.ends_with(b"/con") { return Err("Invalid solution format".to_string()); }

            match fmt {
                b"[B[D" => { 
                    // primal solution only
                    _ = entry.next()?; // stakey
                    let (_,bxx) = entry.next()?.unwrap();
                    if bxx.len() != numvar*8 {
                        return Err("Invalid solution format".to_string());
                    }
                    let mut xx = vec![0.0; numvar];
                    unsafe {
                        xx.as_mut_slice().align_to_mut::<u8>().1.copy_from_slice(bxx);
                    }
                    (xx,None)
                },
                b"[B[D[D[D" |
                b"[B[D[D[D[D" => {
                    // primal and linear dual solution
                    _ = entry.next()?; // stakey
                    let (_,bxx)  = entry.next()?.unwrap();
                    let (_,bslx) = entry.next()?.unwrap();
                    let (_,bsux) = entry.next()?.unwrap();
                    entry.finalize()?;
                    if bxx.len()  != numvar*8 ||
                       bslx.len() != numvar*8 || 
                       bsux.len() != numvar*8
                    {
                        return Err("Invalid solution format".to_string());
                    }
                    let mut xx  = vec![0.0; numvar];
                    let mut slx = vec![0.0; numvar];
                    let mut sux = vec![0.0; numvar];
                    unsafe {
                        xx.as_mut_slice().align_to_mut().1.copy_from_slice(bxx);
                        slx.as_mut_slice().align_to_mut().1.copy_from_slice(bslx);
                        sux.as_mut_slice().align_to_mut().1.copy_from_slice(bsux);
                    }
                    let dobjpart : f64 = izip!(slx.iter(),sux.iter(),self.con_elt.iter()).map(|(sl,su,e)| sl*e.lb - su*e.ub).sum();
                    (xx,Some((slx,sux,dobjpart)))
                },
                _ => return Err("Invalid solution format".to_string())
            }
        };
       
        sol.primal.obj = izip!(xx.permute_by(self.c_subj.as_slice()),self.c_cof.iter()).map(|(&x,&cj)| x*cj).sum();

        if let SolutionStatus::Undefined = psta {}
        else {
            for (v,x) in izip!(self.vars.iter(),sol.primal.var.iter_mut()) {
                *x = match v {
                    Item::Linear { index } => xx[*index],
                    Item::RangedUpper { index } => xx[*index],
                    Item::RangedLower { index } => xx[*index]
                };
            }
            for (v,x) in izip!(self.cons.iter(),sol.primal.con.iter_mut()) {
                *x = match v {
                    Item::Linear { index } => xc[*index],
                    Item::RangedUpper { index } => xc[*index],
                    Item::RangedLower { index } => xc[*index]
                };
            }
        }
        match (dsta,sx,sc) {
            (SolutionStatus::Undefined,_,_) => {},
            (_,Some((slx,sux,dobjpartx)),Some((slc,suc,dobjpartc))) => {
                sol.dual.obj = dobjpartx+dobjpartc;
                for (v,x) in izip!(self.vars.iter(),sol.dual.var.iter_mut()) {
                    *x = match v {
                        Item::Linear { index }      => slx[*index]-sux[*index],
                        Item::RangedUpper { index } => -sux[*index],
                        Item::RangedLower { index } => slx[*index]
                    };
                }
                for (v,x) in izip!(self.cons.iter(),sol.dual.con.iter_mut()) {
                    *x = match v {
                        Item::Linear { index }      => slc[*index]-suc[*index],
                        Item::RangedUpper { index } => -suc[*index],
                        Item::RangedLower { index } => suc[*index]
                    };
                }
            },
            _ => return Err("Invalid solution format".to_string()),
        }

        Ok(())
    }

    /// The b-solution is a solution in the same format as the btask.
    ///
    /// # Info section
    /// The info section is identical to the btask info section. 
    /// ```text
    /// INFO/MOSEKVER III
    /// INFO/name [B
    /// INFO/numvar I
    /// INFO/numcon I
    /// INFO/numcone I
    /// INFO/numbarvar I
    /// INFO/numdomain L
    /// INFO/numafe L
    /// INFO/numacc L
    /// INFO/numdjc L
    /// INFO/numsymmat L
    /// INFO/atruncatetol d 
    /// ```
    /// 
    /// # Solution 
    /// Solution section is the same as in btask. 
    ///
    /// Note that both primal and dual solution values are present, even for values that are
    /// logically unknown or undefined (for e.g. certificates or unknown status).
    ///
    /// ## Basic solution
    /// The entire section is optional. 
    /// ```text
    /// solution/basic/status [B[B       -- prosta, solsta
    /// solution/basic/var [B[D[D[D      {numvar > 0} -- stakey,level,slx,sux
    /// solution/basic/con [B[D[D[D[D    {numcon > 0} -- stakey,level,slc,suc,y
    /// solution/basic/acc [D            {numacc > 0} -- doty
    /// ```
    ///
    /// ## Interior solution
    /// The entire section is optional. 
    /// ```text
    /// solution/interior/status [B[B    -- prosta, solsta
    /// solution/interior/var [B[D[D[D   {numvar > 0} -- stakey,level,slx,sux,snx
    /// solution/interior/barvar [D[D    {numbarvar > 0} -- barx,bars
    /// solution/interior/con [B[D[D[D[D {numcon > 0} -- stakey,level,slc,suc,y
    /// solution/interior/acc [D         {numacc > 0} -- doty
    /// ```
    ///
    /// ## Integer solution
    /// The entire section is optional. 
    /// ```text
    /// solution/integer/status [B[B     -- prosta, solsta
    /// solution/integer/var [B[D[D[D    {numvar > 0} -- stakey,level
    /// solution/integer/barvar [D[D     {numbarvar > 0} -- barx,bars
    /// solution/integer/con [B[D[D[D[D  {numcon > 0} -- stakey,level
    /// ```
    ///
    /// # Information items
    /// ```text
    /// information/int    [I[B[i 
    /// information/double [I[B[d
    /// information/long   [I[B[l 
    /// ```
    fn read_bsolution<R>(&self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution,r : &mut R) -> std::io::Result<()> where R : Read 
    {
        let mut r = bio::Des::new(r)?;

        let mut got_sol_itr = false;
        let mut got_sol_itg = false;
        let mut got_sol_bas = false;
        let mut got_sol_inf = false;

        {
            let entry = r.expect(b"INFO/MOSEKVER").check_fmt(b"III")?;
            _ = entry.next_value::<u32>()?;
            _ = entry.next_value::<u32>()?;
            _ = entry.next_value::<u32>()?;
        }
        _ = r.expect(b"INFO/name")?.check_fmt(b"[B")?.read::<u8>()?;
        let numvar    = r.expect(b"INFO/numvar")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numcon    = r.expect(b"INFO/numcon")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numcone   = r.expect(b"INFO/numcone")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numbarvar = r.expect(b"INFO/numbarvar")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numdomain = r.expect(b"INFO/numdomain")?.check_fmt(b"I")?.next_value::<u64>()?;
        let numafe    = r.expect(b"INFO/numafe")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numacc    = r.expect(b"INFO/numacc")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numdjc    = r.expect(b"INFO/numdjc")?.check_fmt(b"I")?.next_value::<u32>()?;
        let numsymmat = r.expect(b"INFO/numsymmat")?.check_fmt(b"I")?.next_value::<u32>()?;

        Ok(())
    }

    fn read_bsolution_(&self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution, data : &[u8]) -> Result<(),String>
    {
        if ! data.starts_with(b"BASF") { return Err("Invalid solution format".to_string()) }

        let mut bs = bio::BAIO::new(&data[4..]);


        while let Some((name,fmt)) = bs.peek()? {
            if let Some(rest) = name.strip_prefix(b"solution/") {
                let which = name.strip_suffix(b"/status").ok_or_else(|| "Invalid solution format".to_string())?;
                match which {
                    b"basic"    => self.read_one_bsol(sol_bas, &mut bs)?,
                    b"interior" => self.read_one_bsol(sol_itr, &mut bs)?,
                    b"integer"  => self.read_one_bsol(sol_itg, &mut bs)?,
                    _ => return Err("Invalid solution format".to_string()),
                }
            }
        }

        while bs.next()?.is_some() { }

        Ok(())
    }


    /// JSON Task format writer.
    ///
    /// See https://docs.mosek.com/latest/capi/json-format.html
    fn write_jtask<S>(&self, strm : &mut S) -> std::io::Result<()> 
        where 
            S : std::io::Write 
    {
        use json::JSON;
        
        let mut doc = json::Dict::new();
        doc.append("$schema",JSON::String("http://mosek.com/json/schema#".to_string()));

        if let Some(name) = &self.name {
            doc.append("Task/name", name.clone());
        }

        doc.append(
            "Task/info",
            json::Dict::from(|taskinfo| {
                taskinfo.append("numvar",self.vars.len() as i64);
                taskinfo.append("numcon",self.con_elt.len() as i64);
            }));
            
        doc.append(
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

        JSON::Dict(doc).write(strm)
    }
}














impl ModelWithLogCallback for Backend {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str) {
        self.log_cb = Some(Box::new(func));
    }
}

fn ascii_from_bytes_lossy(data : &[u8]) -> String {
    let mut res = String::new();
    for &b in data.iter() {
        match b {
            b'\n' => res.push('\n'),
            b'\r' => res.push_str("\\r"),
            32..127 => res.push(b as char),
            _ => res.push('.')
        }
    }
    res
}

fn subseq_location_from<T>(pos : usize, src : &[T], seq : &[T]) -> Option<usize> where T : Eq {
    if pos+seq.len() > src.len() {
        return None;
    }

    for i in pos..src.len()-seq.len()+1 {
        if unsafe{ *src.get_unchecked(i) == *seq.get_unchecked(0) } {
            let mut found = true;
            for (j0,j1) in (i..i+seq.len()).enumerate() {
                if unsafe{ *src.get_unchecked(j1) != *seq.get_unchecked(j0) } {
                    found = false;
                    break;
                }
            }
            if found {
                return Some(i);
            }
        }
    }
    None
}
fn subseq_location<T>(src : &[T], seq : &[T]) -> Option<usize> where T : Eq {
    subseq_location_from(0,src, seq)
}


fn bounds_to_bbk(e : & Element) -> u8 {
    match (e.lb.is_finite(),e.ub.is_finite()) {
        (false,false) => b'f',
        (true,false) => b'l',
        (false,true) => b'u',
        (true,true) => if e.lb<e.ub||e.lb>e.ub { b'r' } else { b'f' },
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_optserver() {
        let addr = "solve.mosek.com:30080".to_string();
        let mut m = Model::new(Some("SuperModel"));
        m.set_parameter((), SolverAddress(addr));

        let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
        let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
        let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
        let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

        // Create variable 'x' of length 4
        let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

        // Create constraints
        let _ = m.constraint(None, x.index(1), less_than(10.0));
        let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
        let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
        let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

        // Set the objective function to (c^t * x)
        m.objective(Some("obj"), Sense::Maximize, x.dot(c));

        // Solve the problem
        m.write_problem("lo1-nosol.jtask");
        m.solve();

        // Get the solution values
        let (psta,dsta) = m.solution_status(SolutionType::Default);
        println!("Status = {:?}/{:?}",psta,dsta);
        let xx = m.primal_solution(SolutionType::Default,&x);
        println!("x = {:?}", xx);
    }
}
