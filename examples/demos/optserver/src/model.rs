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
mod json {
    pub enum Item {
        String(String),
        Int(i64),
        Float(f64),
        Null,
        NaN,
        List(Vec<Item>),
        Dict(Vec<(String,Item)>)
    }
    
    impl Item {
        pub fn get_string(&self) -> Option<&String> { if let Item::String(s) = self { Some(s) } else { None } }
        pub fn get_int(&self) -> Option<i64> { if let Item::Int(i) = self { Some(*i) } else { None }}
        pub fn get_float(&self) -> Option<f64> { if let Item::Float(f) = self{ Some(*f) } else { None }}
        pub fn get_list(&self) -> Option<&[Item]> { if let Item::List(l) = self { Some(l.as_slice()) } else { None }}
        pub fn get_dict(&self) -> Option<&[(String,Item)]> { if let Item::Dict(d) = self { Some(d.as_slice()) } else { None }}

        pub fn to_float(&self) -> Option<f64> {
            self.get_float().or_else(|| self.get_int().map(|v|v as f64))
        }
        pub fn get_float_list(&self) -> Option<Vec<f64>> { 
            if let Item::List(l) = self {
                if l.iter().all(|item| match item { Item::Int(_)|Item::Float(_) => true, _ => false } ) {
                    Some(l.iter().map(|v| match v { Item::Int(i) => *i as f64, Item::Float(f) => *f, _ => 0.0}).collect())
                }
                else {
                    None
                }
            }
            else {
                None
            }
        }
        pub fn get_int_list(&self) -> Option<Vec<i64>> { 
            if let Item::List(l) = self {
                if l.iter().all(|item| match item { Item::Int(_) => true, _ => false } ) {
                    Some(l.iter().map(|v| match v { Item::Int(i) => *i, _ => 0}).collect())
                }
                else {
                    None
                }
            }
            else {
                None
            }
        }
    }



    pub fn skip_ws<I>(it : &mut std::iter::Peekable<I>) 
        where I : Iterator<Item = char> 
    {
        while let Some(' ') = it.peek() { it.next(); } 
    }

    pub fn expect<I>(it : &mut std::iter::Peekable<I>, c : char) -> Result<(),String> 
        where I : Iterator<Item = char> 
    {
        it.next().ok_or_else(|| format!("Expected '{}'",c))
            .and_then(|v| if v == c { Ok(()) } else { Err(format!("Expected '{}'",c)) } )
    }

    pub fn parse_string_into<I>(it : &mut std::iter::Peekable<I>,res : & mut String) -> Result<(),String>
        where I : Iterator<Item = char> 
    {
        skip_ws(it);
        expect(it,'"')?;

        while let Some(c) = it.next() {
            if c == '"' {
                return Ok(())
            }
            else if c == '\\' {
                if let Some(c) = it.next() {
                    match c {
                        'n' => res.push('\n'),
                        't' => res.push('\t'),
                        'r' => res.push('\r'),
                        '0' => res.push('\0'),
                        'x' => {
                            if let (Some(hi),Some(lo)) = (it.next(),it.next()) {
                                let hi = match hi {
                                    '0'..='9' => hi as u8 - '0' as u8,
                                    'a'..='f' => hi as u8 - 'a' as u8 + 10,
                                    'A'..='F' => hi as u8 - 'A' as u8 + 10,
                                    _ => return Err("String entry syntax error".to_string())
                                };
                                let lo = match lo {
                                    '0'..='9' => hi as u8 - '0' as u8,
                                    'a'..='f' => hi as u8 - 'a' as u8 + 10,
                                    'A'..='F' => hi as u8 - 'A' as u8 + 10,
                                    _ => return Err("String entry syntax error".to_string())
                                };
                                res.push(((hi << 4) + lo) as char);
                            }
                        },
                        _ => return Err("Premature end of file".to_string())
                    }
                }
            }
            else {
                res.push(c);
            }
        }
        Err("Premature end of file".to_string())
    }
    pub fn parse_string<I>(it : &mut std::iter::Peekable<I>) -> Result<String,String>
        where I : Iterator<Item = char> 
    {
        let mut res = String::new();
        parse_string_into(it,& mut res)?;
        Ok(res)

    }

    pub fn parse_key_value<I>(it : & mut std::iter::Peekable<I>) -> Result<(String,Item),String>
        where I : Iterator<Item = char> 
    {
        skip_ws(it);
        let key = parse_string(it)?;
        skip_ws(it);
        if let Some(':') = it.peek() {
            skip_ws(it);
            it.next();
            let value = parse_item(it)?;
            Ok((key,value))
        }
        else {
            Err("Expected a number".to_string())
        }

    }


    pub fn parse_number_with<I>(it : & mut std::iter::Peekable<I>, res : &mut String) -> Result<Item,String>
        where I : Iterator<Item = char> 
    {
        if let Some(c) = it.next() {
            // [0-9]+ ( '.' [0-9]+ )? ( [eE] [+-]? [0-9]+)?
            match c {
                '0'..='9' => res.push(c),
                'i' => 
                    if let (Some('n'),Some('f')) = (it.next(),it.next()) {
                        if res.chars().nth(0).map(|c| c == '-').unwrap_or(false) {
                            return Ok(Item::Float(f64::NEG_INFINITY));
                        }
                        else {
                            return Ok(Item::Float(f64::INFINITY));
                        }
                    },
                _ => return Err("Expected a number".to_string())
            }

            while let Some(&c) = it.peek() {
                match c {
                    '0'..='9' => res.push(c),
                    _ => break
                }
            }
            let mut is_float = false;
            if let Some('.') = it.peek() {
                is_float = true;
                res.push('.'); it.next();
                while let Some(&c) = it.peek() {
                    match c {
                        '0'..='9' => res.push(c),
                        _ => break
                    }
                }
            }

            if let Some(c) = it.peek() {
                if *c == 'e' || *c == 'E' {
                    is_float = true;
                    res.push(*c);
                    it.next();

                    if let Some(c) = it.peek() {
                        if *c == '+' || *c == '-' {
                            res.push(*c);
                            it.next();
                        }
                        if let Some(c) = it.peek() {
                            match *c {
                                '0'..='9' => res.push(*c),
                                _ => return Err("Expected a number".to_string())
                            }
                            it.next();

                            while let Some(c) = it.peek() {
                                match *c {
                                    '0'..='9' => res.push(*c),
                                    _ => break
                                }
                                it.next();
                            }
                        }
                    }
                }
            }

            if is_float {
                Ok(Item::Float(res.parse().map_err(|_| format!("Invalid float format: '{}'",res))?))
            }
            else {
                Ok(Item::Int(res.parse().map_err(|_| format!("Invalid int format: '{}'",res))?))
            }
         }
        else {
            Err("Expected a number".to_string())
        }
    }

    pub fn parse_item<I>(it : & mut std::iter::Peekable<I>) -> Result<Item,String>
        where I : Iterator<Item = char>
    {
        skip_ws(it);
        
        if let Some(&c) = it.peek() {
            match c {
                '"' => {
                    it.next();
                    let mut res = String::new();
                    parse_string_into(it, &mut res)?;
                    Ok(Item::String(res))
                }
                '-'|'+' => {
                    it.next();
                    let mut res = String::new();
                    skip_ws(it);
                    res.push(c);
                    parse_number_with(it,&mut res)
                },
                '0'..='9'|'i' => {
                    let mut res = String::new();
                    parse_number_with(it,&mut res)
                },
                'n' => {
                    it.next();
                    if let (Some('u'),Some('l'),Some('l')) = (it.next(),it.next(),it.next()) {
                        Ok(Item::Null)
                    }
                    else {
                        Err("Syntax error in file".to_string())
                    }
                },
                '[' => {
                    it.next();
                    let mut res = Vec::new();
                    skip_ws(it);
                    if let Some(']') = it.peek() {
                        Ok(Item::List(res))
                    }
                    else {
                        res.push(parse_item(it)?);
                        skip_ws(it);
                        while let Some(',') = it.peek() {
                            it.next();
                            res.push(parse_item(it)?);
                            skip_ws(it);
                        }
                        if let Some(']') = it.peek() {
                            Ok(Item::List(res))
                        }
                        else {
                            Err("Syntax error in file".to_string())
                        }
                    }
                },
                '{' => {
                    let mut res = Vec::new();
                    skip_ws(it);
                    if let Some('}') = it.peek() {
                        Ok(Item::Dict(res))
                    } 
                    else {
                        res.push(parse_key_value(it)?);
                        skip_ws(it);
                        while let Some(',') = it.peek() {
                            it.next();
                            res.push(parse_key_value(it)?);
                            skip_ws(it);
                        }
                        if let Some(']') = it.peek() {
                            Ok(Item::Dict(res))
                        }
                        else {
                            Err("Syntax error in file".to_string())
                        }

                        
                    }
                    
                },
                _ => Err("Syntax error in file".to_string())
            }
        }
        else {
            Err("Syntax error in file".to_string())
        }
    }


    pub fn parse(data : &str) -> Result<Item,String> {
        let mut it = data.chars().peekable();
        parse_item(&mut it)
    }
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
    fn format_to(&self, dst : &mut String) {
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
    }

    fn write_jtask(&self,f : &mut std::fs::File) -> Result<usize,String> {
        let mut data = String::new();
        self.format_to(&mut data);
        f.write(data.as_ref()).map_err(|err| err.to_string())
    }

    fn format(&self) -> String {
        let mut data = String::new();
        self.format_to(&mut data);
        data
    }    
}


fn extract_solution(item : &json::Item, vars : &[Item], cons : &[Item], sol  : & mut Solution) -> Result<(),String> {
    let mut xx = None;
    let mut slx = None;
    let mut sux = None;
    let mut xc = None;
    let mut slc = None;
    let mut suc = None;
    for (k,v) in item.get_dict().ok_or_else(|| "Format error".to_string())? {
        match k.as_str() {
            "solsta" => { 
                ((*sol).primal.status,(*sol).dual.status) = match v.get_string().ok_or_else(|| "Format error".to_string())?.as_str() {
                    "unknown" => (SolutionStatus::Unknown,SolutionStatus::Unknown),
                    "optimal" => (SolutionStatus::Optimal,SolutionStatus::Optimal),
                    "integer_optimal" => (SolutionStatus::Optimal,SolutionStatus::Undefined),
                    "prim_feas" => (SolutionStatus::Feasible,SolutionStatus::Unknown),
                    "dual_feas" => (SolutionStatus::Unknown,SolutionStatus::Feasible),
                    "prim_and_dual_feas" => (SolutionStatus::Feasible,SolutionStatus::Feasible),
                    "prim_infeas_cer" => (SolutionStatus::Undefined,SolutionStatus::CertInfeas),
                    "dual_infeas_cer" => (SolutionStatus::CertInfeas,SolutionStatus::Undefined),
                    "prim_illposed_cer" => (SolutionStatus::Undefined,SolutionStatus::CertIllposed),
                    "dual_illposed_cer" => (SolutionStatus::CertIllposed,SolutionStatus::Undefined),
                    _ => (SolutionStatus::Unknown,SolutionStatus::Unknown)
                };
            },
            "xx"|"slx"|"sux"|"xc"|"slc"|"suc" => {
                let lst = v.get_list().ok_or_else(|| "Format error".to_string())?;
                let mut res = Vec::with_capacity(lst.len());
                for v in lst.iter() {
                    match v {
                        json::Item::Float(f) => res.push(*f),
                        json::Item::Int(i)   => res.push(*i as f64),
                        _ => return Err("Invalid format".to_string())
                    }
                }
                match k.as_str() {
                    "xx"  => xx  = Some(res),
                    "slx" => slx = Some(res),
                    "sux" => sux = Some(res),
                    "xc"  => xc  = Some(res),
                    "slc" => slc = Some(res),
                    "suc" => suc = Some(res),
                    _ => {}
                }
            },
            _ => {}
        }
    }

    if let Some(xx) = xx {
        sol.primal.var.resize(vars.len(),0.0);
        for (v,dst) in vars.iter().zip(sol.primal.var.iter_mut()) {
            *dst = xx[v.index()];
        }
    }
    if let (Some(slx),Some(sux)) = (slx,sux) {
        sol.dual.var.resize(vars.len(),0.0);
        for (v,dst) in vars.iter().zip(sol.dual.var.iter_mut()) {
            match *v {
                Item::Linear{index} => *dst = slx[index]-sux[index],
                Item::RangedLower{index} => *dst = slx[index],
                Item::RangedUpper{index} => *dst = -sux[index]
            }
        }
    }
    if let Some(xc) = xc {
        sol.primal.con.resize(cons.len(), 0.0);
        for (v,dst) in cons.iter().zip(sol.primal.con.iter_mut()) {
            *dst = xc[v.index()];
        }
    }

    if let (Some(slc),Some(suc)) = (slc,suc) {
        sol.dual.con.resize(cons.len(),0.0);
        for (v,dst) in cons.iter().zip(sol.dual.con.iter_mut()) {
            match *v {
                Item::Linear{index} => *dst = slc[index]-suc[index],
                Item::RangedLower{index} => *dst = slc[index],
                Item::RangedUpper{index} => *dst = -suc[index]
            }
        }
    }

    Ok(())
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

    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution) -> Result<(),String>
    {
        let data = self.format();
        let recv_body : String = ureq::post(format!("{}/api/v1/submit+solve",self.hostname).as_str())
            .header("Content-Type","application/x-mosek-jtask")
            .header("Accept","application/x-mosek-jtask")
            .send(data).map_err(|err| err.to_string())?
            .body_mut()
            .read_to_string().map_err(|err| err.to_string())?;

        let data = json::parse(recv_body.as_str())?;

        let tld = data.get_dict().ok_or_else(|| "Format error".to_string())?;
        for (k,v) in tld.iter() {
            match k.as_str() {
                "$schema" => {},
                "Task/solutions" => {
                    let secs = v.get_dict().ok_or_else(|| "Format error".to_string())?;
                
                    for (whichsol,v) in secs.iter() {
                        match whichsol.as_str() {
                            "interior" => extract_solution(v, self.vars.as_slice(), self.cons.as_slice(), sol_itr)?,
                            "integer"  => extract_solution(v, self.vars.as_slice(), self.cons.as_slice(), sol_itg)?,
                            "basic"    => extract_solution(v, self.vars.as_slice(), self.cons.as_slice(), sol_bas)?,
                            _ => {},
                        };
                    }

                },
                "Task/information" => {
                    if let Some(d) = v.get_dict() {
                        for (k,v) in d.iter() {
                            if k == "double" {
                                if let Some(d) = v.get_dict() {
                                    for (k,v) in d.iter() {
                                        match k.as_str() {
                                            "sol_itr_primal_obj" => sol_itr.primal.obj = v.to_float().unwrap_or(0.0),
                                            "sol_itr_dual_obj"   => sol_itr.dual.obj   = v.to_float().unwrap_or(0.0),
                                            "sol_bas_primal_obj" => sol_bas.primal.obj = v.to_float().unwrap_or(0.0),
                                            "sol_bas_dual_obj"   => sol_bas.dual.obj   = v.to_float().unwrap_or(0.0),
                                            "sol_itg_primal_obj" => sol_itg.primal.obj = v.to_float().unwrap_or(0.0),
                                            _ => {},
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                _ => {}
            }
        }

        Ok(())
    }

    fn objective(&mut self, _name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.sense_max = match sense { Sense::Maximize => true, Sense::Minimize => false };
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

struct OptserverHost(String);
impl SolverParameterValue<ModelOptserver> for OptserverHost {
    type Key = &'static str;
    fn set(self,_parname : Self::Key, model : & mut ModelOptserver) -> Result<(),String> {
        model.hostname = self.0;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use mosekcomodel::*;
    use super::*;

    type Model = ModelAPI<ModelOptserver>;
    fn lo1() -> (Model,Variable<1>) {
        let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
        let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
        let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
        let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

        // Create a model with the name 'lo1'
        let mut m = Model::new(Some("lo1"));
        // Create variable 'x' of length 4
        let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

        // Create constraints
        let _ = m.constraint(None, x.index(1), less_than(10.0));
        let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
        let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
        let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

        // Set the objective function to (c^t * x)
        m.objective(Some("obj"), Sense::Maximize, x.dot(c));

        (m,x)
    }
    #[test]
    fn test_write() {
        let (m,_) = lo1();
        m.write_problem("lo1-nosol.jtask");
    }
    #[test]
    fn test_solve() {
        if let Ok(host) = std::env::var("OPTSERVER_HOST") {
            let (mut m,x) = lo1();
            m.set_parameter("optserver", OptserverHost(host));
            m.solve();

            // Get the solution values
            let (psta,dsta) = m.solution_status(SolutionType::Default);
            println!("Status = {:?}/{:?}",psta,dsta);
            let xx = m.primal_solution(SolutionType::Default,&x);
            println!("x = {:?}", xx);
        }
    }
}


