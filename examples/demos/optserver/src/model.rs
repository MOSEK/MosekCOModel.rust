use std::path::Path;
use mosekcomodel::*;
use mosekcomodel::domain::LinearRangeDomain;
use itertools::izip;

use mosekcomodel::expr::workstack::WorkStack;
use mosekcomodel::utils::iter::ChunksByIterExt;

enum ConeType {
    Unbounded,
    Fixed,
    Nonnegative,
    Nonpositive,
    QuadraticCone,
    RotatedQuadraticCone,
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

//
//enum LinearBoundType {
//    Unbounded,
//    Fixed,
//    Nonnegative,
//    Nonpositive
//}

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

    rs : WorkStack,
    ws : WorkStack,
    xs : WorkStack,
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
}

impl ModelAPI for ModelOptserver {
    fn try_constraint<const N : usize,E,I,D>(& mut self, name : Option<&str>, expr :  E, dom : I) -> Result<D::Result,String>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            I : IntoShapedDomain<N,Result=D>,
            D : ConstraintDomain<N,Self>,
            Self : Sized
    {
        expr.into_expr().eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs).map_err(|e| format!("{:?}",e))?;
        let (eshape,_,_,_,_) = self.rs.peek_expr();
        if eshape.len() != N { panic!("Inconsistent shape for evaluated expression") }
        let mut shape = [0usize; N]; shape.copy_from_slice(eshape);

        dom.try_into_domain(shape)?.add_constraint(self,name)
    }
    
    fn update<const N : usize, E>(&mut self, item : &Constraint<N>, e : E) -> Result<(),String>
        where 
            E    : expr::IntoExpr<N>
    {
        unimplemented!("Not implemented");
    }
    
    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path> {
        unimplemented!("Not implemented");
    }
    fn solve(& mut self) -> Result<(),String> {
        unimplemented!("Not implemented");
    }
    fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus) {
        unimplemented!("Not implemented");
    }

    fn primal_objective_value(&self, solid : SolutionType) -> Option<f64> {
        unimplemented!("Not implemented");
    }
    fn dual_objective_value(&self, solid : SolutionType) -> Option<f64> {
        unimplemented!("Not implemented");
    }
    fn primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> where Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
    fn sparse_primal_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String> where Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
    fn dual_solution<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> where Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
    fn primal_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> where Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
    fn dual_solution_into<const N : usize, I:ModelItem<N,Self>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> where Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
    fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String> where E : IntoExpr<N>, Self : Sized+BaseModelTrait {
        unimplemented!("Not implemented");
    }
}



impl BaseModelTrait for ModelOptserver {
    fn try_free_variable<const N : usize>
        (&mut self,
         name  : Option<&str>,
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

    fn try_linear_variable<const N : usize,R>
        (&mut self, 
         name : Option<&str>,
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
    
    fn try_ranged_variable<const N : usize,R>(&mut self, name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
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

    fn try_linear_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearDomain<N>) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String>
        where 
            Self : Sized 
    {
        let (_eshape,ptr,_,subj,cof) = self.rs.pop_expr();
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

    fn try_ranged_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (_eshape,ptr,_,subj,cof) = self.rs.pop_expr();
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

    fn try_update(& mut self, idxs : &[usize]) -> Result<(),String>
    {
        let (shape,ptr,sp,subj,cof) = self.rs.pop_expr();
        if shape.iter().product::<usize>() != idxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = idxs.iter().max() {
            if i >= self.con_rhs.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

        if let Some(sp) = sp {
            let mut it = izip!(sp.iter(),subj.chunks_ptr(ptr),cof.chunks_ptr(ptr)).peekable();
            for &i in idxs.iter() {
                if let Some((_,subj,cof)) = it.peek().and_then(|v| if *(v.0) == i { Some(v) } else { None }) {
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
                else {
                    self.a_ptr[i] = [0,0];
                }
            }

        }
        else {
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

    fn primal_var_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        unimplemented!("Not implemented");
    }
    fn dual_var_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        unimplemented!("Not implemented");
    }
    fn primal_con_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        unimplemented!("Not implemented");
    }
    fn dual_con_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        unimplemented!("Not implemented");
    }

}

