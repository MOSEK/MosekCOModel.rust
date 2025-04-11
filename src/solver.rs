use std::ops::ControlFlow;
use std::path::Path;

use model::Disjunction;

use crate::*;
use crate::domain::*;

// Defines the public interface for a Model
trait BaseModel {
    fn new(name : Option<&str>) -> Self;
    fn write_problem<P>(&self, filename : P) where P : AsRef<Path>;
    fn try_variable<I,D>(& mut self, name : Option<&str>, dom : I) -> Result<D::Result,String>
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait;
    fn variable<I,D>(& mut self, name : Option<&str>, dom : I) -> D::Result
        where 
            I : IntoDomain<Result = D>,
            D : VarDomainTrait
    {
        self.try_variable(name,dom).unwrap()
    }
    fn try_ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> Result<(Variable<N>,Variable<N>),String> 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>;
    fn ranged_variable<const N : usize,D>(&mut self, name : Option<&str>, dom : D) -> (Variable<N>,Variable<N>) 
        where 
            D : IntoLinearRange<Result = LinearRangeDomain<N>>
    {
        self.try_ranged_variable(name,dom).unwrap()
    }
    
    fn try_ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> Result<(Constraint<N>,Constraint<N>),String> 
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>;
    fn ranged_constraint<const N : usize,E,D>(&mut self, name : Option<&str>, expr : E, dom : D) -> (Constraint<N>,Constraint<N>)
        where E : IntoExpr<N>,
              E::Result : ExprTrait<N>,
              D : IntoShapedLinearRange<N>
    {
        self.try_ranged_constraint(name,expr,dom).unwrap()
    }

    fn try_constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Result<Constraint<N>,String>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>;
    
    fn constraint<const N : usize,E,D>(& mut self, name : Option<&str>, expr :  E, dom : D) -> Constraint<N>
        where
            E : IntoExpr<N>, 
            <E as IntoExpr<N>>::Result : ExprTrait<N>,
            D : IntoShapedDomain<N>,
            D::Result : ConstraintDomain<N>
    {
        self.try_constraint(name,expr,dom).unwrap()
    }

    fn try_disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Result<Disjunction,String> where D : disjunction::DisjunctionTrait;
    fn disjunction<D>(& mut self, name : Option<&str>, terms : D) -> Disjunction where D : disjunction::DisjunctionTrait {
        self.try_disjunction(name, terms).unwrap()
    }

    fn update<const N : usize, E>(&mut self, item : &Constraint<N>, e : E) where E : expr::IntoExpr<N>;

    fn objective<E : expr::IntoExpr<0>>(& mut self,
                                        name  : Option<&str>,
                                        sense : Sense,
                                        expr  : E);
    fn set_parameter<T>(& mut self, parname : &str, parval : T) 
        where T : SolverParameterValue;

    fn set_double_parameter(&mut self, parname : &str, parval : f64);
    fn try_set_double_parameter(&mut self, parname : &str, parval : f64) -> Result<(),String>;
    fn set_int_parameter(&mut self, parname : &str, parval : i32);
    fn try_set_int_parameter(&mut self, parname : &str, parval : i32) -> Result<(),String>;
    fn set_str_parameter(&mut self, parname : &str, parval : &str);
    fn try_set_str_parameter(&mut self, parname : &str, parval : &str) -> Result<(),String>;
    fn put_optserver(&mut self, hostname : &str, access_token : Option<&str>);
    fn clear_optserver(&mut self);
    fn solve(& mut self);
    fn solution_status(&self, solid : SolutionType) -> (SolutionStatus,SolutionStatus);
    fn primal_objective(&self, solid : SolutionType) -> Option<f64>;
    fn primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String>;
    fn sparse_primal_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<(Vec<f64>,Vec<[usize; N]>),String>; 
    fn dual_solution<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String>;
    fn primal_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String>;
    fn dual_solution_into<const N : usize, I:ModelItem<N>>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String>;
    fn evaluate_primal<const N : usize, E>(& mut self, solid : SolutionType, expr : E) -> Result<NDArray<N>,String> where E : IntoExpr<N>;
}

trait ModelWithCallbacks {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str);
    fn set_solution_callback<F>(&mut self, func : F) where F : 'static+FnMut(&Model); 
    fn set_callback<F>(&mut self, func : F) where F : 'static+FnMut() -> ControlFlow<(),()>;
}

trait Solver {
    fn try_add_vector_constraint<const N : usize,E,D>(&mut self, expr : E, domain : D) -> Result<Vec<usize>,String> where E : IntoExpr<N>, D : IntoShapedDomain<N,Result = ConicDomain<N>>;
    fn try_add_linear_constraint<const N : usize,E,D>(&mut self, expr : E, domain : D) -> Result<Vec<usize>,String> where E : IntoExpr<N>, D : IntoShapedDomain<N,Result = LinearDomain<N>>;
    fn try_add_ranged_constraint<const N : usize,E,D>(&mut self, expr : E, domain : D) -> Result<Vec<usize>,String> where E : IntoExpr<N>, D : IntoShapedDomain<N,Result = LinearRangeDomain<N>>;
    fn try_add_psd_constraint   <const N : usize,E,D>(&mut self, expr : E, domain : D) -> Result<Vec<usize>,String> where E : IntoExpr<N>, D : IntoShapedDomain<N,Result = PSDDomain<N>>;

    fn try_add_vector_variable<const N : usize,D>(&mut self, domain : D) -> Result<Vec<usize>,String> where D : IntoDomain<Result = ConicDomain<N>>;
    fn try_add_linear_variable<const N : usize,D>(&mut self, domain : D) -> Result<Vec<usize>,String> where D : IntoDomain<Result = LinearDomain<N>>;
    fn try_add_ranged_variable<const N : usize,D>(&mut self, domain : D) -> Result<Vec<usize>,String> where D : IntoDomain<Result = LinearRangeDomain<N>>;
    fn try_add_psd_variable   <const N : usize,D>(&mut self, domain : D) -> Result<Vec<usize>,String> where D : IntoDomain<Result = PSDDomain<N>>;

    fn set_int_parameter(&mut self, parname : &str, parval : i32);
    fn try_set_int_parameter(&mut self, parname : &str, parval : i32) -> Result<(),String>;
    fn set_str_parameter(&mut self, parname : &str, parval : &str);
    fn try_set_str_parameter(&mut self, parname : &str, parval : &str) -> Result<(),String>;

    fn solve(& mut self);
}




//trait TaskTrait {
//    fn append_acc_seq(&mut self,...)
//    fn append_accs_seq(&mut self,...)
//    fn append_afes(&mut self,...)
//    fn append_barvars(&mut self,...)
//    fn append_cons(&mut self,...)
//    fn append_djcs(&mut self,...)
//
//    fn append_dual_exp_cone_domain(&mut self,...)
//    fn append_dual_geo_mean_cone_domain(&mut self,...)
//    fn append_dual_power_cone_domain(&mut self,...)
//    fn append_primal_exp_cone_domain(&mut self,...)
//    fn append_primal_geo_mean_cone_domain(&mut self,...)
//    fn append_primal_power_cone_domain(&mut self,...)
//    fn append_quadratic_cone_domain(&mut self,...)
//    fn append_r_domain(&mut self,...)
//    fn append_rminus_domain(&mut self,...)
//    fn append_rplus_domain(&mut self,...)
//    fn append_r_quadratic_cone_domain(&mut self,...)
//    fn append_rzero_domain(&mut self,...)
//    fn append_svec_psd_cone_domain(&mut self,...)
//
//    fn append_sparse_sym_mat(&mut self,...)
//    fn append_vars(&mut self,...)
//    fn empty_afe_barf_row_list(& self,...)
//    fn evaluate_accs(& self,...)
//    fn get_acc_dot_y_s(& self,...)
//    fn get_acc_n(& self,...)
//    fn get_acc_n_tot(& self,...)
//    fn get_bars_slice(& self,...)
//    fn get_barx_slice(& self,...)
//    fn get_dim_barvar_j(& self,...)
//    fn get_dual_obj(& self,...)
//    fn get_len_barvar_j(& self,...)
//    fn get_num_acc(& self,...)
//    fn get_num_afe(& self,...)
//    fn get_num_barvar(& self,...)
//    fn get_num_con(& self,...)
//    fn get_num_djc(& self,...)
//    fn get_num_var(& self,...)
//    fn get_primal_obj(& self,...)
//    fn get_slc(& self,...)
//    fn get_slx(& self,...)
//    fn get_sol_sta(& self,...)
//    fn get_suc(& self,...)
//    fn get_sux(& self,...)
//    fn get_xc(& self,...)
//    fn get_xx(& self,...)
//    fn get_y(& self,...)
//    fn optimize(&mut self,...)
//    fn optimize_rmt(&mut self,...)
//
//    fn put_barvar_name(&mut self,...)
//    fn put_acc_name(&mut self,...)
//    fn put_con_name(&mut self,...)
//    fn put_djc_name(&mut self,...)
//    fn put_obj_name(&mut self,...)
//    fn put_var_name(&mut self,...)
//
//    fn put_afe_barf_entry(&mut self,...)
//    fn put_afe_f_entry_list(&mut self,...)
//    fn put_afe_f_row_list(&mut self,...)
//    fn put_afe_g_list(&mut self,...)
//    fn put_a_row_list(&mut self,...)
//    fn put_a_row_slice(&mut self,...)
//    fn put_bara_ij(&mut self,...)
//    fn put_barc_block_triplet(&mut self,...)
//    fn put_cfix(&mut self,...)
//    fn put_c_list(&mut self,...)
//    fn put_codecallback(&mut self,...)
//    fn put_con_bound_list(&mut self,...)
//    fn put_con_bound_slice(&mut self,...)
//    fn put_djc(&mut self,...)
//    fn put_int_param(&mut self,...)
//    fn put_intsolcallback(&mut self,...)
//    fn put_na_dou_param(&mut self,...)
//    fn put_na_int_param(&mut self,...)
//    fn put_na_str_param(&mut self,...)
//    fn put_obj_sense(&mut self,...)
//    fn put_stream_callback(&mut self,...)
//    fn put_var_bound_slice(&mut self,...)
//    fn put_var_bound_slice_const(&mut self,...)
//    fn put_var_type_list(&mut self,...)
//    fn solution_def(& self,...)
//    fn write_data(& self,...)
//}

