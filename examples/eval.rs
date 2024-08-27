/// This library provides a C-callable API for the expression evaluation functionality.

extern crate mosekmodel;

use std::ptr::null;

use mosekmodel::expr::workstack::WorkStack;
use mosekmodel::expr::eval;


#[no_mangle]
pub extern "C" fn workstack_new(cap : usize) -> *mut WorkStack {
    Box::into_raw(Box::new(WorkStack::new(cap)))
}

#[no_mangle]
pub extern "C" fn workstack_delete(s : *mut WorkStack) {
    unsafe {
        _ = Box::from_raw(s);
    }
}

#[no_mangle]
pub extern "C" fn workstack_pop
(   s     : *mut WorkStack,
    nd    : *mut usize,
    nelm  : *mut usize,
    nnz   : *mut usize,
    aptr  : *mut *const usize,
    asubj : *mut *const usize,
    acof  : *mut *const f64,
    sparsity : *mut * const usize)
{
    unsafe {
        let (shape,ptr,sp,subj,cof) = (*s).pop_expr();
        (*nd)    = shape.len();
        (*nelm)  = ptr.len()-1; 
        (*nnz)   = *ptr.last().unwrap();
        (*aptr)  = ptr.as_ptr();
        (*asubj) = subj.as_ptr();
        (*acof)  = cof.as_ptr();
        if let Some(sp) = sp {
            (*sparsity) = sp.as_ptr();
        }
        else {
            (*sparsity) = null();
        }
    } 
}


pub extern "C" fn expression
(   nd : usize,
    nelm : usize, 
    nnz  : usize,
    shape : * const usize,
    aptr : * const usize,
    asubj : * const usize,
    acof : * const f64,
    sparsity : * const usize,
    rs : * mut WorkStack)
{
    unsafe {
        let shape = std::slice::from_raw_parts(shape,nd);
        let aptr  = std::slice::from_raw_parts(aptr,nelm+1);
        let asubj = std::slice::from_raw_parts(asubj,nnz);
        let acof  = std::slice::from_raw_parts(acof,nnz);
        let nnz   = *aptr.last().unwrap();
        let nelm  = aptr.len()-1;

        let (rptr,rsp,rsubj,rcof) = (*rs).alloc_expr(shape,nnz,nelm);
        rptr.copy_from_slice(aptr);
        rsubj.copy_from_slice(asubj);
        rcof.copy_from_slice(acof);
        if let Some(rsp) = rsp {
            rsp.clone_from_slice(std::slice::from_raw_parts(sparsity, nelm));
        }
    }
}

#[no_mangle]
pub fn scalar_expr_mul
(   datand       : usize,
    datashape    : * const usize,
    datannz      : usize,
    datasparsity : * const usize,
    data         : * const f64,
    rs           : * mut WorkStack, 
    ws           : * mut WorkStack, 
    xs           : * mut WorkStack) 
{
    unsafe {
        let shape = std::slice::from_raw_parts(datashape, datand);
        let totsize : usize = shape.iter().product();
        let sp = if totsize > datannz { Some(std::slice::from_raw_parts(datasparsity, datannz)) } else { None };

        eval::scalar_expr_mul(
            shape,
            sp,
            std::slice::from_raw_parts(data, datannz),
            &mut *rs,
            &mut *ws,
            &mut *xs);
    }
    
}

#[no_mangle]
pub extern "C" fn diag
( anti : i32, 
  index : i64, 
  rs : * mut WorkStack, 
  ws : * mut WorkStack, 
  xs : * mut WorkStack) 
{
    unsafe {
        eval::diag(
            anti != 0,
            index,
            &mut *rs,
            &mut *ws,
            &mut *xs);
    }
}

#[no_mangle]
pub extern "C"  fn permute_axes
( perm : * const usize,
  nd   : usize,
  rs   : * mut WorkStack,
  ws   : * mut WorkStack,
  xs   : * mut WorkStack) 
{
    unsafe {
        match nd {
            0 => { let p = [0usize; 0]; eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            1 => { let mut p = [0usize; 1]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            2 => { let mut p = [0usize; 2]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            3 => { let mut p = [0usize; 3]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            4 => { let mut p = [0usize; 4]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            5 => { let mut p = [0usize; 5]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            6 => { let mut p = [0usize; 6]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            7 => { let mut p = [0usize; 7]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            8 => { let mut p = [0usize; 8]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            9 => { let mut p = [0usize; 9]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            10 => { let mut p = [0usize; 10]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            11 => { let mut p = [0usize; 11]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            12 => { let mut p = [0usize; 12]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            13 => { let mut p = [0usize; 13]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            14 => { let mut p = [0usize; 14]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            15 => { let mut p = [0usize; 15]; p.copy_from_slice(std::slice::from_raw_parts(perm,nd)); eval::permute_axes(&p, &mut *rs, &mut *ws, &mut *xs); },
            _ => unimplemented!("permute axes for N > 15")
        }
    }
}


/// Add `n` expression residing on `ws`. Result pushed to `rs`.
#[no_mangle]
pub extern "C" fn add
( n  : usize,
  rs : * mut WorkStack,
  ws : * mut WorkStack,
  xs : * mut WorkStack)
{
    unsafe {
        eval::add(n,& mut *rs, & mut *ws, & mut *xs);
    }
}

/// Evaluates `lhs` * expr.
#[no_mangle]
pub extern "C" fn mul_left_dense
( mdata : * const f64,
  mnnz  : usize,
  mdimi : usize,
  mdimj : usize,
  rs    : * mut WorkStack,
  ws    : * mut WorkStack,
  xs    : * mut WorkStack)
{
    unsafe {
        eval::mul_left_dense(
            std::slice::from_raw_parts(mdata,mnnz),
            mdimi,
            mdimj,
            &mut *rs,
            &mut *ws,
            &mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn mul_right_dense
( mdata : * const f64,
  mnnz  : usize,
  mdimi : usize,
  mdimj : usize,
  rs    : * mut WorkStack,
  ws    : * mut WorkStack,
  xs    : * mut WorkStack) 
{
    unsafe {
        eval::mul_right_dense(
            std::slice::from_raw_parts(mdata,mnnz),
            mdimi,
            mdimj,
            &mut *rs,
            &mut *ws,
            &mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn mul_left_sparse
( mheight   : usize,
  mwidth    : usize,
  msparsity : * const usize,
  mdata     : * const f64,
  mnnz      : usize,
  rs        : * mut WorkStack,
  ws        : * mut WorkStack,
  xs        : * mut WorkStack) 
{
    unsafe {
        eval::mul_left_sparse(
            mheight,
            mwidth,
            std::slice::from_raw_parts(msparsity,mnnz),
            std::slice::from_raw_parts(mdata,mnnz),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

// expr x matrix
#[no_mangle]
pub extern "C" fn mul_right_sparse
( mheight   : usize,
  mwidth    : usize,
  msparsity : * const usize,
  mdata     : * const f64,
  mnnz      : usize,
  rs        : * mut WorkStack,
  ws        : * mut WorkStack,
  xs        : * mut WorkStack)
{
    unsafe {
        eval::mul_right_sparse(
            mheight,
            mwidth,
            std::slice::from_raw_parts(msparsity,mnnz),
            std::slice::from_raw_parts(mdata,mnnz),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn mul_elem
( datand       : usize,
  datashape    : * const usize,
  datannz      : usize,
  datasparsity : * const usize,
  data         : * const f64,
  rs           : * mut WorkStack,
  ws           : * mut WorkStack,
  xs           : * mut WorkStack) 
{
    unsafe {
        let shape = std::slice::from_raw_parts(datashape,datand);
        let totsize : usize = shape.iter().product();
        
        eval::mul_elem(
            shape,
            if totsize > datannz { Some(std::slice::from_raw_parts(datasparsity,datannz)) } else { None },
            std::slice::from_raw_parts(data ,datannz),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn dot_sparse
( sparsity : * const usize,
  data     : * const f64,
  mnnz     : usize,
  rs       : * mut WorkStack,
  ws       : * mut WorkStack,
  xs       : * mut WorkStack)
{
    unsafe {
        eval::dot_sparse(
            std::slice::from_raw_parts(sparsity,mnnz),
            std::slice::from_raw_parts(data,mnnz),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn dot_vec
( data : *const f64,
  nnz  : usize,
  rs   : * mut WorkStack,
  ws   : * mut WorkStack,
  xs   : * mut WorkStack) 
{
    unsafe {
        eval::dot_vec(
            std::slice::from_raw_parts(data,nnz),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn stack
( dim : usize,
  n   : usize, 
  rs  : * mut WorkStack,
  ws  : * mut WorkStack, 
  xs  : * mut WorkStack)
{
    unsafe {
        eval::stack(
            dim,
            n,
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn sum_last
( num : usize, 
  rs  : * mut WorkStack,
  ws  : * mut WorkStack, 
  xs  : * mut WorkStack) 
{
    unsafe {
        eval::sum_last(
            num,
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}


#[no_mangle]
pub extern "C" fn repeat
( dim : usize,
  num : usize,
  rs : * mut WorkStack,
  ws : * mut WorkStack,
  xs : * mut WorkStack) 
{
    unsafe {
        eval::repeat(
            dim,
            num,
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn into_symmetric
( dim : usize,
  rs : * mut WorkStack,
  ws : * mut WorkStack,
  xs : * mut WorkStack) 
{
    unsafe {
        eval::into_symmetric(
            dim,
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn triangular_part
( upper : i32,
  with_diag : i32,
  rs : * mut WorkStack, 
  ws : * mut WorkStack, 
  xs : * mut WorkStack) 
{ 
    unsafe {
        eval::triangular_part(
            upper != 0,
            with_diag != 0,
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn sum
( rs : * mut WorkStack, 
  ws : * mut WorkStack, 
  xs : * mut WorkStack) 
{
    unsafe {
        eval::sum(
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub fn slice
( nd    : usize,
  begin : *const usize, 
  end   : *const usize,
  rs    : * mut WorkStack, 
  ws    : * mut WorkStack,
  xs    : * mut WorkStack) 
{
    unsafe {
        eval::slice(
            std::slice::from_raw_parts(begin, nd),
            std::slice::from_raw_parts(end, nd),
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn inplace_reduce_shape
(   m : usize,
    rs : * mut WorkStack, 
    xs : * mut WorkStack) 
{
    unsafe {
        eval::inplace_reduce_shape(
            m,
            & mut *rs,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn inplace_reshape_one_row(m : usize, dim : usize, rs : * mut WorkStack, xs : * mut WorkStack) 
{
    unsafe { eval::inplace_reshape_one_row(m, dim, & mut *rs, &mut *xs) };
}

#[no_mangle]
pub extern "C" fn inplace_reshape(nd : usize, rshape : *const usize,rs : * mut WorkStack, xs : * mut WorkStack) 
{
    unsafe { eval::inplace_reshape(std::slice::from_raw_parts(rshape, nd), & mut *rs, & mut *xs) };
}

#[no_mangle]
pub extern "C" fn scatter(nd : usize, rshape : *const usize, nnz : usize, sparsity : *const usize, rs : * mut WorkStack, ws : * mut WorkStack, xs : * mut WorkStack)
{
    unsafe { 
        eval::scatter(
            std::slice::from_raw_parts(rshape, nd), 
            std::slice::from_raw_parts(sparsity, nnz), 
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

#[no_mangle]
pub extern "C" fn gather(nd : usize, rshape : * const usize, rs : * mut WorkStack, ws : * mut WorkStack, xs : * mut WorkStack)
{
    unsafe { eval::gather(std::slice::from_raw_parts(rshape, nd), & mut *rs, & mut *ws, & mut *xs); }
}

#[no_mangle]
pub extern "C" fn gather_to_vec(rs : * mut WorkStack, ws : * mut WorkStack, xs : * mut WorkStack) 
{
    unsafe { eval::gather_to_vec(&mut *rs, &mut *ws, &mut *xs); }
}

#[no_mangle]
pub extern "C" fn finalize(
    rs : * mut WorkStack, 
    ws : * mut WorkStack, 
    xs : * mut WorkStack)
{
    unsafe {
        eval::eval_finalize(
            & mut *rs,
            & mut *ws,
            & mut *xs);
    }
}

