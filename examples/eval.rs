/// This library provides a C-callable API for the expression evaluation functionality.

extern crate mosekmodel;

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


pub extern "C" fn expression
( nd : usize,
  nelm : usize, 
  nnz  : usize,
  shape : * const usize,
  aptr : * const usize,
  asubj : * const usize,
  acof : * const f64,
  sparsity : * const usize) 
{
    unsafe {
        let shape = std::slice::from_raw_parts(shape,nd);
        let aptr  = std::slice::from_raw_parts(aptr,nelm+1);
        let asubj = std::slice::from_raw_parts(asubj,nnz);
        let acof  = std::slice::from_raw_parts(acof,nnz);
        let sp    = if totsize > nelm { Some(std::slice::from_raw_parts(sparsity,nelm)) } else { None };

        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape,nnz,nelm);
        rptr.copy_from_slice(aptr);
        rsubj.copy_from_slice(asubj);
        rcof.copy_from_slice(acof);
        if let Some(sp) = sp {
            sp.clone_from_slice(std::slice::from_raw_parts(sparsity, nelm));
        }
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
        eval::permute_axes(
            std::slice::from_raw_parts(perm,nd),
            &mut *rs,
            &mut *ws,
            &mut *xs);
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

