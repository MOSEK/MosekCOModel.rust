* Implement constraint modification/update
* Implement constraint and variable deletion
* Expressions:
  * Pick: Given an expression, pick a subset of elements into a vector
    expression
  * [Done] Tril,Triu: For square matrix, select lower triangular or upper triangular,
    zeroing all other elements.
  * [Done] TrilIntoVec, TriuIntoVec: For square matrix, select lower triangular, upper
    triangular and put then into a diagonal in row-major order
  * [Done] Diag: For a matrix where all dimensions are the same, select diagonal
    elements. Special case for diag(A*B).
  * [Done] Permute (and transpose): Permute dimensions of expressions
* Parameters? 




# Overloading

Direct overloading is not possible. We have to use traits.

For example we would like to support combinations expression multiplication:

- `mul(f64, Expr<N>) -> Expr<N>` scalar multiplication with an expression of any size, left-right symmetric
- `mul(&[f64], Expr<0>) -> Expr<1>` scalar expression multiplied by a parameter vector, left-right symmetric
- `mul(DenseMatrix, Expr<0>) -> Expr<2>` matrix-scalar multiplication, left-right symmetric
- `mul(SparseMatrix, Expr<0>) -> Expr<2>` matrix-scalar multiplication, left-right symmetric
- `mul(NDArray<N>, Expr<0>) -> Expr<N>` nd array and scalar multiplication, left-right symmetric
- `mul(NDSparseArray<N>, Expr<0>) -> Expr<N>` nd array and scalar multiplication, left-right symmetric
- `mul(DenseMatrix, Expr<1>) -> Expr<1>` matrix-vector multiplication
- `mul(Expr<1>,DenseMatrix) -> Expr<1>` matrix-vector multiplication
- `mul(DenseMatrix, Expr<2>) -> Expr<2>` matrix-matrix multiplication
- `mul(Expr<2>,DenseMatrix) -> Expr<2>` matrix-matrix multiplication
- `mul(SparseMatrix, Expr<1>) -> Expr<1>` matrix-vector multiplication
- `mul(Expr<1>,SparseMatrix) -> Expr<1>` matrix-vector multiplication
- `mul(SparseMatrix, Expr<2>) -> Expr<2>` matrix-matrix multiplication
- `mul(Expr<2>,SparseMatrix) -> Expr<2>` matrix-matrix multiplication

We create two threads:
```
trait<const N : usize> LeftMultipliable { 
    type Result;
    fn mul(self, othre : Expr<N>) -> Self::Result;
} 

trait<const N : usize> RightMultipliable { 
    type Result;
    fn mul(self, othre : Expr<N>) -> Self::Result;
} 
```
A type that can be left-hand side of a multiplication `M * E` implements
`LeftMultipliable`, while a type that can be right-hand side in a multiplcation
`E * M` implements `RightMultipliable`.


To support right-hand side multiplication:
```
impl<const N : usize> Expr<N> {
    // In Expr<N> we forward the multiplication to the RHS object
    pub fn mul<RHS>(self, rhs : RHS) where Self : Sized, RHS : RightMultipliable<N> { rhs.mul_right(self) }
}

impl<const N : usize> RightMultipliable<N> for f64 {
    type Result = Expr<N>;
    fn mul_right(self, other : Expr<N>) -> Self::Result { ... }
}

impl RightMultipliable<0> for &[f64] {
    type Result = Expr<1>
    fn mul_right(self, other : Expr<0>) -> Self::Result { ... }
}

impl RightMultipliable<2> for &[f64] {
    type Result = Expr<1>
    fn mul_right(self, other : Expr<2>) -> Self::Result { ... }
}

impl RightMultipliable<1> for DenseMatrix {
    type Result = Expr<1>
    fn mul_right(self, other : Expr<1>) -> Self::Result { ... }
}

impl RightMultipliable<1> for DenseMatrix {
    type Result = Expr<2>
    fn mul_right(self, other : Expr<2>) -> Self::Result { ... }
}
```

To support left-hand side multiplication:
```
impl LeftMultipliable<const N : usize> for f64 {
    type Result = Expr<N>;
    fn mul(self, other : Expr<N>) -> Self::Result { ... }
}

impl LeftMultipliable<0> for &[f64] {
    type Result = Expr<1>;
    fn mul(self, other : Expr<0>) -> Self::Result { ... }
}

impl LeftMultipliable<2> for &[f64] {
    type Result = Expr<1>;
    fn mul(self, other : Expr<2>) -> Self::Result { ... }
}

impl LeftMultipliable<0> for DenseMatrix {
    type Result = Expr<2>;
    fn mul(self, other : Expr<0>) -> Self::Result { ... }
}

impl LeftMultipliable<1> for DenseMatrix {
    type Result = Expr<1>;
    fn mul(self, other : Expr<1>) -> Self::Result { ... }
}

impl LeftMultipliable<2> for DenseMatrix {
    type Result = Expr<2>;
    fn mul(self, other : Expr<2>) -> Self::Result { ... }
}
```



# Features

- Update and modify constraints or parts of constraints






# Generative expressions


