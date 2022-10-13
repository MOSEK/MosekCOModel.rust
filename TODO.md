* Implement constraint modification/update
* Implement constraint and variable deletion
* Expressions:
  * Pick: Given an expression, pick a subset of elements into a vector
    expression
  * Tril,Triu: For square matrix, select lower triangular or upper triangular,
    zeroing all other elements.
  * TrilIntoVec, TriuIntoVec: For square matrix, select lower triangular, upper
    triangular and put then into a diagonal in row-major order
  * Diag: For a matrix where all dimensions are the same, select diagonal
    elements. Special case for diag(A*B).
  * Permute (and transpose): Permute dimensions of expressions

