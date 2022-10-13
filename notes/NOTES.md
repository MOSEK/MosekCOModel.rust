# Designs to reconsider

## References and move
Currently, we build Exprs by moving operands into the Expr structure. Consider using instead references:
- PROS:
  - Smaller memory use per expression
  - Resuse same sub-expression without copying
  - Won't have to clone operands.
- CONS:
  - Less clean syntax,
  - More complicated lifetimes of expressions

## Matrixes

We have a sparse and a dense matrix. Implementing these as Traits instead might
be useful. This would allow things like:
- Matrix objects that borrow data
- Matrix objects that generate the data instead of storing it.

