# Designs to reconsider

## References and move
Currently, we build Exprs by moving operands into the Expr structure. Consider using instead references:
- PROS:
  - Smaller memory use per expression
  - Resuse same sub-expression without copying
  - Won't have to clone operands.
- CONS:
  - Less clean syntax,
  - More complicated lifetimes of expressions, especially in the helper
    functions for things like multiplication and addition

