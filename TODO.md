
# The Canonical TODO list!

This is not a roadmap, but a list of stuff to pick up at some point when there
is time and interest in it.

## Parameters
 
This is a major change. This requires a complete rework of the expression
evaluation format to include linear parameter expressions. Definitely not a
short-term goal.

What is required:
- An N-dimensional parameter struct and expressions for expressions that multiply or add variables and parameters,
- Functionality to update parameter values in the Model object,
- Functionality to recompute and update non-zeros in the coefficient matrixes before a resolve

## Warmstart and initial solutions

Starting points for MIP or initial solutions for warm-starting Simplex (or
maybe Interior point, some day).

Functionality for inputting full or partial solution values that a solver can choose to use in any way it wishes.

## Split project into Model part and solvers parameters

Split the project into 
- a main project providing the `ModelAPI`, domains, expressions and evaluation, and which does not depend on the `mosek` crate,
- backend projects that implement the solver backend for `ModelAPI`, for example `mosekcomodel_mosek`, `mosekcomodel_highs`, etc.

