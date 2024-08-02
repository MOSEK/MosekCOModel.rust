This demo shows an animated bounding ellipsoids model.

The problem: Given a set of 2-dimensional ellipses, find 
1. The minimal (by area) ellipse that contains all ellipses.
2. The maximum (by area) ellipse that is contained in all ellipses (if all ellipses intersect).

Both can be formulated as a
quadratic (or semidefinite) optimization problem.

See:
- Ben-Tal and Nemirovski "Lectures on Modern Convex Optimization", 2001.
- An implementation in Julia: _`https://jump.dev/JuMP.jl/stable/tutorials/conic/min_ellipse/`

# Computational errors

This example illustrates well computational errors in semindefinite
programming. The outer ellipsoid generally produces a quite precise
approximation (visually, it is seen to exactly touch the edges of the contained
ellipsoids), but the inner ellipsoid has more visually obvious errors:
Especially when it is has a high aspect ratio, it clearly tends to intersect
the ellipses rather than just touch.
