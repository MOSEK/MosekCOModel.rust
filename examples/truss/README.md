# Truss design

This model implements a simple (quadratic, non-robust) truss design model.

A truss is a construction consisting of 
- a set of nodes
- a subset of which are fixed, and
- a set of bars that connect the nodes

All nodes must be connected directly or indirectly to all other nodes, and for an N-dimensional truss 
there must be at least N (linearly independent) fixed nodes.

We consider a set of indenentant sets of stresses `F_1`, ..., `F_m`, each `F_i`
being a vector of stresses on the individual nodes. We now wish to assign
volumes to the bars such that the maximum total stress for all `F_i` is
minimized, subject to the amount of volume available.

The model can be generalized to any number of dimensions and any number of sets
of forces  but in the present example we work in 2 dimensions and with just one set of forces.

See:
- Ben-Tal and Nemirovski "Lectures on Modern Convex Optimization", 2001.
