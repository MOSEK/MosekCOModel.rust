A set of graphical demos using optimization.

# Building

To build all demos, type

```
cargo build --workspace
```

# demos

## `ellipsoid-approximation`

Computes two-dimensional ellipsoid approximations of the union and the intersection of a set of ellipsoids:

- The smallest ellipsoid containing a given set of ellipsoids
- The largest ellipsoid contained in all of a given set of ellipsoids

## `ellipsoid-approximation-3d`
Computes the minimal three-dimensional ellipsoid containing a given set of ellipsoids/

## `lowner-john-2d`

Computes in two dimensions
- The minimal ellipsoid containing a set of points
- The maximal ellipsoid contained in the intersection of a set of convex polygons, or equivalently, the maximum ellipsoid contained in `{ x∊ R² | Ax<b }`.

## `lowner-john-outer-3d`
Computes the minimal ellipsoid containing a set of points in three dimensions.

## `truss`

Implements a simple truss design model: Given an amount of material, a base
layout of a truss and a set of forces acting on the truss, compute the
allocation of material to the individual bars on the truss that minimizes the
overall stress on the truss when subject to the forces.

This requires an input file specifying base truss layout and forces. To run, use

```
cargo run -p truss truss/data/bridge.trs
```

# `optserver`

Demonstrates how to implement an alternative backend for
`mosekcomodel::ModelAPI`. In this case, it defines a backend supporting linear
and ranged constraints and variables, and offloads optimization to a MOSEK Optserver
instance.
