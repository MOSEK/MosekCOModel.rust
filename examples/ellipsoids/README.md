This library provides functions to generate ellipsoidal constraints. Depending
on the constraint, the ellipsoids need to be represented in different ways. The base representation, `Pq` is
```
{ x : || Px+q || <= 1 }
```
expanding this we get the notation `Abc`
```
{ x : x'Ax + b'x + c <= 0 }
```
where
- `A = P'P`
- `b = 2q`
- `c = q'q-1`
Since `P` is symmetric positive (semi)definite, we know `P` has a symmetric square root, so we can use variables
```
Psq = P²
Pq  = P * q
```
so the representation becomes `PsqPq`
```
{ x : || √Psq + (Psq^{-2})q || <= 1 }
```


The inverse representation of `Pq` is `Zw`
```
{ Zx+w : ||x|| <= 1 }
```

Currently following constraints are supported:

1. Given 
    - a fixed ellipsoid `E`
    - ellipse variables (Psq,Pq) such that `Psq=P²`, `Pq=P*q`
    Define a constraint to the effect that `(P,q)` must contain each `E`.
2. Given 
    - a fixed ellipsoid `E`
    - ellipse variables (Z,w) such that `Z=P^{-1}`, `w=-Zq` 
    define a constraint to the effect that `(P,q)` must be contained in `E`.
3. Given 
    - a set of points `P_i`,
    - ellipse variables `(P,q)`
    define a constraint to the effect that `Pq` contains all points `P_i`
4. Given 
    - a scalar variable `v`
    - ellipse variables `(P,q)`
    define a constraint such that `v` is greater than a convex growing function of the volume if `(P,q)`. The effect of minimizing `v` will then be to minimize the volume of `(P,q)`.
5. Given 
    - a scalar variable `v`
    - an ellipse `(Z,w)` such that `Z=P^{-1}`, `w = Z*q`
    define a constraint such that `v` is less than a concave decreasing function of the volume if `(P,q)`. The effect of maximizing `v` will then be to maximize the volume of `(P,q)`.

Note that the constraints cannot be arbitrarily mixed since the representations of `(P,q)` differ.
