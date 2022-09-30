# Ideas to implement:

- Evaluating expressions in Model. Only primal values make sense:
```
    m : Model;
    e : ExprTrait;
    let sol = m.primal_solution<E:ExptTrait>(soltype,e);
```
