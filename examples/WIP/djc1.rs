////
//   Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//   File:      $${file}
//
//   Purpose: Demonstrates how to solve the problem with two disjunctions:
//
//      minimize    2x0 + x1 + 3x2 + x3
//      subject to   x0 + x1 + x2 + x3 >= -10
//                  (x0-2x1<=-1 and x2=x3=0) or (x2-3x3<=-2 and x1=x2=0)
//                  x0=2.5 or x1=2.5 or x2=2.5 or x3=2.5
////
package com.mosek.fusion.examples;
import mosek.fusion.*;

public class djc1 {
  public static void main(String[] args)
  throws SolutionError {

    try(Model M = new Model("djc1"))
    {
      // Create variable 'x' of length 4
      Variable x = M.variable("x", 4);

      // First disjunctive constraint
      M.disjunction( DJC.AND( DJC.term(Expr.dot(new double[]{1,-2,0,0}, x), Domain.lessThan(-1)), // x0 - 2x1 <= -1  
                              DJC.term(x.slice(2, 4), Domain.equalsTo(0)) ),                      // x2 = x3 = 0
                     DJC.AND( DJC.term(Expr.dot(new double[]{0,0,1,-3}, x), Domain.lessThan(-2)), // x2 - 3x3 <= -2
                              DJC.term(x.slice(0, 2), Domain.equalsTo(0)) ) );                    // x0 = x1 = 0

      // Second disjunctive constraint
      // Array of terms reading x_i = 2.5 for i = 0,1,2,3
      Term[] terms = new Term[4];
      for(int i = 0; i < 4; i++)
        terms[i] = DJC.term(x.index(i), Domain.equalsTo(2.5));
      // The disjunctive constraint from the array of terms
      M.disjunction(terms);

      // The linear constraint
      M.constraint(Expr.sum(x), Domain.greaterThan(-10));

      // Objective
      M.objective(ObjectiveSense.Minimize, Expr.dot(new double[]{2,1,3,1}, x));

      // Useful for debugging
      M.writeTask("djc.ptf");                                 // Save to a readable file
      M.setLogHandler(new java.io.PrintWriter(System.out));   // Enable log output

      // Solve the problem
      M.solve();

      if (M.getPrimalSolutionStatus() == SolutionStatus.Optimal) {
        double[] sol = x.level();
        System.out.printf("[x0,x1,x2,x3] = [%e, %e, %e, %e]\n", sol[0], sol[1], sol[2], sol[3]);
      }
      else {
        System.out.printf("Anoter solution status");
      }
    }
  }
}
