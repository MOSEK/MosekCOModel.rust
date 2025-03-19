package com.mosek.fusion.examples;
import mosek.fusion.*;

public class timing {
  static final int REP = 10;
  static double stacking1()
    throws SolutionError {
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {100,100,100});
      int[][] sp = new int[(1000000-1)/11+1][];
      for (int i = 0, j = 0; j < 1000000; ++i, j += 11) { sp[i] = new int[]{ j/10000,(j%10000)/100,j%100 }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{100,100,100}).sparse(sp));

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.stack(d, new Expression[]{x,y,x,y}).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Stacking, mixed",(T1-T0)/REP);
    }
  }

  static double stacking2()
    throws SolutionError {
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {100,100,100});
      int[][] sp = new int[(1000000-1)/11+1][];

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.stack(d, new Expression[]{x,x,x,x}).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Stacking, dense",(T1-T0)/REP);
    }
  }

  static double stacking3()
    throws SolutionError {
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {100,100,100});
      int[][] sp = new int[(1000000-1)/11+1][];
      for (int i = 0, j = 0; j < 1000000; ++i, j += 11) { sp[i] = new int[]{ j/10000,(j%10000)/100,j%100 }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{100,100,100}).sparse(sp));

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.stack(d, new Expression[]{y,y,y,y}).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Stacking, sparse",(T1-T0)/REP);
    }
  }

  static double mul1()
    throws SolutionError 
  {
    int N = 400;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      Matrix m = Matrix.dense(N,N, 1.1);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(Expr.add(x,Expr.transpose(x)),m).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul dense X * dense M", (T1-T0)/REP);
    }
  }

  static double mul2()
    throws SolutionError 
  {
    int N = 400;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      Matrix m = Matrix.dense(N,N, 1.1);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(Expr.add(y,Expr.transpose(y)),m).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul sparse X * dense M", (T1-T0)/REP);
    }
  }

  static double mul3()
    throws SolutionError 
  {
    int N = 800;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      int mnnz = (N*N-1)/11+1;
      int[] subi = new int[mnnz];
      int[] subj = new int[mnnz];
      double[] valij = new double[mnnz];
      for (int i = 0, j = 0; i < mnnz; ++i, j += 11) { subi[i] = j/N; subj[i] = j%N; valij[i] = (j%100)/50.0; }
      Matrix m = Matrix.sparse(N,N,subi,subj,valij);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(Expr.add(x,Expr.transpose(x)),m).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul dense X * sparse M", (T1-T0)/REP);
    }
  }
  
  static double mul4()
    throws SolutionError 
  {
    int N = 800;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      int mnnz = (N*N-1)/11+1;
      int[] subi = new int[mnnz];
      int[] subj = new int[mnnz];
      double[] valij = new double[mnnz];
      for (int i = 0, j = 0; i < mnnz; ++i, j += 11) { subi[i] = j/N; subj[i] = j%N; valij[i] = (j%100)/50.0; }
      Matrix m = Matrix.sparse(N,N,subi,subj,valij);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(Expr.add(y,Expr.transpose(y)),m).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul sparse X * sparse M", (T1-T0)/REP);
    }
  }
  









  static double mul5()
    throws SolutionError 
  {
    int N = 400;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      Matrix m = Matrix.dense(N,N, 1.1);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(m,Expr.add(x,Expr.transpose(x))).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul dense M * dense X", (T1-T0)/REP);
    }
  }

  static double mul6()
    throws SolutionError 
  {
    int N = 800;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      Matrix m = Matrix.dense(N,N, 1.1);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(m,Expr.add(y,Expr.transpose(y))).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul dense M * sparse X", (T1-T0)/REP);
    }
  }

  static double mul7()
    throws SolutionError 
  {
    int N = 800;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      int mnnz = (N*N-1)/11+1;
      int[] subi = new int[mnnz];
      int[] subj = new int[mnnz];
      double[] valij = new double[mnnz];
      for (int i = 0, j = 0; i < mnnz; ++i, j += 11) { subi[i] = j/N; subj[i] = j%N; valij[i] = (j%100)/50.0; }
      Matrix m = Matrix.sparse(N,N,subi,subj,valij);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(m,Expr.add(x,Expr.transpose(x))).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul sparse M * dense X", (T1-T0)/REP);
    }
  }
  
  static double mul8()
    throws SolutionError 
  {
    int N = 800;
    try (Model M = new Model()) {
      Variable x = M.variable(new int[] {N,N});
      int[][] sp = new int[(N*N-1)/7+1][];
      for (int i = 0, j = 0; j < N*N; ++i, j += 7) { sp[i] = new int[]{ j/N,j%N }; }
      Variable y = M.variable(Domain.unbounded().withShape(new int[]{N,N}).sparse(sp));

      int mnnz = (N*N-1)/11+1;
      int[] subi = new int[mnnz];
      int[] subj = new int[mnnz];
      double[] valij = new double[mnnz];
      for (int i = 0, j = 0; i < mnnz; ++i, j += 11) { subi[i] = j/N; subj[i] = j%N; valij[i] = (j%100)/50.0; }
      Matrix m = Matrix.sparse(N,N,subi,subj,valij);

      WorkStack rs = new WorkStack();
      WorkStack ws = new WorkStack();
      WorkStack xs = new WorkStack();

      double T0 = 0.001 * System.currentTimeMillis();
      for (int i = 0; i < REP; ++i) {
          for (int d = 0; d < 3; ++d) {
              rs.clear();
              Expr.mul(m,Expr.add(y,Expr.transpose(y))).eval(rs,ws,xs);
          }
      }
      double T1 = 0.001 * System.currentTimeMillis();

      return (T1-T0)/REP;
      //System.out.printf("%-30s: Avg time: %.2f secs\n","Mul sparse M * sparse X", (T1-T0)/REP);
    }
  }





  public static void main(String[] args)
  throws SolutionError {
    if (args.length > 1) {
        if (args[1].equals("stacking1"))
            System.out.printf("%.3f",stacking1());
        else if (args[1].equals("stacking2"))
            System.out.printf("%.3f",stacking2());
        else if (args[1].equals("stacking3"))
            System.out.printf("%.3f",stacking3());
        else if (args[1].equals("mul1"))
            System.out.printf("%.3f",mul1());
        else if (args[1].equals("mul2"))
            System.out.printf("%.3f",mul2());
        else if (args[1].equals("mul3"))
            System.out.printf("%.3f",mul3());
        else if (args[1].equals("mul4"))
            System.out.printf("%.3f",mul4());
        else if (args[1].equals("mul5"))
            System.out.printf("%.3f",mul5());
        else if (args[1].equals("mul6"))
            System.out.printf("%.3f",mul6());
        else if (args[1].equals("mul7"))
            System.out.printf("%.3f",mul7());
        else if (args[1].equals("mul8"))
            System.out.printf("%.3f",mul8());
    }
  }
}

