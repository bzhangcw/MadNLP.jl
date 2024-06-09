using MadNLP, MadNLPHSL, MadNLPMumps, MadNLPPardiso
using CUTEst

problem = CUTEstModel("A4X12", decode=true)

r = madnlp(problem, linear_solver=MadNLPMumps.MumpsSolver, max_wall_time=900.0, max_iter=10, print_level=MadNLP.INFO, tol=1e-6)
r = madnlp(problem, linear_solver=MadNLPPardiso.PardisoSolver, max_wall_time=900.0, max_iter=10, print_level=MadNLP.INFO, tol=1e-6)
r = madnlp(problem, linear_solver=MadNLPHSL.Ma57Solver, max_wall_time=900.0, max_iter=10, print_level=MadNLP.INFO, tol=1e-6)