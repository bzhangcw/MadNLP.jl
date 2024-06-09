using MadNLP, MadNLPHSL, MadNLPMumps, MadNLPPardiso
using CUTEst

fname = "BRAINPC1"
problem = CUTEstModel(fname, decode=true)

r = madnlp(
    problem, 
    linear_solver= MadNLPHSL.Ma57Solver, 
    max_wall_time=900.0, 
    max_iter=1000, 
    print_level=MadNLP.INFO,
    tol=1e-6,
    output_file="$fname.log"
)

finalize(problem)