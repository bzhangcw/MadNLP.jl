using JuMP, NLPModels, CUTEst
using CSV, DataFrames, ProgressMeter, Dates
# my version
using MadNLP, MadNLPHSL, MadNLPMumps, MadNLPPardiso
ENV["MADNLP_PARDISO_LIBRARY_PATH"] = "/home/chuwen/panua-pardiso-20240229-linux/lib/libpardiso.so"


# file path 
fname = ARGS[1]
@info "" "file path" fname
solvername = ARGS[2]
if solvername == "pardiso"
    solver = MadNLPPardiso.PardisoSolver
elseif solvername == "hsl"
    solver = MadNLPHSL.Ma57Solver
end
@info "" "using solver" solver


# warm_up_probs

problem = CUTEstModel("A4X12", decode=true)
r = madnlp(problem, linear_solver=solver, max_wall_time=900.0, max_iter=10, print_level=MadNLP.INFO, tol=1e-6)

fstamp = Dates.format(Dates.now(), dateformat"yyyy/mm/dd HH:MM:SS")
fstamppath = Dates.format(Dates.now(), dateformat"yyyymmddHHMM")
csvfile = open("cutest-$fstamppath.csv", "w")

# chuwen's collection
df = CSV.read(fname, DataFrame)
p = Progress(length(df.name); showspeed=true)

header = [
    :name
    :nvar
    :ncon
    :nnzj
]
compute = Dict(
    :neq => (meta) -> length(meta.jfix)
)
attrs = [
    # bounds_multipliers_reliable  
    :primal_feas
    :dual_feas                    
    # dual_residual_reliable       
    # primal_residual_reliable
    :total_time                 
    :linear_solver_time       
    :iter                        
    # solution
    # iter_reliable                
    # solution_reliable
    # solver_specific
    # solver_specific_reliable
    # multipliers                  
    # multipliers_L                
    # multipliers_U                
    # multipliers_reliable         
    :status
    :status_reliable
    # time_reliable
    :objective
    :objective_reliable
]

write(csvfile, join([header..., keys(compute)..., attrs..., "update"], ","), "\n")
getmadnlpkey(stats, attr) = begin 
    if hasproperty(stats, attr)
        getfield(stats, attr) 
    elseif hasproperty(stats, :counters)
        cc = getfield(stats, :counters) 
        if hasproperty(cc, attr)
            getfield(cc, attr)
        else
            0.0
        end
    else 
        0.0
    end
end


for ℓ in eachrow(df)
    @info "" "problem name" ℓ.name
    if ℓ.n < 100
        @info "" ℓ.name too small
        continue
    end
    try
        problem = CUTEstModel(ℓ.name)
 
        stats = madnlp(problem, 
            linear_solver=solver, 
            max_wall_time=900.0, 
            max_iter=1000, 
            print_level=MadNLP.INFO, 
            tol=1e-6
        )
        _header=[
            getfield(problem.meta, attr) for attr in header
        ]
        _tocompute=[
            compute[key](problem.meta) for key in keys(compute)
        ]
        _attrs = [
            getmadnlpkey(stats, attr) for attr in attrs
        ]
        write(
            csvfile, join([_header..., _tocompute..., _attrs...], ","), 
            ",", fstamp, "\n"
        )
        flush(csvfile)
        finalize(problem)
    catch e
        finalize(problem)
    finally
    end
    ProgressMeter.next!(p)
end


