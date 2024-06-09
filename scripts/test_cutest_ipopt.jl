using JuMP, AmplNLWriter, AmplNLReader, NLPModels, CUTEst, NLPModelsIpopt
using CSV, DataFrames, ProgressMeter, Dates


fstamp = Dates.format(Dates.now(), dateformat"yyyy/mm/dd HH:MM:SS")
fstamppath = Dates.format(Dates.now(), dateformat"yyyymmddHHMM")
csvfile = open("cutest-$fstamppath.csv", "w")

df = CSV.read("data/cutest_nlp_problem.csv", DataFrame)
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
    :elapsed_time                 
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
for ℓ in eachrow(df)
    println(ℓ.name)
    if ℓ.n < 100
        @info "" ℓ.name too small
        continue
    end
    try
        problem = CUTEstModel(ℓ.name, decode=true)
        stats = ipopt(problem, 
            max_iter=1000, 
            linear_solver="pardiso",
            pardisolib="/home/chuwen/panua-pardiso-20240229-linux/lib/libpardiso.so"
        )
        _header=[
            getfield(problem.meta, attr) for attr in header
        ]
        _tocompute=[
            compute[key](problem.meta) for key in keys(compute)
        ]
        _attrs = [
            getfield(stats, attr) for attr in attrs
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
