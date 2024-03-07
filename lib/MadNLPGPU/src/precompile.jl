import PrecompileTools

PrecompileTools.@setup_workload begin
    struct HS15Model{T,VT} <: NLPModels.AbstractNLPModel{T,VT}
        meta::NLPModels.NLPModelMeta{T, VT}
        counters::NLPModels.Counters
    end

    function HS15Model(T = Float64; x0=CUDA.zeros(Float64,2), y0=CUDA.zeros(Float64,2))
        return HS15Model(
            NLPModels.NLPModelMeta(
                2,     #nvar
                ncon = 2,
                nnzj = 4,
                nnzh = 3,
                x0 = x0,
                y0 = y0,
                lvar = CuArray([-Inf, -Inf]),
                uvar = CuArray([0.5, Inf]),
                lcon = CuArray([1.0, 0.0]),
                ucon = CuArray([Inf, Inf]),
                minimize = true
            ),
            NLPModels.Counters()
        )
    end

    function NLPModels.obj(nlp::HS15Model, x::AbstractVector)
        CUDA.@allowscalar begin 
            return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
        end
    end

    function NLPModels.grad!(nlp::HS15Model, x::AbstractVector, g::AbstractVector)
        CUDA.@allowscalar begin
            z = x[2] - x[1]^2
            g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
            g[2] = 200.0 * z
        end
        return
    end

    function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector)
        CUDA.@allowscalar begin 
            c[1] = x[1] * x[2]
            c[2] = x[1] + x[2]^2
        end
    end

    function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
        CUDA.@allowscalar begin
            copyto!(I, [1, 1, 2, 2])
            copyto!(J, [1, 2, 1, 2])
        end
    end

    function NLPModels.jac_coord!(nlp::HS15Model, x::AbstractVector, J::AbstractVector)
        CUDA.@allowscalar begin
            J[1] = x[2]    # (1, 1)
            J[2] = x[1]    # (1, 2)
            J[3] = 1.0     # (2, 1)
            J[4] = 2*x[2]  # (2, 2)
        end
        return J
    end

    function NLPModels.jprod!(nlp::HS15Model, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
        CUDA.@allowscalar begin
            jv[1] = x[2] * v[1] + x[1] * v[2]
            jv[2] = v[1] + 2 * x[2] * v[2]
        end
        return jv
    end

    function NLPModels.jtprod!(nlp::HS15Model, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
        CUDA.@allowscalar begin 
            jv[1] = x[2] * v[1] + v[2]
            jv[2] = x[1] * v[1] + 2 * x[2] * v[2]
        end
        return jv
    end
    
    function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
        CUDA.@allowscalar begin 
            copyto!(I, [1, 2, 2])
            copyto!(J, [1, 1, 2])
        end
    end

    function NLPModels.hess_coord!(nlp::HS15Model, x, y, H::AbstractVector; obj_weight=1.0)
        CUDA.@allowscalar begin
            # Objective
            H[1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
            H[2] = obj_weight * (-400.0 * x[1])
            H[3] = obj_weight * 200.0
            # First constraint
            H[2] += y[1] * 1.0
            # Second constraint
            H[3] += y[2] * 2.0
        end
        return H
    end
    m = HS15Model()
    
    PrecompileTools.@compile_workload begin
        
        s = MadNLP.MadNLPSolver(m; print_level = MadNLP.ERROR)
        MadNLP.madnlp(m)
        MadNLP.restore!(s)
        MadNLP.robust!(s)
    end
end
