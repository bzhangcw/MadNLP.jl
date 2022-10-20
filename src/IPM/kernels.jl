
# KKT system updates -------------------------------------------------------
# Set diagonal
function set_aug_diagonal!(kkt::AbstractKKTSystem, solver::MadNLPSolver{T}) where T
    kkt.pr_diag .= solver.zl./(solver.x.-solver.xl) .+ solver.zu./(solver.xu.-solver.x)
    fill!(kkt.du_diag, zero(T))
    return
end
function set_aug_diagonal!(kkt::SparseUnreducedKKTSystem, solver::MadNLPSolver{T}) where T
    fill!(kkt.pr_diag, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_lower .= .-sqrt.(solver.zl_r)
    kkt.u_lower .= .-sqrt.(solver.zu_r)
    kkt.l_diag  .= solver.xl_r .- solver.x_lr
    kkt.u_diag  .= solver.x_ur .- solver.xu_r
    return
end

# Robust restoration
function set_aug_RR!(kkt::AbstractKKTSystem, solver::MadNLPSolver, RR::RobustRestorer)
    kkt.pr_diag .= solver.zl./(solver.x.-solver.xl) .+ solver.zu./(solver.xu.-solver.x) .+ RR.zeta.*RR.D_R.^2
    kkt.du_diag .= .-RR.pp./RR.zp .- RR.nn./RR.zn
    return
end
function set_aug_RR!(kkt::SparseUnreducedKKTSystem, solver::MadNLPSolver, RR::RobustRestorer)
    kkt.pr_diag .= RR.zeta.*RR.D_R.^2
    kkt.du_diag .= .-RR.pp./RR.zp.-RR.nn./RR.zn
    kkt.l_lower .=.-sqrt.(solver.zl_r)
    kkt.u_lower .=.-sqrt.(solver.zu_r)
    kkt.l_diag  .= solver.xl_r .- solver.x_lr
    kkt.u_diag  .= solver.x_ur .- solver.xu_r
    return
end

# Set RHS
function set_aug_rhs!(solver::MadNLPSolver, kkt::AbstractKKTSystem, c)
    primal(solver.p) .= .-solver.f .+ solver.mu ./ (solver.x.-solver.xl) .- solver.mu ./ (solver.xu.-solver.x) .-solver.jacl
    dual(solver.p)   .= .-c
    return
end
function set_aug_rhs!(solver::MadNLPSolver, kkt::SparseUnreducedKKTSystem, c)
    primal(solver.p) .= .-solver.f .+ solver.zl .- solver.zu .- solver.jacl
    dual(solver.p) .= .-c
    dual_lb(solver.p) .= (solver.xl_r-solver.x_lr) .* kkt.l_lower .+ solver.mu ./ kkt.l_lower
    dual_ub(solver.p) .= (solver.xu_r-solver.x_ur) .* kkt.u_lower .- solver.mu ./ kkt.u_lower
    return
end

function set_aug_rhs_ifr!(solver::MadNLPSolver{T}, kkt::SparseUnreducedKKTSystem,c) where T
    fill!(primal(solver._w1), zero(T))
    fill!(dual_lb(solver._w1), zero(T))
    fill!(dual_ub(solver._w1), zero(T))
    dual(solver._w1) .= .-c
    return
end

# Set RHS RR
function set_aug_rhs_RR!(
    solver::MadNLPSolver, kkt::AbstractKKTSystem, RR::RobustRestorer, rho,
)
    primal(solver.p) .= .-RR.f_R.-solver.jacl.+RR.mu_R./(solver.x.-solver.xl).-RR.mu_R./(solver.xu.-solver.x)
    dual(solver.p) .= .-solver.c.+RR.pp.-RR.nn.+(RR.mu_R.-(rho.-solver.y).*RR.pp)./RR.zp.-(RR.mu_R.-(rho.+solver.y).*RR.nn)./RR.zn
    return
end

# Finish
function finish_aug_solve!(solver::MadNLPSolver, kkt::AbstractKKTSystem, mu)
    dual_lb(solver.d) .= (mu.-solver.zl_r.*solver.dx_lr)./(solver.x_lr.-solver.xl_r).-solver.zl_r
    dual_ub(solver.d) .= (mu.+solver.zu_r.*solver.dx_ur)./(solver.xu_r.-solver.x_ur).-solver.zu_r
    return
end
function finish_aug_solve!(solver::MadNLPSolver, kkt::SparseUnreducedKKTSystem, mu)
    dual_lb(solver.d) .*= .-kkt.l_lower
    dual_ub(solver.d) .*=   kkt.u_lower
    dual_lb(solver.d) .= (mu.-solver.zl_r.*solver.dx_lr) ./ (solver.x_lr.-solver.xl_r) .- solver.zl_r
    dual_ub(solver.d) .= (mu.+solver.zu_r.*solver.dx_ur) ./ (solver.xu_r.-solver.x_ur) .- solver.zu_r
    return
end

# Initial
function set_initial_rhs!(solver::MadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    primal(solver.p) .= .-solver.f .+ solver.zl .- solver.zu
    fill!(dual(solver.p), zero(T))
    return
end
function set_initial_rhs!(solver::MadNLPSolver{T}, kkt::SparseUnreducedKKTSystem) where T
    primal(solver.p) .= .-solver.f.+solver.zl.-solver.zu
    fill!(dual(solver.p), zero(T))
    fill!(dual_lb(solver.p), zero(T))
    fill!(dual_ub(solver.p), zero(T))
    return
end

# Set ifr
function set_aug_rhs_ifr!(solver::MadNLPSolver{T}, kkt::AbstractKKTSystem) where T
    fill!(primal(solver._w1), zero(T))
    dual(solver._w1) .= .-solver.c
    return
end

# Finish RR
function finish_aug_solve_RR!(dpp, dnn, dzp, dzn, l, dl, pp, nn, zp, zn, mu_R, rho)
    dpp .= (mu_R.+pp.*dl.-(rho.-l).*pp)./zp
    dnn .= (mu_R.-nn.*dl.-(rho.+l).*nn)./zn
    dzp .= (mu_R.-zp.*dpp)./pp.-zp
    dzn .= (mu_R.-zn.*dnn)./nn.-zn
    return
end

# Kernel functions ---------------------------------------------------------
is_valid(val::Real) = !(isnan(val) || isinf(val))
function is_valid(vec::AbstractArray)
    @inbounds for i=1:length(vec)
        is_valid(vec[i]) || return false
    end
    return true
end
is_valid(args...) = all(is_valid(arg) for arg in args)

function get_varphi(obj_val, x_lr, xl_r, xu_r, x_ur, mu)
    varphi = obj_val
    @inbounds @simd for i=1:length(x_lr)
        xll = x_lr[i]-xl_r[i]
        xll < 0 && return Inf
        varphi -= mu*log(xll)
    end
    @inbounds @simd for i=1:length(x_ur)
        xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return Inf
        varphi -= mu*log(xuu)
    end
    return varphi
end

@inline get_inf_pr(c) = norm(c, Inf)

function get_inf_du(f, zl, zu, jacl, sd)
    inf_du = 0.0
    @inbounds @simd for i=1:length(f)
        inf_du = max(inf_du,abs(f[i]-zl[i]+zu[i]+jacl[i]))
    end
    return inf_du/sd
end

function get_inf_compl(x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu, sc)
    inf_compl = 0.0
    @inbounds @simd for i=1:length(x_lr)
        inf_compl = max(inf_compl,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu))
    end
    @inbounds @simd for i=1:length(x_ur)
        inf_compl = max(inf_compl,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu))
    end
    return inf_compl/sc
end

function get_varphi_d(f, x, xl, xu, dx, mu)
    varphi_d = 0.0
    @inbounds @simd for i=1:length(f)
        varphi_d += (f[i] - mu/(x[i]-xl[i]) + mu/(xu[i]-x[i])) *dx[i]
    end
    return varphi_d
end

function get_alpha_max(x, xl, xu, dx, tau)
    alpha_max = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max=min(alpha_max,(-x[i]+xl[i])*tau/dx[i]))
        dx[i]>0 && (alpha_max=min(alpha_max,(-x[i]+xu[i])*tau/dx[i]))
    end
    return alpha_max
end

function get_alpha_z(zl_r, zu_r, dzl, dzu, tau)
    alpha_z = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i] < 0 && (alpha_z=min(alpha_z,-zl_r[i]*tau/dzl[i]))
     end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i] < 0 && (alpha_z=min(alpha_z,-zu_r[i]*tau/dzu[i]))
    end
    return alpha_z
end

function get_obj_val_R(p, n, D_R, x, x_ref, rho, zeta)
    obj_val_R = 0.
    @inbounds @simd for i=1:length(p)
        obj_val_R += rho*(p[i]+n[i]) .+ zeta/2*D_R[i]^2*(x[i]-x_ref[i])^2
    end
    return obj_val_R
end

@inline get_theta(c) = norm(c, 1)

function get_theta_R(c, p, n)
    theta_R = 0.0
    @inbounds @simd for i=1:length(c)
        theta_R += abs(c[i]-p[i]+n[i])
    end
    return theta_R
end

function get_inf_pr_R(c, p, n)
    inf_pr_R = 0.0
    @inbounds @simd for i=1:length(c)
        inf_pr_R = max(inf_pr_R,abs(c[i]-p[i]+n[i]))
    end
    return inf_pr_R
end

function get_inf_du_R(f_R, l, zl, zu, jacl, zp, zn, rho, sd)
    inf_du_R = 0.0
    @inbounds @simd for i=1:length(zl)
        inf_du_R = max(inf_du_R,abs(f_R[i]-zl[i]+zu[i]+jacl[i]))
    end
    @inbounds @simd for i=1:length(zp)
        inf_du_R = max(inf_du_R,abs(rho-l[i]-zp[i]))
    end
    @inbounds @simd for i=1:length(zn)
        inf_du_R = max(inf_du_R,abs(rho+l[i]-zn[i]))
    end
    return inf_du_R / sd
end

function get_inf_compl_R(x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, pp, zp, nn, zn, mu_R, sc)
    inf_compl_R = 0.0
    @inbounds @simd for i=1:length(x_lr)
        inf_compl_R = max(inf_compl_R,abs((x_lr[i]-xl_r[i])*zl_r[i]-mu_R))
    end
    @inbounds @simd for i=1:length(xu_r)
        inf_compl_R = max(inf_compl_R,abs((xu_r[i]-x_ur[i])*zu_r[i]-mu_R))
    end
    @inbounds @simd for i=1:length(pp)
        inf_compl_R = max(inf_compl_R,abs(pp[i]*zp[i]-mu_R))
    end
    @inbounds @simd for i=1:length(nn)
        inf_compl_R = max(inf_compl_R,abs(nn[i]*zn[i]-mu_R))
    end
    return inf_compl_R / sc
end

function get_alpha_max_R(x, xl, xu, dx, pp, dpp, nn, dnn, tau_R)
    alpha_max_R = 1.0
    @inbounds @simd for i=1:length(x)
        dx[i]<0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xl[i])*tau_R/dx[i]))
        dx[i]>0 && (alpha_max_R=min(alpha_max_R,(-x[i]+xu[i])*tau_R/dx[i]))
    end
    @inbounds @simd for i=1:length(pp)
        dpp[i]<0 && (alpha_max_R=min(alpha_max_R,-pp[i]*tau_R/dpp[i]))
    end
    @inbounds @simd for i=1:length(nn)
        dnn[i]<0 && (alpha_max_R=min(alpha_max_R,-nn[i]*tau_R/dnn[i]))
    end
    return alpha_max_R
end

function get_alpha_z_R(zl_r, zu_r, dzl, dzu, zp, dzp, zn, dzn, tau_R)
    alpha_z_R = 1.0
    @inbounds @simd for i=1:length(zl_r)
        dzl[i]<0 && (alpha_z_R=min(alpha_z_R,-zl_r[i]*tau_R/dzl[i]))
    end
    @inbounds @simd for i=1:length(zu_r)
        dzu[i]<0 && (alpha_z_R=min(alpha_z_R,-zu_r[i]*tau_R/dzu[i]))
    end
    @inbounds @simd for i=1:length(zp)
        dzp[i]<0 && (alpha_z_R=min(alpha_z_R,-zp[i]*tau_R/dzp[i]))
    end
    @inbounds @simd for i=1:length(zn)
        dzn[i]<0 && (alpha_z_R=min(alpha_z_R,-zn[i]*tau_R/dzn[i]))
    end
    return alpha_z_R
end

function get_varphi_R(obj_val, x_lr, xl_r, xu_r, x_ur, pp, nn, mu_R)
    varphi_R = obj_val
    @inbounds @simd for i=1:length(x_lr)
        xll = x_lr[i]-xl_r[i]
        xll < 0 && return Inf
        varphi_R -= mu_R*log(xll)
    end
    @inbounds @simd for i=1:length(x_ur)
        xuu = xu_r[i]-x_ur[i]
        xuu < 0 && return Inf
        varphi_R -= mu_R*log(xuu)
    end
    @inbounds @simd for i=1:length(pp)
        pp[i] < 0 && return Inf
        varphi_R -= mu_R*log(pp[i])
    end
    @inbounds @simd for i=1:length(pp)
        nn[i] < 0 && return Inf
        varphi_R -= mu_R*log(nn[i])
    end
    return varphi_R
end

function get_F(c, f, zl, zu, jacl, x_lr, xl_r, zl_r, xu_r, x_ur, zu_r, mu)
    F = 0.0
    @inbounds for i=1:length(c)
        F = max(F, c[i])
    end
    @inbounds for i=1:length(f)
        F = max(F, f[i]-zl[i]+zu[i]+jacl[i])
    end
    @inbounds for i=1:length(x_lr)
        x_lr[i] >= xl_r[i] || return Inf
        zl_r[i] >= 0       || return Inf
        F = max(F, (x_lr[i]-xl_r[i])*zl_r[i]-mu)
    end
    @inbounds for i=1:length(x_ur)
        xu_r[i] >= x_ur[i] || return Inf
        zu_r[i] >= 0       || return Inf
        F = max(F, (xu_r[i]-xu_r[i])*zu_r[i]-mu)
    end
    return F
end

function get_varphi_d_R(f_R, x, xl, xu, dx, pp, nn, dpp, dnn, mu_R, rho)
    varphi_d = 0.0
    @inbounds @simd for i=1:length(x)
        varphi_d += (f_R[i] - mu_R/(x[i]-xl[i]) + mu_R/(xu[i]-x[i])) * dx[i]
    end
    @inbounds @simd for i=1:length(pp)
        varphi_d += (rho - mu_R/pp[i]) * dpp[i]
    end
    @inbounds @simd for i=1:length(nn)
        varphi_d += (rho - mu_R/nn[i]) * dnn[i]
    end
    return varphi_d
end

function initialize_variables!(x, xl, xu, bound_push, bound_fac)
    @inbounds @simd for i=1:length(x)
        if xl[i]!=-Inf && xu[i]!=Inf
            x[i] = min(
                xu[i]-min(bound_push*max(1,abs(xu[i])), bound_fac*(xu[i]-xl[i])),
                max(xl[i]+min(bound_push*max(1,abs(xl[i])),bound_fac*(xu[i]-xl[i])),x[i]),
            )
        elseif xl[i]!=-Inf && xu[i]==Inf
            x[i] = max(xl[i]+bound_push*max(1,abs(xl[i])), x[i])
        elseif xl[i]==-Inf && xu[i]!=Inf
            x[i] = min(xu[i]-bound_push*max(1,abs(xu[i])), x[i])
        end
    end
end

function adjust_boundary!(x_lr::VT, xl_r, x_ur, xu_r, mu) where {T, VT <: AbstractVector{T}}
    adjusted = 0
    c1 = eps(T)*mu
    c2= eps(T)^(3/4)
    @inbounds @simd for i=1:length(xl_r)
        if x_lr[i]-xl_r[i] < c1
            xl_r[i] -= c2*max(1,abs(x_lr[i]))
            adjusted += 1
        end
    end
    @inbounds @simd for i=1:length(xu_r)
        if xu_r[i]-x_ur[i] < c1
            xu_r[i] += c2*max(1, abs(x_ur[i]))
            adjusted += 1
        end
    end
    return adjusted
end

function get_rel_search_norm(x, dx)
    rel_search_norm = 0.0
    @inbounds @simd for i=1:length(x)
        rel_search_norm = max(
            rel_search_norm,
            abs(dx[i]) / (1.0 + abs(x[i])),
        )
    end
    return rel_search_norm
end

# Utility functions
function get_sd(l, zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(l, 1)+norm(zl_r, 1)+norm(zu_r, 1)) / max(1, (length(l)+length(zl_r)+length(zu_r))),
    ) / s_max
end

function get_sc(zl_r, zu_r, s_max)
    return max(
        s_max,
        (norm(zl_r,1)+norm(zu_r,1)) / max(1,length(zl_r)+length(zu_r)),
    ) / s_max
end

function get_mu(mu, mu_min, mu_linear_decrease_factor, mu_superlinear_decrease_power, tol)
    return max(
        mu_min,
        tol/10,
        min(mu_linear_decrease_factor*mu, mu^mu_superlinear_decrease_power),
    )
end

@inline get_tau(mu, tau_min) = max(tau_min, 1-mu)

function get_alpha_min(theta, varphi_d, theta_min, gamma_theta, gamma_phi, alpha_min_frac, del, s_theta, s_phi)
    if varphi_d<0
        if theta<=theta_min
            return alpha_min_frac*min(
                gamma_theta,gamma_phi*theta/(-varphi_d),
                del*theta^s_theta/(-varphi_d)^s_phi,
            )
        else
            return alpha_min_frac*min(
                gamma_theta,
                -gamma_phi*theta/varphi_d,
            )
        end
    else
        return alpha_min_frac*gamma_theta
    end
end

function is_switching(varphi_d, alpha, s_phi, del, theta, s_theta)
    return (varphi_d < 0) && (alpha*(-varphi_d)^s_phi > del*theta^s_theta)
end

function is_armijo(varphi_trial, varphi, eta_phi, alpha, varphi_d)
    return (varphi_trial <= varphi + eta_phi*alpha*varphi_d)
end

function is_sufficient_progress(theta_trial::T, theta, gamma_theta, varphi_trial, varphi, gamma_phi, has_constraints) where T
    (has_constraints && ((theta_trial<=(1-gamma_theta)*theta+10*eps(T)*abs(theta))) || ((varphi_trial<=varphi-gamma_phi*theta +10*eps(T)*abs(varphi))))
end

function augment_filter!(filter, theta, varphi, gamma_theta)
    push!(filter, ((1-gamma_theta)*theta, varphi-gamma_theta*theta))
end

function is_filter_acceptable(filter, theta, varphi)
    !isnan(theta) || return false
    !isinf(theta) || return false
    !isnan(varphi) || return false
    !isinf(varphi) || return false

    for (theta_F,varphi_F) in filter
        theta <= theta_F || varphi <= varphi_F || return false
    end
    return true
end

function is_barr_obj_rapid_increase(varphi, varphi_trial, obj_max_inc)
    return (varphi_trial >= varphi) && (log10(varphi_trial-varphi) > obj_max_inc + max(1.0, log10(abs(varphi))))
end

function reset_bound_dual!(z, x, mu, kappa_sigma)
    z .= max.(min.(z, (kappa_sigma*mu)./x), (mu/kappa_sigma)./x)
    return
end
function reset_bound_dual!(z, x1, x2, mu, kappa_sigma)
    z .= max.(min.(z, (kappa_sigma*mu)./(x1.-x2)), (mu/kappa_sigma)./(x1.-x2))
    return
end

function get_ftype(filter,theta,theta_trial,varphi,varphi_trial,switching_condition,armijo_condition,
                   theta_min,obj_max_inc,gamma_theta,gamma_phi,has_constraints)
    is_filter_acceptable(filter,theta_trial,varphi_trial) || return " "
    !is_barr_obj_rapid_increase(varphi,varphi_trial,obj_max_inc) || return " "

    if theta <= theta_min && switching_condition
        armijo_condition && return "f"
    else
        is_sufficient_progress(
            theta_trial,theta,gamma_theta,varphi_trial,varphi,gamma_phi,has_constraints) && return "h"
    end

    return " "
end

# fixed variable treatment ----------------------------------------------------
function _get_fixed_variable_index(
    mat::SparseMatrixCSC{Tv,Ti1}, ind_fixed::Vector{Ti2}
) where {Tv,Ti1,Ti2}
    fixed_aug_index = Int[]
    for i in ind_fixed
        append!(fixed_aug_index,append!(collect(mat.colptr[i]+1:mat.colptr[i+1]-1)))
    end
    append!(fixed_aug_index,setdiff!(Base._findin(mat.rowval,ind_fixed),mat.colptr))

    return fixed_aug_index
end

function fixed_variable_treatment_vec!(vec, ind_fixed)
    @inbounds for i in ind_fixed
        vec[i] = 0.0
    end
end

function fixed_variable_treatment_z!(zl, zu, f, jacl, ind_fixed)
    @inbounds @simd for i in ind_fixed
        z = f[i]+jacl[i]
        if z >= 0.0
            zl[i] = z
            zu[i] = 0.0
        else
            zl[i] = 0.0
            zu[i] = -z
        end
    end
end

function dual_inf_perturbation!(px, ind_llb, ind_uub, mu, kappa_d)
    @inbounds @simd for i in ind_llb
        px[i] -= mu*kappa_d
    end
    @inbounds @simd for i in ind_uub
        px[i] += mu*kappa_d
    end
end
