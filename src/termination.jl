
@enum OptimalityNorm L_INF L2

"A description of solver termination criteria."
mutable struct TerminationCriteria
    "The norm that we are measuring the optimality criteria in."
    optimality_norm::OptimalityNorm

    # Let p correspond to the norm we are using as specified by optimality_norm.
    # If the algorithm terminates with termination_reason =
    # TERMINATION_REASON_OPTIMAL then the following hold:
    # | primal_objective - dual_objective | <= eps_optimal_absolute +
    #  eps_optimal_relative * ( | primal_objective | + | dual_objective | )
    # norm(primal_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
    #  norm(right_hand_side, p)
    # norm(dual_residual, p) <= eps_optimal_absolute + eps_optimal_relative *
    #   norm(objective_vector, p)
    # It is possible to prove that a solution satisfying the above conditions
    # also satisfies SCS's optimality conditions (see link above) with ϵ_pri =
    # ϵ_dual = ϵ_gap = eps_optimal_absolute = eps_optimal_relative. (ϵ_pri,
    # ϵ_dual, and ϵ_gap are SCS's parameters).

    """
    Absolute tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_absolute::Float64

    """
    Relative tolerance on the duality gap, primal feasibility, and dual
    feasibility.
    """
    eps_optimal_relative::Float64

    """
    If termination_reason = TERMINATION_REASON_TIME_LIMIT then the solver has
    taken at least time_sec_limit time.
    """
    time_sec_limit::Float64

    """
    If termination_reason = TERMINATION_REASON_ITERATION_LIMIT then the solver has taken at least iterations_limit iterations.
    """
    iteration_limit::Int32

    """
    If termination_reason = TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT then
    cumulative_kkt_matrix_passes is at least kkt_pass_limit.
    """
    kkt_matrix_pass_limit::Float64
end

function construct_termination_criteria(;
    optimality_norm = L_INF,
    eps_optimal_absolute = 1.0e-3,
    eps_optimal_relative = 1.0e-3,
    time_sec_limit = Inf,
    iteration_limit = typemax(Int32),
    kkt_matrix_pass_limit = Inf,
)
    return TerminationCriteria(
        optimality_norm,
        eps_optimal_absolute,
        eps_optimal_relative,
        time_sec_limit,
        iteration_limit,
        kkt_matrix_pass_limit,
    )
end

function validate_termination_criteria(criteria::TerminationCriteria)
    if criteria.time_sec_limit <= 0
        error("time_sec_limit must be positive")
    end
    if criteria.iteration_limit <= 0
        error("iteration_limit must be positive")
    end
    if criteria.kkt_matrix_pass_limit <= 0
        error("kkt_matrix_pass_limit must be positive")
    end
end

"""
Information about the quadratic program that is used in the termination
criteria. We store it in this struct so we don't have to recompute it.
"""
struct CachedQuadraticProgramInfo
    l_inf_norm_primal_linear_objective::Float64
    l_inf_norm_primal_right_hand_side::Float64
    l2_norm_primal_linear_objective::Float64
    l2_norm_primal_right_hand_side::Float64
end

function cached_quadratic_program_info(qp::QuadraticProgrammingProblem)
    return CachedQuadraticProgramInfo(
        norm(qp.objective_vector, Inf),
        norm(qp.right_hand_side, Inf),
        norm(qp.objective_vector, 2),
        norm(qp.right_hand_side, 2),
    )
end

"""
Check if the algorithm should terminate declaring the optimal solution is found.
"""
function optimality_criteria_met(
    optimality_norm::OptimalityNorm,
    abs_tol::Float64,
    rel_tol::Float64,
    convergence_information::ConvergenceInformation,
    qp_cache::CachedQuadraticProgramInfo,
)
    ci = convergence_information
    gap = ci.relative_optimality_gap

    if optimality_norm == L_INF
        primal_err = ci.relative_l_inf_primal_residual
        dual_err = ci.relative_l_inf_dual_residual
    elseif optimality_norm == L2
        primal_err = ci.relative_l2_primal_residual
        dual_err = ci.relative_l2_dual_residual
    else
        error("Unknown optimality_norm")
    end

    return dual_err < rel_tol &&
            primal_err < rel_tol &&
            gap < rel_tol
end

"""
Checks if the given iteration_stats satisfy the termination criteria. Returns
a TerminationReason if so, and false otherwise.
"""
function check_termination_criteria(
    criteria::TerminationCriteria,
    qp_cache::CachedQuadraticProgramInfo,
    iteration_stats::IterationStats,
)
    for convergence_information in iteration_stats.convergence_information
        if optimality_criteria_met(
            criteria.optimality_norm,
            criteria.eps_optimal_absolute,
            criteria.eps_optimal_relative,
            convergence_information,
            qp_cache,
        )
        return TERMINATION_REASON_OPTIMAL
        end
    end
    if iteration_stats.iteration_number >= criteria.iteration_limit
        return TERMINATION_REASON_ITERATION_LIMIT
    elseif iteration_stats.cumulative_kkt_matrix_passes >=
            criteria.kkt_matrix_pass_limit
        return TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT
    elseif iteration_stats.cumulative_time_sec >= criteria.time_sec_limit
        return TERMINATION_REASON_TIME_LIMIT
    else
        return false # Don't terminate.
    end
end

function termination_reason_to_string(termination_reason::TerminationReason)
    # Strip TERMINATION_REASON_ prefix.
    return string(termination_reason)[20:end]
end