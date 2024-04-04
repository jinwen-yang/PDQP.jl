
function weighted_norm(
    vec::CuVector{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

mutable struct CuSolutionWeightedAverage 
    avg_primal_solutions::CuVector{Float64}
    avg_dual_solutions::CuVector{Float64}
    primal_solutions_count::Int64
    dual_solutions_count::Int64
    avg_primal_product::CuVector{Float64}
    avg_dual_product::CuVector{Float64}
    avg_primal_obj_product::CuVector{Float64} 
end

mutable struct CuBufferAvgState
    avg_primal_solution::CuVector{Float64}
    avg_dual_solution::CuVector{Float64}
    avg_primal_product::CuVector{Float64}
    avg_dual_product::CuVector{Float64} 
    avg_primal_gradient::CuVector{Float64}
    avg_primal_obj_product::CuVector{Float64} 
end

"""
Initialize weighted average
"""
function cu_initialize_solution_weighted_average(
    primal_size::Int64,
    dual_size::Int64,
)
    return CuSolutionWeightedAverage(
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        0,
        0,
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, primal_size),
    )
end

"""
Reset weighted average
"""
function reset_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
)
    solution_weighted_avg.avg_primal_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.avg_primal_solutions))
    solution_weighted_avg.avg_dual_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.avg_dual_solutions))
    solution_weighted_avg.primal_solutions_count = 0
    solution_weighted_avg.dual_solutions_count = 0

    solution_weighted_avg.avg_primal_product .= CUDA.zeros(Float64, length(solution_weighted_avg.avg_dual_solutions))
    solution_weighted_avg.avg_dual_product .= CUDA.zeros(Float64, length(solution_weighted_avg.avg_primal_solutions))
    solution_weighted_avg.avg_primal_obj_product .= CUDA.zeros(Float64, length(solution_weighted_avg.avg_primal_solutions))
    return
end

"""
Update weighted average of primal solution
"""
function add_to_primal_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0
    solution_weighted_avg.avg_primal_solutions .+=
        weight * (current_primal_solution - solution_weighted_avg.avg_primal_solutions)
    solution_weighted_avg.primal_solutions_count += 1
    return
end

"""
Update weighted average of dual solution
"""
function add_to_dual_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.dual_solutions_count >= 0
    solution_weighted_avg.avg_dual_solutions .+= 
        weight * (current_dual_solution - solution_weighted_avg.avg_dual_solutions)
    solution_weighted_avg.dual_solutions_count += 1
    return
end

"""
Update weighted average of primal product
"""
function add_to_primal_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0
    solution_weighted_avg.avg_primal_product .+=
        weight * (current_primal_product - solution_weighted_avg.avg_primal_product)
    return
end

"""
Update weighted average of dual product
"""
function add_to_dual_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.dual_solutions_count >= 0
    solution_weighted_avg.avg_dual_product .+=
        weight * (current_dual_product - solution_weighted_avg.avg_dual_product)
    return
end


"""
Update weighted average of primal objective product
"""
function add_to_primal_obj_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_obj_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.primal_solutions_count >= 0
    solution_weighted_avg.avg_primal_obj_product .+=
        weight * (current_primal_obj_product - solution_weighted_avg.avg_primal_obj_product)
    return
end


"""
Update weighted average
"""
function add_to_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    weight::Float64,
    current_primal_product::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    current_primal_obj_product::CuVector{Float64},
)
    add_to_primal_solution_weighted_average!(
        solution_weighted_avg,
        current_primal_solution,
        weight,
    )
    add_to_dual_solution_weighted_average!(
        solution_weighted_avg,
        current_dual_solution,
        weight,
    )

    add_to_primal_product_weighted_average!(
        solution_weighted_avg,
        current_primal_product,
        weight,
    )
    add_to_dual_product_weighted_average!(
        solution_weighted_avg,
        current_dual_product,
        weight,
    )
    add_to_primal_obj_product_weighted_average!(
        solution_weighted_avg,
        current_primal_obj_product,
        weight,
    )
    return
end

"""
Compute average solutions
"""
function compute_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    buffer_avg::CuBufferAvgState,
    problem::CuQuadraticProgrammingProblem,
)
    buffer_avg.avg_primal_solution .= copy(solution_weighted_avg.avg_primal_solutions)
    buffer_avg.avg_dual_solution .= copy(solution_weighted_avg.avg_dual_solutions)
    buffer_avg.avg_primal_product .= copy(solution_weighted_avg.avg_primal_product)
    buffer_avg.avg_dual_product .= copy(solution_weighted_avg.avg_dual_product)
    buffer_avg.avg_primal_obj_product .= copy(solution_weighted_avg.avg_primal_obj_product)

    buffer_avg.avg_primal_gradient .= problem.objective_vector .- solution_weighted_avg.avg_dual_product
    buffer_avg.avg_primal_gradient .+= buffer_avg.avg_primal_obj_product

end

"""
Compute weighted KKT residual for restarting
"""
function compute_weight_kkt_residual(
    problem::CuQuadraticProgrammingProblem,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64}, 
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64}, 
    buffer_kkt::CuBufferKKTState,
    primal_weight::Float64,
    primal_norm_params::Float64, 
    dual_norm_params::Float64, 
    # scaled_problem::ScaledQpProblem, ##
)
    ## construct buffer_kkt
    buffer_kkt.primal_solution .= copy(primal_iterate)
    buffer_kkt.dual_solution .= copy(dual_iterate)
    buffer_kkt.primal_product .= copy(primal_product)
    buffer_kkt.primal_gradient .= copy(primal_gradient)
    buffer_kkt.primal_obj_product .= copy(primal_obj_product)

    # primal
    compute_primal_residual!(problem, buffer_kkt)
    primal_objective = primal_obj(problem, buffer_kkt.primal_solution, buffer_kkt.primal_obj_product)

    l_inf_primal_residual = norm([buffer_kkt.constraint_violation; buffer_kkt.lower_variable_violation; buffer_kkt.upper_variable_violation], Inf)
    relative_l_inf_primal_residual = l_inf_primal_residual / (1 + max(norm(problem.right_hand_side, Inf), norm(buffer_kkt.primal_product, Inf)))

    # dual
    compute_dual_stats!(problem, buffer_kkt)
    dual_objective = buffer_kkt.dual_stats.dual_objective

    l_inf_dual_residual = norm([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation], Inf)
    relative_l_inf_dual_residual = l_inf_dual_residual / (1 + max(norm(problem.objective_vector, Inf), norm(buffer_kkt.primal_obj_product, Inf), norm(dual_product, Inf)))

    # gap
    gap = abs(primal_objective - dual_objective)
    abs_obj =
        max(abs(primal_objective),
        abs(dual_objective))
    relative_gap = gap / (1 + abs_obj)

    weighted_kkt_residual = max(primal_weight * l_inf_primal_residual, 1/primal_weight * l_inf_dual_residual, abs(primal_objective - dual_objective))

    relative_weighted_kkt_residual = max(primal_weight * relative_l_inf_primal_residual, 1/primal_weight * relative_l_inf_dual_residual, relative_gap)

    return KKTrestart(weighted_kkt_residual, relative_weighted_kkt_residual)
end


mutable struct CuRestartInfo
    """
    The primal_solution recorded at the last restart point.
    """
    primal_solution::CuVector{Float64}
    """
    The dual_solution recorded at the last restart point.
    """
    dual_solution::CuVector{Float64}
    """
    KKT residual at last restart. This has a value of nothing if no restart has occurred.
    """
    last_restart_kkt_residual::Union{Nothing,KKTrestart} 
    """
    The length of the last restart interval.
    """
    last_restart_length::Int64
    """
    The primal distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    primal_distance_moved_last_restart_period::Float64
    """
    The dual distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    dual_distance_moved_last_restart_period::Float64
    """
    Reduction in the potential function that was achieved last time we tried to do a restart.
    """
    kkt_reduction_ratio_last_trial::Float64

    primal_product::CuVector{Float64}
    dual_product::CuVector{Float64} 
    primal_gradient::CuVector{Float64}
    primal_obj_product::CuVector{Float64} 
end

"""
Initialize last restart info
"""
function create_last_restart_info(
    problem::CuQuadraticProgrammingProblem,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64},
)
    return CuRestartInfo(
        copy(primal_solution),
        copy(dual_solution),
        nothing,
        1,
        0.0,
        0.0,
        1.0,
        copy(primal_product),
        copy(dual_product),
        copy(primal_gradient),
        copy(primal_obj_product),
    )
end

"""
Check restart criteria based on weighted KKT
"""
function should_do_adaptive_restart_kkt(
    problem::CuQuadraticProgrammingProblem,
    candidate_kkt::KKTrestart, 
    restart_params::RestartParameters,
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    buffer_kkt::CuBufferKKTState,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
)
    
    last_restart = compute_weight_kkt_residual(
        problem,
        last_restart_info.primal_solution,
        last_restart_info.dual_solution,
        last_restart_info.primal_product,
        last_restart_info.dual_product,
        last_restart_info.primal_gradient,
        last_restart_info.primal_obj_product,
        buffer_kkt,
        primal_weight,
        primal_norm_params,
        dual_norm_params,
    )

    do_restart = false

    kkt_candidate_residual = candidate_kkt.relative_kkt_residual
    kkt_last_residual = last_restart.relative_kkt_residual  
    kkt_reduction_ratio = kkt_candidate_residual / kkt_last_residual

    if kkt_reduction_ratio < restart_params.necessary_reduction_for_restart
        if kkt_reduction_ratio < restart_params.sufficient_reduction_for_restart
            do_restart = true
        elseif kkt_reduction_ratio > last_restart_info.kkt_reduction_ratio_last_trial
            do_restart = true
        end
    end
    last_restart_info.kkt_reduction_ratio_last_trial = kkt_reduction_ratio
  
    return do_restart
end


"""
Check restart
"""
function run_restart_scheme(
    problem::CuQuadraticProgrammingProblem,
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    last_restart_info::CuRestartInfo,
    iterations_completed::Int64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_weight::Float64,
    verbosity::Int64,
    restart_params::RestartParameters,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64},
    buffer_avg::CuBufferAvgState,
    buffer_kkt::CuBufferKKTState,
    buffer_primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64}, 
)
    if solution_weighted_avg.primal_solutions_count > 0 &&
        solution_weighted_avg.dual_solutions_count > 0
    else
        return RESTART_CHOICE_NO_RESTART
    end

    restart_length = solution_weighted_avg.primal_solutions_count
    artificial_restart = false
    do_restart = false
    
    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed
        do_restart = true
        artificial_restart = true
    end

    if restart_params.restart_scheme == NO_RESTARTS
        reset_to_average = false
        candidate_kkt_residual = nothing
    else
        current_kkt_res = compute_weight_kkt_residual(
            problem,
            current_primal_solution,
            current_dual_solution,
            primal_product,
            dual_product, 
            buffer_primal_gradient,
            primal_obj_product,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )
        avg_kkt_res = compute_weight_kkt_residual(
            problem,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            buffer_avg.avg_primal_product,
            buffer_avg.avg_dual_product, 
            buffer_avg.avg_primal_gradient,
            buffer_avg.avg_primal_obj_product,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )

        reset_to_average = should_reset_to_average(
            current_kkt_res,
            avg_kkt_res,
            restart_params.restart_to_current_metric,
        )

        if reset_to_average
            candidate_kkt_residual = avg_kkt_res
        else
            candidate_kkt_residual = current_kkt_res
        end
    end

    if !do_restart
        # Decide if we are going to do a restart.
        if restart_params.restart_scheme == ADAPTIVE_KKT
            do_restart = should_do_adaptive_restart_kkt(
                problem,
                candidate_kkt_residual,
                restart_params,
                last_restart_info,
                primal_weight,
                buffer_kkt,
                primal_norm_params,
                dual_norm_params,
            )
        elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
            restart_params.restart_frequency_if_fixed <= restart_length
            do_restart = true
        end
    end

    if !do_restart
        return RESTART_CHOICE_NO_RESTART
    else
        if reset_to_average
            if verbosity >= 4
                print("  Restarted to average")
            end
            current_primal_solution .= copy(buffer_avg.avg_primal_solution)
            current_dual_solution .= copy(buffer_avg.avg_dual_solution)
            primal_product .= copy(buffer_avg.avg_primal_product)
            primal_obj_product .= copy(buffer_avg.avg_primal_obj_product)
            dual_product .= problem.objective_vector .- buffer_avg.avg_primal_gradient
            dual_product .+= buffer_avg.avg_primal_obj_product
            buffer_primal_gradient .= copy(buffer_avg.avg_primal_gradient)
        else
        # Current point is much better than average point.
            if verbosity >= 4
                print("  Restarted to current")
            end
        end

        if verbosity >= 4
            print(" after ", rpad(restart_length, 4), " iterations")
            if artificial_restart
                println("*")
            else
                println("")
            end
        end
        reset_solution_weighted_average!(solution_weighted_avg)

        update_last_restart_info!(
            last_restart_info,
            current_primal_solution,
            current_dual_solution,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
            candidate_kkt_residual,
            restart_length,
            primal_product,
            dual_product,
            buffer_primal_gradient,
            primal_obj_product,
        )

        if reset_to_average
            return RESTART_CHOICE_RESTART_TO_AVERAGE
        else
            return RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
        end
    end
end

"""
Compute primal weight at restart
"""
function compute_new_primal_weight(
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    primal_weight_update_smoothing::Float64,
    verbosity::Int64,
)
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / primal_distance
        # Exponential moving average.
        # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
        # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)
        if verbosity >= 4
            Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
        end

        return primal_weight
    else
        return primal_weight
    end
end

"""
Update last restart info
"""
function update_last_restart_info!(
    last_restart_info::CuRestartInfo,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    avg_primal_solution::CuVector{Float64},
    avg_dual_solution::CuVector{Float64},
    primal_weight::Float64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    candidate_kkt_residual::Union{Nothing,KKTrestart},
    restart_length::Int64,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64}, 
    primal_gradient::CuVector{Float64},
    primal_obj_product::CuVector{Float64},
)
    last_restart_info.primal_distance_moved_last_restart_period =
        weighted_norm(
            avg_primal_solution - last_restart_info.primal_solution,
            primal_norm_params,
        ) / sqrt(primal_weight)
    last_restart_info.dual_distance_moved_last_restart_period =
        weighted_norm(
            avg_dual_solution - last_restart_info.dual_solution,
            dual_norm_params,
        ) * sqrt(primal_weight)
    last_restart_info.primal_solution .= copy(current_primal_solution)
    last_restart_info.dual_solution .= copy(current_dual_solution)

    last_restart_info.last_restart_length = restart_length
    last_restart_info.last_restart_kkt_residual = candidate_kkt_residual

    last_restart_info.primal_product .= copy(primal_product)
    last_restart_info.dual_product .= copy(dual_product)
    last_restart_info.primal_gradient .= copy(primal_gradient)

    last_restart_info.primal_obj_product .= copy(primal_obj_product)

end

"""
Initialize primal weight
"""
function select_initial_primal_weight(
    problem::CuQuadraticProgrammingProblem,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_importance::Float64,
    verbosity::Int64,
)
    rhs_vec_norm = weighted_norm(problem.right_hand_side, dual_norm_params)
    obj_vec_norm = weighted_norm(problem.objective_vector, primal_norm_params)
    if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
        primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
    else
        primal_weight = primal_importance
    end
    if verbosity >= 6
        println("Initial primal weight = $primal_weight")
    end
    return primal_weight
end

