# struct ConstantStepsizeParams end

# struct PdhgParameters
#     l_inf_ruiz_iterations::Int
#     l2_norm_rescaling::Bool
#     pock_chambolle_alpha::Union{Float64,Nothing}
#     primal_importance::Float64
#     scale_invariant_initial_primal_weight::Bool
#     verbosity::Int64
#     record_iteration_stats::Bool
#     termination_evaluation_frequency::Int32
#     termination_criteria::TerminationCriteria
#     restart_params::RestartParameters
#     step_size_policy_params::Union{
#         AdaptiveStepsizeParams,
#         ConstantStepsizeParams,
#     }
# end

mutable struct CuPdhgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    current_primal_obj_product::CuVector{Float64} #
    solution_weighted_avg::CuSolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
    l2_norm_objective_matrix::Float64
    l2_norm_constraint_matrix::Float64
end


mutable struct CuBufferState
    next_primal::CuVector{Float64}
    next_dual::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    next_primal_product::CuVector{Float64}
    next_dual_product::CuVector{Float64}
    delta_dual_product::CuVector{Float64}
    next_primal_obj_product::CuVector{Float64} #
    delta_primal_obj_product::CuVector{Float64} #
end


# function define_norms(
#     primal_size::Int64,
#     dual_size::Int64,
#     step_size::Float64,
#     primal_weight::Float64,
# )
#     return 1 / step_size * primal_weight, 1 / step_size / primal_weight
# end
  

function pdhg_specific_log(
    # problem::QuadraticProgrammingProblem,
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        # "   %5d inv_step_size=%9g ",
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        norm(current_primal_solution),
        norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

# function pdhg_final_log(
#     problem::QuadraticProgrammingProblem,
#     avg_primal_solution::Vector{Float64},
#     avg_dual_solution::Vector{Float64},
#     verbosity::Int64,
#     iteration::Int64,
#     termination_reason::TerminationReason,
#     last_iteration_stats::IterationStats,
# )

#     if verbosity >= 2
#         println("Avg solution:")
#         Printf.@printf(
#             "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
#             last_iteration_stats.convergence_information[1].l_inf_primal_residual,
#             last_iteration_stats.convergence_information[1].primal_objective,
#             last_iteration_stats.convergence_information[1].l_inf_dual_residual,
#             last_iteration_stats.convergence_information[1].dual_objective
#         )
#         Printf.@printf(
#             "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
#             norm(avg_primal_solution, 1),
#             norm(avg_primal_solution),
#             norm(avg_primal_solution, Inf)
#         )
#         Printf.@printf(
#             "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
#             norm(avg_dual_solution, 1),
#             norm(avg_dual_solution),
#             norm(avg_dual_solution, Inf)
#         )
#     end

#     generic_final_log(
#         problem,
#         avg_primal_solution,
#         avg_dual_solution,
#         last_iteration_stats,
#         verbosity,
#         iteration,
#         termination_reason,
#     )
# end

# function power_method_failure_probability(
#     dimension::Int64,
#     epsilon::Float64,
#     k::Int64,
# )
#     if k < 2 || epsilon <= 0.0
#         return 1.0
#     end
#     return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) * sqrt(dimension) * (1.0 - epsilon)^(k - 1 / 2) # FirstOrderLp.jl old version (epsilon * (k - 1)) instead of sqrt(epsilon * (k - 1)))
# end

# function estimate_maximum_singular_value(
#     matrix::SparseMatrixCSC{Float64,Int64};
#     probability_of_failure = 0.01::Float64,
#     desired_relative_error = 0.1::Float64,
#     seed::Int64 = 1,
# )
#     epsilon = 1.0 - (1.0 - desired_relative_error)^2
#     x = randn(Random.MersenneTwister(seed), size(matrix, 2))

#     number_of_power_iterations = 0
#     while power_method_failure_probability(
#         size(matrix, 2),
#         epsilon,
#         number_of_power_iterations,
#     ) > probability_of_failure
#         x = x / norm(x, 2)
#         x = matrix' * (matrix * x)
#         number_of_power_iterations += 1
#     end
    
#     return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
#     number_of_power_iterations
# end

"""
Kernel to compute primal solution in the next iteration
"""
function compute_next_primal_solution_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    current_primal_obj_product::CuDeviceVector{Float64}, #
    current_avg_primal_obj_product::CuDeviceVector{Float64}, #
    momentum_coefficient::Float64, #
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    next_primal::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            next_primal[tx] = current_primal_solution[tx] - (step_size / primal_weight) * (objective_vector[tx] + (1 - 1 / momentum_coefficient) * current_avg_primal_obj_product[tx] + (1 / momentum_coefficient) * current_primal_obj_product[tx] - current_dual_product[tx])

            next_primal[tx] = min(variable_upper_bound[tx], max(variable_lower_bound[tx], next_primal[tx]))
        end
    end
    return 
end

"""
Compute primal solution in the next iteration
"""
function compute_next_primal_solution!(
    problem::CuQuadraticProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    current_primal_obj_product::CuVector{Float64}, #
    current_avg_primal_obj_product::CuVector{Float64}, #
    momentum_coefficient::Float64, #
    step_size::Float64,
    primal_weight::Float64,
    next_primal::CuVector{Float64},
    next_primal_product::CuVector{Float64},
    next_primal_obj_product::CuVector{Float64}, #
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)
    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_next_primal_solution_kernel!(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        current_primal_solution,
        current_dual_product,
        current_primal_obj_product,
        current_avg_primal_obj_product,
        momentum_coefficient,
        step_size,
        primal_weight,
        problem.num_variables,
        next_primal,
    )

    # next_primal_product .= problem.constraint_matrix * next_primal
    # next_primal_obj_product .= problem.objective_matrix * next_primal

    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, next_primal, 0, next_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    CUDA.CUSPARSE.mv!('N', 1, problem.objective_matrix, next_primal, 0, next_primal_obj_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2) 
end

"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    next_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    extrapolation_coefficient::Float64, #
    num_equalities::Int64,
    num_constraints::Int64,
    next_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx] - extrapolation_coefficient * (next_primal_product[tx] - current_primal_product[tx]))
        end
    elseif (num_equalities + 1) <= tx <= num_constraints
        @inbounds begin
            next_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - next_primal_product[tx] - extrapolation_coefficient * (next_primal_product[tx] - current_primal_product[tx]))
            next_dual[tx] = max(next_dual[tx], 0.0)
        end
    end
    return 
end

"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
    problem::CuQuadraticProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    extrapolation_coefficient::Float64, #
    step_size::Float64,
    primal_weight::Float64,
    next_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    next_dual::CuVector{Float64},
    next_dual_product::CuVector{Float64},
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        next_primal_product,
        step_size,
        primal_weight,
        extrapolation_coefficient,
        problem.num_equalities,
        problem.num_constraints,
        next_dual,
    )

    # next_dual_product .= problem.constraint_matrix_t * next_dual
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    solver_state.delta_primal .= buffer_state.next_primal .- solver_state.current_primal_solution
    solver_state.delta_dual .= buffer_state.next_dual .- solver_state.current_dual_solution
    # solver_state.delta_dual_product .= buffer_state.next_dual_product .- solver_state.current_dual_product
    solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    solver_state.current_dual_solution .= copy(buffer_state.next_dual)
    solver_state.current_dual_product .= copy(buffer_state.next_dual_product)
    solver_state.current_primal_product .= copy(buffer_state.next_primal_product)
    solver_state.current_primal_obj_product .= copy(buffer_state.next_primal_obj_product)

    weight = 1 / (1.0 + solver_state.solution_weighted_avg.primal_solutions_count / 2.0) #
    
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        solver_state.current_primal_obj_product,
    )
end

"""
Take PDHG step with ConstantStepsize
"""
function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuQuadraticProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)

    momentum_coefficient = 1.0 + solver_state.solution_weighted_avg.primal_solutions_count / 2.0

    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        solver_state.current_primal_obj_product,
        solver_state.solution_weighted_avg.avg_primal_obj_product,
        momentum_coefficient,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_primal,
        buffer_state.next_primal_product,
        buffer_state.next_primal_obj_product,
    )
    
    extrapolation_coefficient = (solver_state.solution_weighted_avg.dual_solutions_count) / (solver_state.solution_weighted_avg.dual_solutions_count + 1.0)

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        extrapolation_coefficient,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_primal_product,
        solver_state.current_primal_product,
        buffer_state.next_dual,
        buffer_state.next_dual_product,
    )

    solver_state.cumulative_kkt_passes += 1

    update_solution_in_solver_state!(
        solver_state, 
        buffer_state,
    )

    iter = solver_state.solution_weighted_avg.primal_solutions_count + 2
    norm_Q, norm_A = solver_state.l2_norm_objective_matrix, solver_state.l2_norm_constraint_matrix
    primal_weight = solver_state.primal_weight

    # solver_state.step_size = sqrt((iter / norm_A) / (2 * norm_Q + iter * norm_A))
    # solver_state.primal_weight = sqrt(((2 * norm_Q + iter * norm_A) / norm_A) / iter)
    # solver_state.step_size = 1.99 * iter / (norm_Q / primal_weight + sqrt(4*norm_A^2*iter^2 + norm_Q^2 / primal_weight^2))

    solver_state.step_size = min(0.99 * iter / (norm_Q / primal_weight + sqrt(norm_A^2*iter^2 + norm_Q^2 / primal_weight^2)), (iter-1)/(iter-2) * solver_state.step_size)
end


"""
Main algorithm
"""
function optimize_gpu(
    params::PdhgParameters,
    original_problem::QuadraticProgrammingProblem,
)
    validate(original_problem)
    qp_cache = cached_quadratic_program_info(original_problem)
    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )

    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    d_problem = d_scaled_problem.scaled_qp
    buffer_lp = qp_cpu_to_gpu(original_problem)

    norm_Q, number_of_power_iterations_Q = estimate_maximum_singular_value(scaled_problem.scaled_qp.objective_matrix)
    norm_A, number_of_power_iterations_A = estimate_maximum_singular_value(scaled_problem.scaled_qp.constraint_matrix)

    # initialization
    solver_state = CuPdhgSolverState(
        CUDA.zeros(Float64, primal_size),    # current_primal_solution
        CUDA.zeros(Float64, dual_size),      # current_dual_solution
        CUDA.zeros(Float64, primal_size),    # delta_primal
        CUDA.zeros(Float64, dual_size),      # delta_dual
        CUDA.zeros(Float64, dual_size),      # current_primal_product
        CUDA.zeros(Float64, primal_size),    # current_dual_product
        CUDA.zeros(Float64, primal_size),    # current_primal_obj_product
        cu_initialize_solution_weighted_average(primal_size, dual_size),
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        nothing,
        nothing,
        norm_Q,
        norm_A,
    )

    buffer_state = CuBufferState(
        CUDA.zeros(Float64, primal_size),      # next_primal
        CUDA.zeros(Float64, dual_size),        # next_dual
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # next_primal_product
        CUDA.zeros(Float64, primal_size),      # next_dual_product
        CUDA.zeros(Float64, primal_size),      # delta_dual_product
        CUDA.zeros(Float64, primal_size),      # next_primal_obj_product
        CUDA.zeros(Float64, primal_size),      # delta_next_primal_obj_product
    )

    buffer_avg = CuBufferAvgState(
        CUDA.zeros(Float64, primal_size),      # avg_primal_solution
        CUDA.zeros(Float64, dual_size),        # avg_dual_solution
        CUDA.zeros(Float64, dual_size),        # avg_primal_product
        CUDA.zeros(Float64, primal_size),      # avg_dual_product
        CUDA.zeros(Float64, primal_size),      # avg_primal_gradient
        CUDA.zeros(Float64, primal_size),      # avg_primal_obj_product
    )

    buffer_original = CuBufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # dual_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # primal_obj_product
    )

    buffer_kkt = CuBufferKKTState(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # primal_obj_product
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
    buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
    buffer_primal_gradient .+= solver_state.current_primal_obj_product

    # stepsize
    if params.step_size_policy_params isa AdaptiveStepsizeParams # TODO: decide initial stepsize
        solver_state.cumulative_kkt_passes += 0.5
        solver_state.step_size = 1.0 / (norm(scaled_problem.scaled_qp.objective_matrix, Inf)+norm(scaled_problem.scaled_qp.constraint_matrix, Inf)) #

        if params.scale_invariant_initial_primal_weight
            solver_state.primal_weight = select_initial_primal_weight(
                scaled_problem.scaled_qp,
                1.0,
                1.0,
                params.primal_importance,
                params.verbosity,
            )
        else
            solver_state.primal_weight = params.primal_importance
        end

    else
        # norm_Q, norm_A = solver_state.l2_norm_objective_matrix, solver_state.l2_norm_constraint_matrix
        solver_state.cumulative_kkt_passes += number_of_power_iterations_Q + number_of_power_iterations_A

        # solver_state.step_size = sqrt(1 / norm_A / (2 * norm_Q + norm_A))
        # solver_state.primal_weight = sqrt((2 * norm_Q + norm_A) / norm_A)
        if params.scale_invariant_initial_primal_weight
            solver_state.primal_weight = select_initial_primal_weight(
                scaled_problem.scaled_qp,
                1.0,
                1.0,
                params.primal_importance,
                params.verbosity,
            )
        else
            solver_state.primal_weight = params.primal_importance
        end
        # solver_state.step_size = 1.99 * 2 / (norm_Q / solver_state.primal_weight + sqrt(4*norm_A^2*4 + norm_Q^2 / solver_state.primal_weight^2))
        solver_state.step_size = 0.99 * 2 / (norm_Q / solver_state.primal_weight + sqrt(4*norm_A^2 + norm_Q^2 / solver_state.primal_weight^2))
    end

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
        buffer_primal_gradient,
        solver_state.current_primal_obj_product,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading()

    

    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.primal_solutions_count == 0 || solver_state.solution_weighted_avg.dual_solutions_count == 0
                buffer_avg.avg_primal_solution .= copy(solver_state.current_primal_solution)
                buffer_avg.avg_dual_solution .= copy(solver_state.current_dual_solution)
                buffer_avg.avg_primal_product .= copy(solver_state.current_primal_product)
                buffer_avg.avg_dual_product .= copy(solver_state.current_dual_product)
                buffer_avg.avg_primal_gradient .= copy(buffer_primal_gradient)
                buffer_avg.avg_primal_obj_product .= copy(solver_state.current_primal_obj_product) 
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            ### KKT ###
            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_dual_product,
                buffer_avg.avg_primal_gradient,
                buffer_avg.avg_primal_obj_product,
                buffer_original,
                buffer_kkt,
            )
            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                
                avg_primal_solution = zeros(primal_size)
                avg_dual_solution = zeros(dual_size)
                gpu_to_cpu!(
                    buffer_avg.avg_primal_solution,
                    buffer_avg.avg_dual_solution,
                    avg_primal_solution,
                    avg_dual_solution,
                )

                pdhg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    iteration,
                    termination_reason,
                    current_iteration_stats,
                )

                return unscaled_saddle_point_output(
                    scaled_problem,
                    avg_primal_solution,
                    avg_dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                )
            end

            buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product
            buffer_primal_gradient .+= solver_state.current_primal_obj_product

            current_iteration_stats.restart_used = run_restart_scheme(
                d_scaled_problem.scaled_qp,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
                solver_state.current_primal_obj_product,
            )

            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                if params.step_size_policy_params isa AdaptiveStepsizeParams 
                    solver_state.primal_weight = compute_new_primal_weight(
                        last_restart_info,
                        solver_state.primal_weight,
                        primal_weight_update_smoothing,
                        params.verbosity,
                    )
                    solver_state.ratio_step_sizes = 1.0
                else
                    # solver_state.step_size = sqrt(1 / norm_A / (2 * norm_Q + norm_A))
                    # solver_state.primal_weight = sqrt((2 * norm_Q + norm_A) / norm_A)
                    solver_state.primal_weight = compute_new_primal_weight(
                        last_restart_info,
                        solver_state.primal_weight,
                        primal_weight_update_smoothing,
                        params.verbosity,
                    )

                    # scale_norm = min(norm_Q/norm_A, norm_A/norm_Q)

                    # if min(solver_state.primal_weight, 1 / solver_state.primal_weight) < scale_norm * 5e-4
                    #     primal_weight_update_smoothing = 0.0
                    #     solver_state.primal_weight = 1.0
                    # end

                    # solver_state.step_size = 1.99 * 2 / (norm_Q / solver_state.primal_weight + sqrt(4*norm_A^2*4 + norm_Q^2 / solver_state.primal_weight^2))
                    solver_state.step_size = 0.99 * 2 / (norm_Q / solver_state.primal_weight + sqrt(4*norm_A^2 + norm_Q^2 / solver_state.primal_weight^2))
                end
            end
        end

        time_spent_doing_basic_algorithm_checkpoint = time()
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                # problem,
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end

        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end