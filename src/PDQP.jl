module PDQP

import LinearAlgebra
import Logging
import Printf
import SparseArrays
import Random

import GZip
import QPSReader
import Statistics
import StatsBase
import StructTypes

const Diagonal = LinearAlgebra.Diagonal
const diag = LinearAlgebra.diag
const dot = LinearAlgebra.dot
const norm = LinearAlgebra.norm
const mean = Statistics.mean
const median = Statistics.median
const nzrange = SparseArrays.nzrange
const nnz = SparseArrays.nnz
const nonzeros = SparseArrays.nonzeros
const rowvals = SparseArrays.rowvals
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spdiagm = SparseArrays.spdiagm
const spzeros = SparseArrays.spzeros
const quantile = Statistics.quantile
const sample = StatsBase.sample

const ThreadPerBlock = 128


include("quadratic_programming.jl")
include("solve_log.jl")
include("quadratic_programming_io.jl")
include("preprocess.jl")
include("termination.jl")
include("iteration_stats_utils.jl")
include("saddle_point.jl")
include("solver.jl")

using CUDA
include("cpu_to_gpu.jl")
include("iteration_stats_utils_gpu.jl")
include("saddle_point_gpu.jl")
include("solver_gpu.jl")

end # module PDQP
