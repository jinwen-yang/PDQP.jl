# PDQP.jl

This repository contains experimental code for solving convex quadratic programming using first-order methods on CPUs or NVIDIA GPUs. 

## Setup

A one-time step is required to set up the necessary packages on the local machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running 

`solve.jl` is the recommended script for using PDQP. The results are written to JSON and text files. All commands below assume that the current directory is the working directory.

```shell
$ julia --project scripts/solve.jl \
--instance_path=INSTANCE_PATH --output_directory=OUTPUT_DIRECTORY \ 
--tolerance=TOLERANCE --time_sec_limit=TIME_SEC_LIMIT --use_gpu=USE_GPU
```

## Interpreting the output

A table of iteration stats will be printed with the following headings.

##### runtime

`#iter` = the current iteration number.

`#kkt` = the cumulative number of times the KKT matrix is multiplied.

`seconds` = the cumulative solve time in seconds.

##### residuals

`pr norm` = the Euclidean norm of primal residuals (i.e., the constraint
violation).

`du norm` = the Euclidean norm of the dual residuals.

`gap` = the gap between the primal and dual objective.

##### solution information

`pr obj` = the primal objective value.

`pr norm` = the Euclidean norm of the primal variable vector.

`du norm` = the Euclidean norm of the dual variable vector.

##### relative residuals

`rel pr` = the Euclidean norm of the primal residuals, relative to the
right-hand-side.

`rel dul` = the Euclidean norm of the dual residuals, relative to the primal
linear objective.

`rel gap` = the relative optimality gap.

# License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
