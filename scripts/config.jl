using Pkg, Distributed, DelimitedFiles

const NP = ARGS[1]
const SOLVER = ARGS[2]
const VERBOSE = ARGS[3] == "true"
const QUICK = ARGS[4] == "true"
const GCOFF = ARGS[5] == "true"
const DECODE = ARGS[6] == "true"
const TEST = ARGS[7] == "true"

# Set verbose option
if SOLVER == "ipopt"
    const PRINT_LEVEL = VERBOSE ? 5 : 0
elseif SOLVER == "knitro"
    const PRINT_LEVEL = VERBOSE ? 3 : 0
else
    using MadNLP
    const PRINT_LEVEL = VERBOSE ? MadNLP.TRACE : MadNLP.ERROR
end

addprocs(parse(Int, NP))
