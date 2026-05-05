using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using DensityInterface
using Distributions
using Random: Random, AbstractRNG
using Test

struct TestControlledEmissionDist end

DensityInterface.DensityKind(::TestControlledEmissionDist) = DensityInterface.HasDensity()
function DensityInterface.logdensityof(::TestControlledEmissionDist, obs, control)
    return -(obs - control)^2
end
Random.rand(::AbstractRNG, ::TestControlledEmissionDist, control) = control

function thrown_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "Allow NaN density" begin
    init = rand_prob_vec(2)
    trans = rand_trans_mat(2)
    dists = [Normal(i, Inf) for i in 1:2]
    hmm = HMM(init, trans, dists)
    obs_seq = rand(5)
    @test isnan(logdensityof(hmm, obs_seq))
end

@testset "ControlledEmissionHMM accessors and control errors" begin
    init = [0.4, 0.6]
    trans = [0.7 0.3; 0.2 0.8]
    dists = [TestControlledEmissionDist(), TestControlledEmissionDist()]
    hmm = ControlledEmissionHMM(init, trans, dists)

    @test initialization(hmm) === init
    @test HMMs.log_initialization(hmm) === hmm.loginit
    @test transition_matrix(hmm) === trans
    @test HMMs.log_transition_matrix(hmm) === hmm.logtrans
    @test transition_matrix(hmm, 1.0) === trans
    @test transition_matrix(hmm, nothing) === trans
    @test HMMs.log_transition_matrix(hmm, 1.0) === hmm.logtrans
    @test HMMs.log_transition_matrix(hmm, nothing) === hmm.logtrans

    controlled_dists = obs_distributions(hmm, 2.0)
    @test controlled_dists isa ControlledEmissions
    @test controlled_dists[1] isa ControlledEmission
    @test logdensityof(controlled_dists[1], 3.0) == -1.0

    err = thrown_error() do
        obs_distributions(hmm, nothing)
    end
    @test err isa ArgumentError
    @test occursin("requires a control value", sprint(showerror, err))

    err = thrown_error() do
        rand(hmm, 3)
    end
    @test err isa MethodError
    @test occursin("requires a control sequence", sprint(showerror, err))
    @test occursin("rand(hmm, control_seq)", sprint(showerror, err))

    err = thrown_error() do
        rand(Random.default_rng(), hmm, 3)
    end
    @test err isa MethodError
    @test occursin("requires a control sequence", sprint(showerror, err))
    @test occursin("rand(hmm, control_seq)", sprint(showerror, err))

    obs_seq = [1.0, 2.0, 3.0]
    control_seq = [1.0, 1.5, 2.0]
    seq_ends = (3,)
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
    err = thrown_error() do
        fit!(hmm, fb_storage, obs_seq; seq_ends)
    end
    @test err isa MethodError
    @test occursin("requires `control_seq`", sprint(showerror, err))
    @test occursin("fit!(hmm, fb_storage, obs_seq, control_seq", sprint(showerror, err))
end
