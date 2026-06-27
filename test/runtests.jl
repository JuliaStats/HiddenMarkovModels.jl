using Aqua: Aqua
using Documenter: Documenter
using HiddenMarkovModels
using JET
using JuliaFormatter: JuliaFormatter
using Literate
using Pkg
using Test

TEST_SUITE = get(ENV, "JULIA_HMM_TEST_SUITE", "Standard")
if TEST_SUITE == "HMMBase"
    Pkg.add("HMMBase")
    using HMMBase: HMMBase
end

Pkg.develop(; path=joinpath(dirname(@__DIR__), "libs", "HMMTest"))

examples_path = joinpath(dirname(@__DIR__), "examples")
examples_script_path = joinpath(@__DIR__, "examples")

for file in readdir(examples_script_path)
    if endswith(file, ".jl")
        rm(joinpath(examples_script_path, file))
    end
end

for file in readdir(examples_path)
    Literate.script(joinpath(examples_path, file), examples_script_path)
end

@testset verbose = true "HiddenMarkovModels.jl" begin
    if TEST_SUITE == "Standard"
        @testset "Code formatting" begin
            if VERSION >= v"1.10"
                @test JuliaFormatter.format(
                    HiddenMarkovModels; verbose=false, overwrite=false
                )
            end
        end

        @testset "Code quality" begin
            Aqua.test_all(
                HiddenMarkovModels; ambiguities=false, deps_compat=(check_extras=false,)
            )
        end

        @testset "Code linting" begin
            using Distributions
            using Zygote
            if VERSION >= v"1.10"
                JET.test_package(HiddenMarkovModels; target_modules=(HiddenMarkovModels,))
            end
        end

        @testset "Distributions" begin
            include("distributions.jl")
        end

        for file in readdir(examples_script_path)
            if endswith(file, ".jl")
                @testset "Example - $file" begin
                    include(joinpath(examples_path, file))
                end
            end
        end

        @testset "Doctests" begin
            Documenter.doctest(HiddenMarkovModels)
        end
    end

    @testset verbose = true "Correctness - $TEST_SUITE" begin
        include("correctness.jl")
    end

    @testset verbose = true "Miscellaneous" begin
        include("misc.jl")
    end

    @testset verbose = true "Duration convention" begin
        include("duration.jl")
    end

    @testset verbose = true "HSMM types and sampling" begin
        include("hsmm.jl")
    end
end
