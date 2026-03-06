using QuantumGateDesign, Test, Random, LinearAlgebra

@testset "Infidelity" begin
    @testset "Target equals state, state vectors have length 1." begin
        N = 10
        N_initial_conditions = 4

        target = rand(MersenneTwister(0), N, N_initial_conditions)
        # Normalize to length-1
        for i in 1:N_initial_conditions
            target[:,i] ./= norm(target[:,i])
        end

        state = target
        obj = Infidelity(target)

        @testset "Value is zero." begin
            @test value(obj, state) ≈ 0 atol=1e-15
        end

        @testset "Gradient is non-negative." begin
            # When the state and target are aligned, can decrease infidelity
            # further by scaling the state
            grad = similar(state)
            state_gradient!(grad, obj, state)
            @test all(x -> x >= 0, grad)
        end

        @testset "Scaling state makes objective value negative." begin
            @test value(obj, state .* 2) < 0
        end


        # Note: gradient is not necessarily zero, since 
    end

    @testset "State and target vectors are orthogonal." begin
        state = [1; 0;; 0; 1]
        target = [0; 1;; 1; 0]
        obj = Infidelity(target)
        @testset "Value is one" begin
            @test value(obj, state) ≈ 1 atol=1e-15
        end
    end

    @testset "Manufactured solution" begin
    end
end


@testset "Generalized Infidelity" begin
    @testset "Target and state are aligned, target is length-1" begin
        N = 10
        N_initial_conditions = 4

        target = rand(MersenneTwister(0), N, N_initial_conditions)
        # Normalize to length-1
        for i in 1:N_initial_conditions
            target[:,i] ./= norm(target[:,i])
        end 

        obj = GeneralizedInfidelity(target)

        @testset "Value equals 0 when state equals target." begin
            state = target
            @test value(obj, state) ≈ 0 atol=1e-15
            grad = similar(state)
            @show state_gradient!(grad, obj, state)
        end


        @testset "Value > 0 when state is aligned with target (but not normalized)" begin
            # When norm term in the generlized factor grows faster than the overlap term when scaling the state. So the generalized infidelity should grow positive.
            state = copy(target)
            scale_factors = rand(MersenneTwister(1), size(state, 2))
            for i in axes(state, 2)
                state[:,i] .*= scale_factors[i]
            end
            @test value(obj, state) > 0
        end
    end

    @testset "Genralized Infidelity equals regular Infidelity when state and target are length-1." begin
        N = 10
        N_initial_conditions = 4

        target = rand(MersenneTwister(0), N, N_initial_conditions)
        # Normalize to length-1
        for i in 1:N_initial_conditions
            target[:,i] ./= norm(target[:,i])
        end 

        obj_gen = GeneralizedInfidelity(target)
        obj_inf = Infidelity(target)

        state = rand(MersenneTwister(0), N, N_initial_conditions)
        # Normalize to length-1
        for i in 1:N_initial_conditions
            state[:,i] ./= norm(state[:,i])
        end 

        @test value(obj_gen, state) ≈ value(obj_inf, state) atol=1e-15
    end

    @testset "Manufactured solution" begin
    end
end
