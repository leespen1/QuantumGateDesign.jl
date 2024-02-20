using LinearMaps
using IterativeSolvers

#=
The problem being solved is 

y' = -2t*y
y(t) = y₀*exp(-t²)

The scalar equation is done for each element of y, so you can solve the initial
value problem for multiple initial conditions.
=#

function apply_A!(dydt, y_in, t)
    dydt .= (-2*t) .* y_in
    return nothing
end

function solver(y0, dt, nsteps)
    y = copy(y0)
    dydt = zeros(length(y))

    t = 0.0

    right_hand_side = zeros(length(y))

    function compute_left_hand_side!(out_vec, y_in)
        apply_A!(dydt, y_in, t)
        out_vec .= y_in
        out_vec .-= 0.5*dt .* dydt
        return nothing
    end

    LHS_map = LinearMaps.LinearMap(
        compute_left_hand_side!,
        length(y), length(y),
        ismutating=true
    )

    for n in 0:(nsteps-1)
        t = n*dt
        apply_A!(dydt, y, t)
        right_hand_side .= y
        right_hand_side .+= 0.5*dt .* dydt

        #display(y)

        t = (n+1)*dt
        gmres!(y, LHS_map, right_hand_side)
    end

    return y
end

# Callable struct version
mutable struct Astruct
    t::Float64
    dt::Float64
    dydt::Vector{Float64}
    function Astruct(t::Float64, dt::Float64, N::Int64)
        dydt = zeros(N)
        new(t, dt, dydt)
    end
end

function (a_struct::Astruct)(out_vec, y_in)
        apply_A!(a_struct.dydt, y_in, a_struct.t)
        out_vec .= y_in
        out_vec .-= 0.5*a_struct.dt .* a_struct.dydt
        return nothing
end

function struct_solver(y0, dt, nsteps)
    y = copy(y0)
    dydt = zeros(length(y))

    right_hand_side = zeros(length(y))

    t0 = 0.0
    a_struct = Astruct(t0, dt, length(y))


    LHS_map = LinearMaps.LinearMap(
        a_struct,
        length(y), length(y),
        ismutating=true
    )

    for n in 0:(nsteps-1)
        t = n*dt
        apply_A!(dydt, y, t)
        right_hand_side .= y
        right_hand_side .+= 0.5*dt .* dydt

        a_struct.t = (n+1)*dt
        gmres!(y, LHS_map, right_hand_side)
    end

    return y
end


function ref_solver(y0, dt, nsteps)
    y = copy(y0)
    dydt = Ref(zeros(length(y)))
    local_dt = Ref(dt)

    t = Ref(0.0)

    right_hand_side = zeros(length(y))

    compute_left_hand_side! = let t=t, dydt = dydt, local_dt = local_dt # convert to locals to help the compiler, probably unnecessary though
      function compute_left_hand_side!(out_vec, y_in)
          apply_A!(dydt[], y_in, t[])
          out_vec .= y_in
          dydt[] .-= 0.5*dt[] .* dydt[] 
        return nothing
      end
    end


    LHS_map = LinearMaps.LinearMap(
        compute_left_hand_side!,
        length(y), length(y),
        ismutating=true
    )

    for n in 0:(nsteps-1)
        t[] = n*dt
        apply_A!(dydt[], y, t[])
        right_hand_side .= y
        right_hand_side .+= 0.5*dt .* dydt[]

        t[] = (n+1)*local_dt[]
        gmres!(y, LHS_map, right_hand_side)
    end

    return y
end

function ref_solver_no_local(y0, dt, nsteps)
    y = copy(y0)
    dydt = Ref(zeros(length(y)))
    local_dt = Ref(dt)

    t = Ref(0.0)

    right_hand_side = zeros(length(y))

    function compute_left_hand_side!(out_vec, y_in)
        apply_A!(dydt[], y_in, t[])
        out_vec .= y_in
        dydt[] .-= 0.5*dt[] .* dydt[] 
      return nothing
    end


    LHS_map = LinearMaps.LinearMap(
        compute_left_hand_side!,
        length(y), length(y),
        ismutating=true
    )

    for n in 0:(nsteps-1)
        t[] = n*dt
        apply_A!(dydt[], y, t[])
        right_hand_side .= y
        right_hand_side .+= 0.5*dt .* dydt[]

        t[] = (n+1)*local_dt[]
        gmres!(y, LHS_map, right_hand_side)
    end

    return y
end
