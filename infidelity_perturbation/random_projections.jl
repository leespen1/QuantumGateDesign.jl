function randunit(n::Integer, type=ComplexF64)
    # Using real instead of complex here made the 1/√N factor disappear. Interesting.
    vector = randn(type, n) 
    vector ./= norm(vector)
    return vector
end

Nsamples = 1_000
inner_products = 

