using GeometricIntegrators
using GeometricIntegrators.GeometricBase.Utils: @big

"""
When used with a symmetric, second-order method, produces a sixth-order method.

Coefficients copied from Geometric Numerical Integration, V.3.2 Symmetric
Composition of Symmetric Methods
"""
struct Order6Stages9<: AbstractSplittingMethod end

GeometricBase.order(::Union{Order6Stages9, Type{Order6Stages9}}) = 6

function coefficients(::SuzukiFractal, ::Type{T}=Float64) where {T}
    a = Array{T}([
        (@big 0.39216144400731413927925056),
        (@big 0.33259913678935943859974864),
        (@big -0.70624617255763935980996482),
        (@big 0.08221359629355080023149045),
        (@big 0.79854399093482996339895035),
    ])
    SplittingCoefficientsSS(:Order6Stages9, 6, a)
end

"""
When used with a symmetric, second-order method, produces an eigth-order method.
"""
struct Order8Stages17<: AbstractSplittingMethod end

GeometricBase.order(::Union{Order8Stages17, Type{Order8Stages17}}) = 8

function coefficients(::SuzukiFractal, ::Type{T}=Float64) where {T}
    a = Array{T}([
        (@big 0.13020248308889008087881763),
        (@big 0.56116298177510838456196441),
        (@big -0.38947496264484728640807860),
        (@big 0.15884190655515560089621075),
        (@big -0.39590389413323757733623154),
        (@big 0.18453964097831570709183254),
        (@big 0.25837438768632204729397911),
        (@big 0.29501172360931029887096624),
        (@big -0.60550853383003451169892108),
    ])
    SplittingCoefficientsSS(:Order8Stages17, 8, a)
end

"""
When used with a symmetric, second-order method, produces an tenth-order method.
"""
struct Order10Stages35<: AbstractSplittingMethod end

GeometricBase.order(::Union{Order10Stages35, Type{Order10Stages35}}) = 8

function coefficients(::SuzukiFractal, ::Type{T}=Float64) where {T}
    a = Array{T}([
        (@big 0.07879572252168641926390768),
        (@big 0.31309610341510852776481247),
        (@big 0.02791838323507806610952027),
        (@big -0.22959284159390709415121340),
        (@big 0.13096206107716486317465686),
        (@big -0.26973340565451071434460973),
        (@big 0.07497334315589143566613711),
        (@big 0.11199342399981020488957508),
        (@big 0.36613344954622675119314812),
        (@big -0.39910563013603589787862981),
        (@big 0.10308739852747107731580277),
        (@big 0.41143087395589023782070412),
        (@big -0.00486636058313526176219566),
        (@big -0.39203335370863990644808194),
        (@big 0.05194250296244964703718290),
        (@big 0.05066509075992449633587434),
        (@big 0.04967437063972987905456880),
        (@big 0.04931773575959453791768001),
    ])
    SplittingCoefficientsSS(:Order10Stages35, 10, a)
end
