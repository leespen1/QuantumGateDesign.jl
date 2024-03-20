using Plots
using LaTeXStrings

# Set up polynomials on interval [-1, 1]
l_10(x) = 0.25*(1+x)^2 * (2-x)
l_11(x) = -0.25*(1+x)^2 * (1-x)
l_n10(x) = l_10(-x)
l_n11(x) = -l_11(-x)

xs = LinRange(-1, 1, 1001)
lw = 2

x_lims = [-1.1, 1.1]
y_lims = [-0.5, 1.1]
pl = plot(xlabel="x", aspectratio=:equal, legend=:outerright, xlims=x_lims, ylims=y_lims)

# Plot tangent lines
plot!(pl, [-1, -0.5], [0, 0.5], label="", lw=lw, linestyle=:dash, color=:gray, alpha=0.75)
plot!(pl, [-1, -0.5], [1, 1], label="", lw=lw,   linestyle=:dash, color=:gray, alpha=0.75)
plot!(pl, [0.5, 1], [1, 1], label="", lw=lw,     linestyle=:dash, color=:gray, alpha=0.75)
plot!(pl, [0.5, 1], [-0.5, 0], label="", lw=lw,  linestyle=:dash, color=:gray, alpha=0.75)

plot!(pl, xs, l_n10.(xs), label=L"\ell_{-1,0}(x)", lw=lw, color=1)
plot!(pl, xs, l_n11.(xs), label=L"\ell_{-1,1}(x)", lw=lw, color=2)
plot!(pl, xs, l_10.(xs), label=L"\ell_{1,0}(x)",   lw=lw, color=3)
plot!(pl, xs, l_11.(xs), label=L"\ell_{1,1}(x)",   lw=lw, color=4)


#yticks = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
xticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
plot!(pl, xticks=xticks, yticks=yticks)

