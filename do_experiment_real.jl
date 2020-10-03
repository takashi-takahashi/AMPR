# using Revise
include("./src/AMPR.jl")
using .AMPR
# using AMPR
using LinearAlgebra, Random
using Distributions, GLMNet
using Plots, LaTeXStrings, Colors
using CSV


Random.seed!(0);

### settings ###
intercept = false;

### load data ###
# --- supernova data ---
@time y_raw = CSV.read(
        joinpath(@__DIR__, "real_data/y.csv"),
    header=false);
y = y_raw.Column1;

@time A_raw = CSV.read(
        joinpath(@__DIR__, "real_data/A.csv"),
    header=false);
A_real = Array(A_raw);
m, n = size(A_real);  # ここでの値は一時的にしか使わない

A = A .- mean(A, dims=1);
A = A ./ std(A, dims=1);
y = y .- mean(y);

m, n = size(A);

### cross validation ###
@time cv_result = glmnetcv(
    A, y, Normal(), 
    intercept=false, standardize=false, 
    tol=1e-5, maxit=10000, nfolds=10, nlambda=100
)

cv_index = argmin(cv_result.meanloss);
λ = cv_result.lambda[1:cv_index] .* m;
λ_cv = [cv_result.lambda[cv_index]] .* m;
println("$cv_index, $(length(cv_result.lambda))");


### naive experiment ###
function do_SS(A, y, n_B, μ_B, λ; tol=1.0e-10, randomize=true, intercept=false)
    λ₀ = λ

    active_array = zeros((n_B, n, length(λ)));
    first_moment_array = zeros((n_B, n, length(λ)));
    intercept_first_moment_array = zeros((n_B, length(λ)));

    Threads.@threads for b in 1:n_B
        if randomize
            penalty_factor = rand(Binomial(1, 0.5), n) .+ 1.0
        else
            penalty_factor = ones(n)
        end
        λ = λ₀ .* (sum(penalty_factor)/n)

        sample_index = sample(1:m, Int(round(μ_B * m)))
        y_b = y[sample_index]
        A_b = A[sample_index, :]
        glmnet_result = glmnet(
            A_b, y_b, Normal(), lambda=λ, 
            intercept=intercept, standardize=false,
            tol=tol, maxit=100000, 
            penalty_factor=penalty_factor,
        )
        if !intercept
            active_array[b, :, :] = (glmnet_result.betas .!= 0.0)
            first_moment_array[b,:, :] = glmnet_result.betas
        else
            active_array[b, :, :] = (glmnet_result.betas .!= 0.0)
            first_moment_array[b,:, :] = glmnet_result.betas
            intercept_first_moment_array[b,:] = glmnet_result.a0
        end
        if b%10 == 0
            println(b)
        end
    end
    return active_array, first_moment_array, intercept_first_moment_array
end
n_B = 1000
μ_B = 1.0

@time active_array, first_moment_array, intercept_first_moment_array = do_SS(
        A, y, n_B, μ_B, λ./m, tol=1.0e-10, randomize=true, intercept=intercept
    );

Π_experiment = vec(mean(active_array[:, :, cv_index], dims=1));
first_moment_experiment = vec(mean(first_moment_array[:, :, cv_index], dims=1));

Π_experiment_path = mean(active_array, dims=1)[1,:,:]';
first_moment_experiment_path = mean(first_moment_array, dims=1)[1,:,:]';


### AMPR ###
t_max = 1000
@time ampr_result_path = ampr(
    A, y, λ, pw=0.5, tol=1.0e-9, dumping=0.5, bootstrap_ratio=μ_B, w=8.0,
    t_max=t_max, info=true, debug=false);  # for path


x1_hat_amp = ampr_result_path.x1_hat[cv_index, :];
Π_amp = ampr_result_path.Π[cv_index, :];

x1_hat_amp_path = ampr_result_path.x1_hat;
Π_amp_path = ampr_result_path.Π;
t_iter_path = ampr_result_path.iter_num;

### visualization ###
sorted_index = [x[2] for x in sort([(x, y) for (x, y) in zip(Π_experiment_path[length(λ),:], 1:n)], by=x->x[1],rev=true)[:, 1]];  # 降順
n_path = 20  # pathの本数


p1 = plot(x1_hat_amp, x1_hat_amp, linecolor=:black, label="x=y")
p1 = plot!(x1_hat_amp, first_moment_experiment, seriestype=:scatter, label="")
p1 = plot!(title="first_moment (at cv optimal)", xlabel="AMPR", ylabel="naive")

p2 = plot(Π_amp, Π_amp, linecolor=:black, label="x=y")
p2 = plot!(Π_amp, Π_experiment, seriestype=:scatter, label="")
p2 = plot!(title=L"\Pi \mathrm{(at\;cv\;optimal)}", xlabel="AMPR", ylabel="naive")

plot!(λ, x1_hat_amp_path,label="", linecolor=:blue)
plot!(λ, first_moment_experiment_path, label="", linecolor=:red, linestyle=:dash)
vline!(λ_cv, color=:black, linestyle=:dash, label="cv optimal", linewidth=2.0)
p3 = plot!(xlabel=L"\lambda", ylabel=L"\hat{x}", title="first moment (dashed: AMPR, tick: naive)", legend=:bottomright, xscale=:log10, )

plot()
plot!(λ, Π_amp_path, label="", linecolor=:blue)
plot!(λ, Π_experiment_path, label="", color=:red, linestyle=:dash)
vline!(λ_cv, color=:black, linestyle=:dash, label="cv optimal", linewidth=2.0)
p4 = plot!(xlabel=L"\lambda", ylabel=L"\Pi", title="stability (dashed: AMPR, tick: naive)", legend=:bottomleft, xscale=:log10)

plot(p1, p2, p3, p4, size=(1500, 1000))
savefig("./img/result.pdf")
savefig("./img/result.png")