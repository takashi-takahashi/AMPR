# using Revise
include("./src/AMPR.jl")
using .AMPR
# using AMPR
using LinearAlgebra, Random
using Distributions, GLMNet
using Plots, LaTeXStrings

Random.seed!(0);

### settings ###
m, n = 4898, 700;
ρ = 0.1;
σ = 0.1;
intercept = false;

### data synthesization ###
x_0 = rand(Normal(0.0, 1.0), n) .* rand(Binomial(1, ρ), n);
A = rand(Normal(0.0, 1.0/n^0.5), (m, n));
A = A .- mean(A, dims=1);
A = A ./ std(A, dims=1);
y = A * x_0 + rand(Normal(0.0, σ), m);
# y = y .- mean(y)


### cross validation ###
@time cv_result = glmnetcv(
    A, y, Normal(), 
    intercept=false, standardize=false, 
    tol=1e-10, maxit=10000, nfolds=10, nlambda=100
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
        A, y, n_B, μ_B, λ./m, tol=1.0e-9, randomize=true, intercept=intercept
    );
Π_experiment = vec(mean(active_array[:, :, cv_index], dims=1));
first_moment_experiment = vec(mean(first_moment_array[:, :, cv_index], dims=1));
intercept_experiment = mean(intercept_first_moment_array[:, cv_index]);

Π_experiment_path = mean(active_array, dims=1)[1,:,:]';
first_moment_experiment_path = mean(first_moment_array, dims=1)[1,:,:]';
intercept_experiment_path = mean(intercept_first_moment_array, dims=1);


### AMPR ###
# @time ampr_result = ampr(
#     A, y, λ_cv, pw=0.5, tol=1.0e-10,
#     t_max=10000, info=true, debug=false);  # for single lambda

@time ampr_result_path = ampr(
    A, y, λ, pw=0.5, tol=1.0e-11, dumping=0.1,
    t_max=10000, info=true, debug=false);  # for path

x1_hat_amp = ampr_result_path.x1_hat[end, :];
Π_amp = ampr_result_path.Π[end, :];

x1_hat_amp_path = ampr_result_path.x1_hat;
Π_amp_path = ampr_result_path.Π;

### visualization ###

sorted_index = [x[2] for x in sort([(x, y) for (x, y) in zip(Π_experiment_path[length(λ),:], 1:n)], by=x->x[1],rev=true)[:, 1]];  # 降順
n_path = 20  # pathの本数

p1 = plot(x1_hat_amp, x1_hat_amp, linecolor=:black, label="x=y")
p1 = plot!(x1_hat_amp, first_moment_experiment, seriestype=:scatter, label="")
p1 = plot!(title="first_moment", xlabel="AMPR", ylabel="naive")

p2 = plot(Π_amp, Π_amp, linecolor=:black, label="x=y")
p2 = plot!(Π_amp, Π_experiment, seriestype=:scatter, label="")
p2 = plot!(title=L"\Pi", xlabel="AMPR", ylabel="naive")

p3 = plot(λ, x1_hat_amp_path[:, sorted_index[1:n_path]], xscale=:log10, label="", color=:blue)
p3 = plot!(λ, first_moment_experiment_path[:, sorted_index[1:n_path]], color=:red, label="", linestyle=:dash)
p3 = plot!(title="x1_hat (blue:AMPR, red:naive)", xlabel=L"\lambda")

p4 = plot(λ, Π_amp_path[:, sorted_index[1:n_path]], label="", color=:blue)
p4 = plot!(λ, Π_experiment_path[:, sorted_index[1:n_path]], label="", color=:red, xscale=:log10, linestyle=:dash)
p4 = plot!(title="Π (blue:AMPR, red:naive)", xlabel=L"\lambda")


plot(p1, p2, p3, p4, size=(1000, 600))
