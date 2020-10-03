module AMPR
include("ApproximateSSUtil.jl")
using .ApproximateSSUtil
using LinearAlgebra, Random, Distributions

export ampr
export Fit

function cal_f1(c, q1u_hat)::Float64
    return c ./ (1.0 .+ c .* q1u_hat)
end

function cal_f2(c, q1u_hat)
    return (c ./ (1.0 .+ c .* q1u_hat)).^2.0
end

"""
    Fit

AMPRの結果型。もし、ampr_result::FitがAMPRから返ってきたものだとすると、
`ampr_result.q1x_hat`, `ampr_result.h1x_hat`, `ampr_result.x1_hat`, `ampr_result.v1x_hat`, `ampr_result.Π`と、`ampr_result.iter_num`が得られる。それぞれ、Onsager係数、局所場、1次モーメント、ブートストラップ分散、stability、収束するまでに要した反復回数である。

それぞれ、各正則化パラメータに対する結果を保持することを想定している。
"""
struct Fit
    "Onsager coefficient"
    q1x_hat::Array{Float64, 2}  # Onsager coefficient
    
    "conjugate variable for variance"
    v1x_hat::Array{Float64, 2}  # variance conjugate 
    
    "local field"
    h1x::Array{Float64, 2}  # local field 
    
    "stability"
    Π::Array{Float64, 2}  # stability
    
    "first moment"
    x1_hat::Array{Float64, 2}  # first moment

    "iteration number"
    iter_num::Array{Float64, 1}  # 反復回数
end

"""
ampr(A, y, λ, [, t_max, ...])

AMPR for linear regression

# Arguments
- `A::Array{Float64, 2}`: design matrix (m x n)
- `y::Array{Float64, 1}`: response (m). y_j takes a value in {-1, 1}
- `λ::Array{Float64, 1}`: regularization parameter
- `t_max=1000`: maximum number of iterations
- `dumping=0.1`: dumping parameter in [0,1]
- `tol=1e-8`: tolerance for rvamp iterations
- `bootstrap_ratio=1.0`: bootstrap sample size ratio
- `pw=0.0`: the probability used in randomized lasso (pw=1.0 corresponds to bolasso)
- `w=2.0`: strengthen parameter. with prob. pw, the regularization parameter is multiplied by w (λ_i -> w * λ_i)
- `clamp_min=1.0e-9`: 
- `clamp_max=1.0e9`: 
- `info=false`: whether to show iteration process
- `debug=false`: flag for debug mode
"""
function ampr(
    A::Array{Float64}, y::Array{Float64}, λ::Array{Float64};
    t_max=100::Int64, dumping=0.1::Float64, tol=1.0e-8::Float64,
    bootstrap_ratio=1.0::Float64, pw=0.0::Float64, w=2.0::Float64,
    clamp_min=1.0e-9, clamp_max=1.0e9, info=true, debug=false
)::Fit
    # preparation
    μ = bootstrap_ratio;
    η = dumping;
    c_max = Int(round(μ * 30));
    c_array = convert(Array{Float64}, 0:c_max);
    poisson_weight = [pdf(Poisson(μ), c) for c in 0:c_max];

    m, n = size(A);
    chi = zeros(n);
    A2 = A.^2.0;

    # initialization
    h1x = randn(n);
    v1x_hat = ones(n);
    q1x_hat = ones(n);

    x1_hat = zeros(n);
    chi1x = zeros(n);
    v1x = zeros(n);

    q1u_hat = zeros(m);
    v1u_hat = zeros(m);
    u1_hat = zeros(m);

    f1 = zeros(m);
    f2 = zeros(m);

    pre_x = zeros(n);
    pre_vx = zeros(n);

    η_p = zeros(n);
    η_m = zeros(n);
    η_p2 = zeros(n);
    η_m2 = zeros(n);

    temp = zeros(n);
    temp2 = zeros(n);
    

    # result array
    q1x_hat_array = zeros((length(λ), n));
    v1x_hat_array = zeros((length(λ), n));
    h1x_array = zeros((length(λ), n));
    x1_hat_array = zeros((length(λ), n));
    Π_array = zeros((length(λ), n));
    t_iter_array = 1.0.*t_max .* ones(length(λ));

    for (λ_index, γ) in enumerate(λ)
        if info
            println("$λ_index/$(length(λ)), γ=$γ")
        end

        for t in 1:t_max
            # save previous information
            pre_x .= x1_hat;
            pre_vx .= v1x;

            # estimation stage
            if debug
                println("\t ### estimation stage ###")
                println("\t x")
            end

            η_p .= (γ .- h1x) ./ v1x_hat.^0.5
            η_m .= (-1.0).*(γ .+ h1x) ./ v1x_hat.^0.5

            η_p2 .= (w .* γ .- h1x) ./ v1x_hat.^0.5
            η_m2 .= (-1.0) .* (w .* γ .+ h1x) ./ v1x_hat.^0.5
            
            x1_hat .= (1.0 .- pw) .* (
                (h1x .- γ) .* normal_sf.(η_p) .+ v1x_hat.^0.5 .* normal_pdf.(η_p)
                .+
                (h1x .+ γ) .* normal_cdf.(η_m) .- v1x_hat.^0.5 .* normal_pdf.(η_m)
            ) ./ q1x_hat .+ pw .*(
                (h1x .- w .* γ) .* normal_sf.(η_p2) .+ v1x_hat.^0.5 .* normal_pdf.(η_p2)
                .+
                (h1x .+ w .* γ) .* normal_cdf.(η_m2) .- v1x_hat.^0.5 .* normal_pdf.(η_m2)
            ) ./ q1x_hat

            chi1x .= clamp.(
                (1.0 .- pw) .* (
                    normal_sf.(η_p) .+ normal_cdf.(η_m)
                ) ./ q1x_hat .+ pw .* (
                    normal_sf.(η_p2) .+ normal_cdf.(η_m2)
                ) ./ q1x_hat,
                clamp_min, clamp_max
            )
            
            v1x .= clamp.(
                (1.0 .- pw) .* (
                    ((h1x .- γ).^2.0 .+ v1x_hat).*normal_sf.(η_p) .+ (2.0.*(h1x .- γ).* v1x_hat.^0.5 .+ η_p .* v1x_hat ).*normal_pdf.(η_p)
                    .+
                    ((h1x .+ γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m) .- (2.0.*(h1x .+ γ).* v1x_hat.^0.5 .+ η_m .* v1x_hat ).*normal_pdf.(η_m)
                )./q1x_hat.^2.0 .+ pw .* (
                    ((h1x .- w .* γ).^2.0 .+ v1x_hat).*normal_sf.(η_p2) .+ (2.0.*(h1x .- w .* γ).* v1x_hat.^0.5 .+ η_p2 .* v1x_hat ).*normal_pdf.(η_p2)
                    .+
                    ((h1x .+ w .* γ).^2.0 .+ v1x_hat).*normal_cdf.(η_m2) .- (2.0.*(h1x .+ w .* γ).* v1x_hat.^0.5 .+ η_m2 .* v1x_hat ).*normal_pdf.(η_m2)
                ) ./ q1x_hat .^ 2.0
                .-
                x1_hat.^2.0,
                clamp_min, clamp_max
            )
            if debug
                println("\t z")
                println()
            end

            # measurement stage
            if debug
                println("\t ### measurement stage ###")
            end
            # q1u_hat .= A2 * chi1x;
            # v1u_hat .= A2 * v1x;
            mul!(q1u_hat, A2, chi1x);
            mul!(v1u_hat, A2, v1x);
            
            # f1 .= 1.0 ./ (1.0 .+ q1u_hat);
            # f2 .= (1.0 ./ (1.0 .+ q1u_hat)).^2.0;
            f1 .= f1.*0.0;
            f2 .= f2.*0.0;
            for (c_index, c) in enumerate(c_array)
                f1 .= f1 .+ cal_f1.(c, q1u_hat).* poisson_weight[c_index];
                f2 .= f2 .+ cal_f2.(c, q1u_hat).* poisson_weight[c_index];
            end

            u1_hat .= f1 .* (y .- A * x1_hat .+ q1u_hat .* u1_hat);

            # conjugate update 
            if debug
                println("\t ### conjugate update ###")
            end
            # temp .= (A2' * f1);
            mul!(temp, A2', f1);
            mul!(temp2, A2', f2 .* v1u_hat .+ (f2 .- f1.^2.0) .* (u1_hat./f1).^2.0);
            q1x_hat .= dumping .* temp .+ (1.0 .- dumping) .* q1x_hat;
            h1x .= dumping .*(A' * u1_hat .+ temp .* x1_hat) .+ (1.0 .- dumping) .* h1x;
            v1x_hat .= dumping .* temp2 .+ (1.0 .- dumping) .* v1x_hat;

            # diff_x = mean((pre_x .- x1_hat).^2.0)/dumping;
            # diff_v = mean((pre_vx .- v1x).^2.0)/dumping;
            diff_x = mean((pre_x .- x1_hat).^2.0)/mean(pre_x.^2.0);
            diff_v = mean((pre_vx .- v1x).^2.0)/mean(pre_x.^2.0);

            if debug
                println("\t t=$t, diff_x=$diff_x, diff_v=$diff_v")
                println("\t min(q1x_hat)=$(minimum(q1x_hat)), min(v1x_hat)=$(minimum(v1x_hat))")
                println("\t mean(q1x_hat)=$(mean(q1x_hat)), mean(v1x_hat)=$(mean(v1x_hat))")
                println("\t mean(chi1x)=$(mean(chi1x)), mean(v1x)=$(mean(v1x))")
                println("\t mean(q1u_hat)=$(mean(q1u_hat)), mean(v1x)=$(mean(v1x))")
                println()
                
            end
            if maximum([diff_x, diff_v]) < tol && 5 < t
                println("converged! diff=$(maximum([diff_x, diff_v])), t=$t")
                t_iter_array[λ_index] = t
                break
            end
        end

        q1x_hat_array[λ_index, :] = q1x_hat;
        v1x_hat_array[λ_index, :] = v1x_hat;
        h1x_array[λ_index, :] = h1x;
        Π_array[λ_index, :] = chi1x .* q1x_hat;
        x1_hat_array[λ_index, :] = x1_hat;
        
    end

    # result = Fit(q1x_hat_array, v1x_hat_array, h1x_array, Π_array, x1_hat_array);
    result = Fit(q1x_hat_array, v1x_hat_array, h1x_array, Π_array, x1_hat_array, t_iter_array);

    return result
end

end