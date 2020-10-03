# AMPR
[Tomoyuki Obuchi and Yoshiyuki Kabashima, *Semi-analytic resampling in lasso*, Journal of Machine Learning Research **20** (2019), no. 70, 1–33.](https://www.jmlr.org/papers/v20/18-109.html)で提案された近似メッセージパッシングに基づく、近似的stability selection法のJulia実装である

# disclaimer
* えーい、と適当に書いたものなので、ひょっとしたらバグってるかもしれない。その時はissueで報告してほしい
* また、パッケージとしては本当に大変いい加減である

# quick start
stability selectionをデフォルトパラメータで実行するためには
```
ampr_result = ampr(A, y, λ)
```
`ampr_result`は`Fit`型のものが返ってくる。１次モーメントとか、stabilityを保持しており、`ampr_result.x1_hat`とかで取り出すことが可能。内訳はAMPR.Fitのdocstring参照。

ダンピング係数`dumping`、反復回数`t_max`、収束基準`tol`、stability selectionのパラメータ`pw, w`を指定したければ
```
ampr(
        A, y, λ, 
        dumping=0.8, t_max=50, tol=1.0e-6, pw=0.5, w=2.0
    )
```
という具合。


# 実行例
実行例のファイルを置いておく。ただしこれはそのまま実行するというより、なんとなく眺めて使い方の雰囲気を知るというようなものではないかと思う。
* do_experiment_synthetic.jl
    - 合成データでの実験
    - ナイーブなSSも実行して、精度を実験と比較している。
* do_experiment_real.jl
    - 実データでの実験
        - `path_to_AMPR/real_data/A.csv`, `path_to_AMPR/real_data/y.csv`が存在する場合の例
        - 実行結果は`path_to_AMPR/img/`以下に出力する。
    - ナイーブなSSも実行して、精度を実験と比較している



# dependency
* Distributions.jl
* GLMNet.jl, CSV.jl
    - 実験で使っている
* Plots, LaTeXStrings, Colors
    - 実験結果の可視化に利用している


# see also
* 提案者による実装として、[matlab版](https://github.com/T-Obuchi/AMPR_lasso_matlab)と、[Python版](https://github.com/T-Obuchi/AMPR_lasso_python)がある。
* また、[期待値伝搬法ベースの手法](https://github.com/takashi-takahashi/ApproximateSS)もある。こちらは、特徴量間に相関があり、かつデータ数 << パラメータ数の状況で有効である.
    - 論文は[こちら](https://iopscience.iop.org/article/10.1088/1742-5468/ababff)


