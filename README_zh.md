# Hodge-SDE 代码与模型包

更新：2026-10-16。训练核心使用经过数值检查的精确 Hodge 投影（`exact_hodge_v1`），三个纯数值参数示例已从本轮正式精确拟合中更新。Fig2 v12、Fig3 exact_estimator_v2、Fig4 v10、Fig5 v18、Fig6 v15 的主图适配器使用独立、锁定 SHA 的外部输入；分支 QA 与本包验收分别记录于 VALIDATION.md。本目录尚未上传 GitHub 或指定许可证。

## 包内内容

- 模型训练、逐折评价和 Fig2–6 主要代码，按 `src/`、`figures/fig2`–`figures/fig6` 分类。
- 已通过 `exact_hodge_v1` 校验的 HCP Hodge-SDE 参数模型：150、250、450 ROI 各一个。按正式缓存文件名顺序选取，不按结果优劣挑选；它们是运行示例，不是群体代表样本。四通道顺序仍为 Sym/Grad/Curl/Harm；第三个反对称通道（完整 API 第四 slot）为精确 Harm。
- 本轮 Fig2–6 当前主图核心；历史 Fig2 v11、Fig4 v9 原生可编辑入口及 Fig3 v13 修正真值流程明确作为 legacy 保留，不是当前主结果默认入口。

包内没有被试数据、被试表、观测/预测时间序列、临床信息、私人编号、数据路径、字体或图谱资产。全体被试缓存未整批复制。模型仅保留运行所需的数值参数，详见 [模型说明](models/README.md)。

## 运行模型

MATLAB 切换到本目录后：

```matlab
trajectory = demo_fitted_model(150);       % 拟合噪声下的随机动力学
trajectory = demo_fitted_model(250,false); % 确定性动力学
trajectory = demo_fitted_model(450);
test_parameter_export;
```

从可复现的合成初值开始，无需原项目加载器或额外工具箱。三个分辨率的随机/确定性轨迹均已核对：相同初值和种子下，导出前后完全一致。更多获准使用的模型用 `export_hodge_parameters(model,filename)` 导出，只复制数值白名单字段，不覆盖已有文件。

## 小检查

```powershell
python -m pip install -r requirements-figures.txt
python -B -m figures.check
python -B -m figures.validation_smoke
python -B -m figures.exact_smoke
```

```matlab
demo_synthetic(false);
test_whiteness_synthetic;
```

检查使用小型合成输入，不重训全体被试。执行记录见 [VALIDATION.md](VALIDATION.md)。依赖表记录本机测试版本；全新环境安装尚需另测。

## 生成最新版主图

准备外部输入后，在包根目录运行：

```powershell
$env:HODGE_WORK_ROOT = 'E:/my_hodge_work'
$env:HODGE_FONT_FILE = 'E:/my_fonts/Helvetica-Neue.ttf'
python -B -m figures.fig2.render_exact
python -B -m figures.fig3.render_exact
python -B -m figures.fig4.render_exact
python -B -m figures.fig5.render_exact
python -B -m figures.fig6.render_exact
```

依次对应 Fig2 v12、Fig3 exact_estimator_v2（冻结修正真值、估计端 exact）、Fig4 v10（a–d exact Harm，e–g 原实验）、Fig5 v18、Fig6 v15 exact_disease_v2。各入口支持 `--check-inputs` 只验输入不绘图。新入口与最终44张 SI 编号见 [精确模型输入契约](figures/EXACT_INPUTS.md)。工作目录须与代码包分开，被试级中间表只保留在工作目录。

Fig2 d／上游 S10 保留九模型原归档 call totals，标注 Archived workflow timing；新 exact train/test-only 耗时另列源表与 Methods，不与旧含 NBE 的调用耗时比较速度。

- [Fig2 输入与公平生成](figures/fig2/README.md)：一步预测、NBE 白度、逐折生成比较与运行时间；保留 Subspace 联合噪声、训练段估计的 NMM 噪声和 Hodge 的 TR/扩散单位。
- [Fig3 严格真值与重画](figures/fig3/README_zh.md)：340 次 b/c/e 试验、280 次长度试验、20 个区域分析模型及来源核对。
- [Fig4 输入](figures/fig4/README.md)：同一扫描内稳定性、跨图谱对应、消融和替换，保留统计与布局。
- Fig5 v18 使用843名评价配对者，g 为本轮843人精确模型匹配对照；历史822/450分母保留在历史验证实现中。Fig6 v15 的正式六项检验、Holm-6、RMS Holm-3及冻结五折不改；新 fullfit MDD LOSO（S41）改善7.481%，区别于旧 common-state（S43）的7.657%；主图 f 的 MDD 人级五折 OOF 改善约17.564%。
- 包仅迁入本轮主图核心。既有 Fig5/Fig6 `render_si` 保留历史分支编号，不能直接当作最终44张 SI 的生成器；最终映射以 [精确输入说明](figures/EXACT_INPUTS.md) 为准。
- 历史验证实现及外部输入约定见 [Fig5 验证说明](figures/fig5/validation/README_zh.md) 与 [Fig6 验证说明](figures/fig6/validation/README_zh.md)。其中主流程/验证分支标签描述原来的分析。新训练统一调用 `src/hlgsde_exact_hodge.m`：固定每个连通分量的参考势后求解图拉普拉斯方程，并直接对边—三角形矩阵作稀疏最小二乘投影；重构、子空间条件和正交检查通过后才返回模型。求解失败会报错，不继续保存。

新版本需要单独重拟合、评分和更新图表数据，不能给旧模型加一个版本标签就视为已迁移。新精确缓存与历史图表输入必须分开保存。

主图白度从已有 NBE 缓存重汇总为 `mean(Q)/mean(Qthr)`；原 Q/置换计算须单独取得 NBE 实现。存档 FigS11 保留原来的逐折比值再平均口径。

## 依赖与发布

训练核心 `src/nonlinear_HLGSDE.m` 需要另行取得 Hodge 工具箱及 rest-system-id Utilities；参数模型示例无需这些训练工具。其他依赖和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

`manifest.json` 将原始来源 SHA256 与当前包文件 SHA256 分开记录。为兼容旧缓存，`B_SOURCE_SHA256`、`DYNAMICS_SOURCE_SHA256` 保持原脚本 provenance；新运行指纹同时写入实际 package 模块 hash，不能把二者混称为执行源码 hash。既有预处理、统计参数和正式图不因整理而改变。数据由读者独立申请或下载；作者须确认代码及拟合参数的分发权和许可证。不要将含真实数据的工作目录上传。
