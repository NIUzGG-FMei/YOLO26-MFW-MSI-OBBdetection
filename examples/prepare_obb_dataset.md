1. "每个来源"是什么
来源（source）= 文件名前 8 位日期前缀（经 misc 合并后称为"有效来源"）。全量数据共 8 个来源：20000101、20040301、20231030/31、20231101、20231218/19/20。划分逻辑是按来源逐个独立执行：每个来源内部，先把自己的场景组打乱，再按 test/val 比例切给三个子集。效果就是——任何一个子集里，8 个来源都按接近总体的比例出现（这就是修复原 train/test 日期失衡的手段）。
2. 为什么越界目标默认丢弃；你的观察是对的
先对齐数字（全量审计）：93321 个目标里
- 越界丢弃 4096 个（4.4%）——真正从训练监督中消失
- difficult 12808 个（13.7%）——只从 clean 标签剔除，训练标签里一个不少
所以你说得对：默认策略下，越界丢弃对训练的实际影响 > difficult 策略（difficult 在训练侧根本没丢）。
丢弃的动机是几何严谨性：OBB 标签要求 4 点凸四边形且坐标归一化到 0,1，越界目标（顶点超出画布）不可能合法地写进 YOLO 格式，必须三选一：
1. 丢弃（默认）——监督干净，但丢真实目标；
2. 顶点钳制回边界——截出的形状不再是矩形（尤其目标在角上时变梯形），违反 OBB 的矩形假设；
3. minAreaRect 重建（旧工程做法）——把可见残片重新拟合成矩形，改写角度与尺寸，即我们分析过的标签噪声源。
还有个不对称风险你说的没提但同样存在：部署时图像边缘进出的目标模型照样能检出来，但因为 GT 被删，验证里被计为 FP——和 difficult 一样把"标注策略"变成"系统误检"。
中间路线（可加，未加）：可见面积占比阈值——裁剪后可见比例 ≥ 阈值（如 0.5）就钳制保留（可见四边形即正确监督，因为不可见部分在图像里本就不存在），低于阈值才丢弃，全程记入审计。需要的话我可以加 IDE_OOB_MIN_VISIBLE 参数实现。
3. 越界丢弃和 IDE_MIN_AREA_PX 的关系
默认模式：无关。代码顺序是 difficult → 越界 → 面积，越界检查先于面积检查且独立——越界目标无论多大面积一律丢弃；IDE_MIN_AREA_PX 只作用于界内目标。
只有 --keep-oob 开启时才有关系：钳制回边界后重新计算可见面积，若 < IDE_MIN_AREA_PX 则按 tiny 丢弃（clip_polygon_to_bounds 后二次面积检查，代码 prepare_obb_dataset.py 的 process_split 内）。
4. train/val/test 是不是从同一个"分钟组"里划分的
不是，恰好相反：分钟组是划分的最小原子单位——同一个组的所有帧整体进一个子集，组绝不跨子集（这是防帧级泄漏的设计，全量运行已验证 207 组零泄漏）。正确说法是：同一个来源（日期）的多个不同分钟组被分给了 train/val/test 三个子集。所以"同一分钟"不会同时出现在两个子集里，但"同一天"会。

 - - 所以，前8位是同一天，分为一层级；每个层级里面，按照分钟进行分组，每个分钟组的帧整体进一个子集，作为基本单位；对每个层级按照比例划分，对分钟组进行划分，得到 train/val/test 子集。

5. "holdout" 和 "misc" 是什么
- misc：组数 < IDE_SMALL_SOURCE_MIN_GROUPS（默认 3）的来源合并成的一个兜底来源。因为来源只有 1-2 个组时，按比例分层会抽出 0-1 组，子集里该来源缺失或噪声爆炸。misc 合并后照常参与分层，所以 misc 的帧会按比例进 train/val/test。全量数据 8 个来源都 ≥ 3 组，本次运行没有产生 misc；小实验数据（4 组）如果开着它才会触发。
- holdout：--split-mode holdout 模式，把指定来源（如 --holdout-sources 20040301）的所有组整体划入 test，train/val 完全看不到这些来源——这才是"对未见来源泛化"的真测试。注意：holdout 的 test 是"来源级隔离"，与 stratified 的 test（同域重采样）语义不同，两者不可混用。你也可以把 misc 写进 holdout 列表，把小来源全部变成 test。
6. 训练之后会不会 test？和 misc 的关系
不会自动 test。当前训练脚本的验证链是：
- 训练中每个 epoch → data.yaml 的 val 子集（251 图，含 difficult）；
- 训练结束后（IDE_RUN_MODE=train_and_clean_val）→ data_clean.yaml 的 val 子集（clean 口径）；
- test 子集（271 图）不参与任何自动流程，留给你用 validate_trained_obb_model.py(或者其它的验证脚本)（改指向 data.yaml/data_clean.yaml 的 test split）做最终终评。
与 misc 的关系：test 里可能出现 misc 帧——在 stratified 模式下，misc 只是"小来源的桶"，其帧照常按比例进三个子集，所以 test 不是"纯陌生来源"（这是分层口径的固有属性）。如果你要"test 与训练完全异源"，用 holdout 模式指定来源（或 misc）。


IDE_INCLUDE_DIFFICULT = True时，模型训练时会用上difficult目标，但同时clean指标的计算是不会加入difficult目标，对吗？
+ Thought: 2.7s
对，但要精确到两个层面：
GT 层面（你的理解正确）：
- 训练：data.yaml → labels/ → difficult 目标参与训练（12808 个）；
- clean 指标：data_clean.yaml → labels_clean/ → difficult 目标不参与 GT 匹配（只统计 63609 个非 difficult 目标）。
但有两个细节要注意：
1. 模型仍会检测出那些 difficult 目标，而 clean 指标里它们没有 GT——这些检测会被计为 FP。所以"clean 指标不含 difficult"严格说是"不含 difficult 的 GT"，检测残留仍会污染 FP/图、误检率（这正是我们讨论过的"标注不确定性转移成系统误检"问题）。如果希望彻底隔离，需要在评估侧把这些位置的检测也排除（当前验证脚本没有做这一步）。
2. "clean 指标"当前只算在 val 子集（训练脚本的 IDE_RUN_MODE=train_and_clean_val 用 data_clean.yaml 的 val）；test 子集不会自动跑，需要你自己用验证脚本对 test split 评估。




各子集每类目标数（prepared_obb_dataset）
labels/（含 difficult，训练与标准评估口径）：
类别	train	val	test	train占比	test占比	Δ
car	23026	3677	2346	41.5%	26.4%	−15.1pp
pedestrian	15255	3322	2516	27.5%	28.3%	+0.8pp
bike	9935	3569	2544	17.9%	28.6%	+10.7pp
awning-bike	2716	783	295	4.9%	3.3%	−1.6pp
van	1800	191	422	3.2%	4.7%	+1.5pp
truck	1168	124	136	2.1%	1.5%	−0.6pp
tricycle	924	215	257	1.7%	2.9%	+1.2pp
bus	715	103	378	1.3%	4.3%	+3.0pp
（labels_clean 口径比例结构相似，car 39.0%→20.7%、bike 17.3%→31.5%，Δ 更大。）
三个关键发现
1. 全局失衡 20-30 倍：car/pedestrian/bike 合计占 86.9%；bus/truck/tricycle 各只有 1-2%。这正是验证报告里 truck mAP50≈0.35、bike mAP50-95≈0.26 的根源。
2. 新发现——源分层 ≠ 类分层：来源比例修好了，但 train/test 的类构成仍偏差（car −15pp、bike +11pp）。原因是"分钟组=单一场景"的类混合高度同质，组级划分继承场景异质性；源分层只能保证来源比例。后果：模型在 car 主导的训练里学到的类别先验，被用在 bike/pedestrian 主导的 test 上——per-class AP 不受影响（按类平均），但 FP/图、误检率这些业务指标会系统性偏向。
3. 弱类 × 难例双重打击：truck/tricycle 既是稀有类（1-2%），difficult 比例又高（18-42%），样本少+标注难叠加。
建议（按投入产出排序）
优先级	手段	说明	成本
1	per-class 指标评估	对比时只看每类 AP 与每类 FP，不比较均值（测试脚本已支持按类输出）	零
2	cls_pw 类加权损失	ultralytics 原生参数，给 bus/truck/tricycle/van ×3-5、awning-bike ×2	低
3	稀有类帧过采样（train 侧）	在现有过采样机制上按"含 truck/bus/tricycle 的帧"选取，而非按来源	低-中
4	目标级 copy-paste 增强	稀有类目标复制粘贴进其他帧（OBB 官方默认关闭，需自定义实现）	中-高
5	两阶段训练	全类训基线 → 用稀有类帧微调	中
不推荐：整图级随机过采样（一帧含 1 个 truck 也含 20 个 car，效率极低）；降采样 car 类（信息损失）。
另外提醒：test 类构成与 train 不同是"场景组划分"的固有属性，不是 bug——但解释指标时要把它考虑进去（如 test 里 bike 占比高，整体 FP 被 bike 虚警推高是预期的）。



1. cls_pw 逆频率加权（train_obb_dataset.py 配置区第 6 组）
- IDE_CLS_PW = 0.5（默认半强度；0.0 关闭 / 1.0 完整逆频率）
- 语义为本仓库 fork 的指数式实现：训练开始时 set_class_weights() 自动统计训练集每类频次，权重 = (1/频次)^cls_pw 且均值归一化为 1，乘进 BCE cls 损失——无需手工指定每类权重。
2. 稀有类帧过采样（训练侧）
- IDE_RARE_OVERSAMPLE_CLASSES = ("bus","truck","tricycle","van") + IDE_RARE_MIN_OBJECTS = 3 + IDE_RARE_OVERSAMPLE_REPEATS = 1
- 训练开始时扫描 train 标签，重复"帧内稀有类目标数 ≥ 阈值"的帧，生成 train_rare_oversampled.txt + 派生 data_oversampled.yaml（train 指向列表，val/test 不动），与 prepare 侧的夜间过采样列表自动叠加（若 data.yaml 的 train 已是列表文件则以它为基准）。