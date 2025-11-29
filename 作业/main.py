import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


"""
Online Retail 数据格式示例：
InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country
536365	85123A	WHITE HANGING HEART T-LIGHT HOLDER	6	2010/12/1 8:26	2.55	17850	United Kingdom
"""


"""数据预处理与特征工程"""
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗缺失值与异常值并规范数据类型

    - 删除 `CustomerID` 缺失的记录
    - 删除异常记录：`Quantity` ≤ 0 或 `UnitPrice` ≤ 0 的行
    - 规范字段：将 `InvoiceDate` 转为 `datetime`，`CustomerID` 转为整型
    """
    # 复制一份数据，避免修改原始 DataFrame
    df = df.copy()

    # 删除 CustomerID 缺失的记录
    df = df.dropna(subset=["CustomerID"])

    # 删除数量为负或单价为 0/负 的异常记录
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 规范数据类型：转换日期与客户ID
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])  # 删除无法解析日期的记录
    df["CustomerID"] = df["CustomerID"].astype(int)
    # 统一商品编码类型，避免出现混合类型导致比较错误
    df["StockCode"] = df["StockCode"].astype(str)

    return df

def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造用户级特征（RFM + 购买时段偏好）

    输出列：
    - CustomerID：用户ID
    - recency_days：最近一次购买距参考日的天数（参考日为数据集中最大日期）
    - frequency：购买次数（按唯一 `InvoiceNo` 计）
    - monetary：总消费金额（`Quantity * UnitPrice`）
    - morning_ratio：晨间订单占比（6:00-12:00）
    - night_ratio：夜间订单占比（18:00-24:00）
    """
    # 复制数据，保证安全
    df = df.copy()

    # 准备金额字段与参考日期
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    ref_date = df["InvoiceDate"].max().normalize()  # 参考日期：最大交易日（取日期部分）

    # RFM：最近、频次、金额
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    recency_days = (ref_date - last_purchase.dt.normalize()).dt.days.rename("recency_days")
    frequency = df.groupby("CustomerID")["InvoiceNo"].nunique().rename("frequency")
    monetary = df.groupby("CustomerID")["TotalPrice"].sum().rename("monetary")

    # 订单级时段偏好：以唯一订单统计晨间/夜间占比
    inv = df[["CustomerID", "InvoiceNo", "InvoiceDate"]].drop_duplicates()
    inv["hour"] = inv["InvoiceDate"].dt.hour
    inv["is_morning"] = (inv["hour"] >= 6) & (inv["hour"] < 12)
    inv["is_night"] = (inv["hour"] >= 18) & (inv["hour"] < 24)

    # 以布尔均值近似占比（True 计为 1，False 计为 0）
    morning_ratio = inv.groupby("CustomerID")["is_morning"].mean().fillna(0.0).rename("morning_ratio")
    night_ratio = inv.groupby("CustomerID")["is_night"].mean().fillna(0.0).rename("night_ratio")

    # 合并所有特征
    user_features = (
        pd.concat([recency_days, frequency, monetary, morning_ratio, night_ratio], axis=1)
        .reset_index()
    )

    return user_features

def compute_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造商品级特征（频次与平均订单量）

    输出列：
    - StockCode：商品编码
    - purchase_frequency：商品被购买的订单频次（唯一 `InvoiceNo` 计）
    - avg_order_quantity：平均订单量（按行项 `Quantity` 的均值）
    """
    # 复制数据以避免副作用
    df = df.copy()

    # 按商品分组，统计被购买的订单数量与平均下单数量
    grouped = df.groupby("StockCode")
    purchase_frequency = grouped["InvoiceNo"].nunique().rename("purchase_frequency")
    avg_order_quantity = grouped["Quantity"].mean().rename("avg_order_quantity")

    product_features = (
        pd.concat([purchase_frequency, avg_order_quantity], axis=1)
        .reset_index()
    )

    return product_features


"""关联规则与用户分群"""
def mine_association_rules(df: pd.DataFrame, min_support: float = 0.01, top_n: int = 5) -> pd.DataFrame:
    """用 Apriori 挖掘频繁项集并输出强关联规则

    - 事务构造：按 `InvoiceNo` 聚合为商品集合（以 `StockCode` 为项）
    - Apriori：自底向上生成频繁项集（支持度阈值 `min_support`）
    - 规则生成：对每个频繁项集（|S|≥2）生成 A→B 规则，计算 `support/confidence/lift`
    - 排序输出：按 `lift` 降序、`confidence` 次序，返回前 `top_n` 条规则
    """
    transactions = (
        df.groupby("InvoiceNo")["StockCode"].apply(lambda x: set(map(str, x))).tolist()
    )
    n_trans = len(transactions)
    if n_trans == 0:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"]).astype({
            "support": float, "confidence": float, "lift": float
        })

    # 计算单项集支持度
    item_counts = {}
    for t in transactions:
        for i in t:
            item_counts[i] = item_counts.get(i, 0) + 1
    L = []
    L1 = [{i} for i, c in item_counts.items() if c / n_trans >= min_support]
    L.append(L1)
    support_dict = {frozenset({i}): item_counts[i] / n_trans for i in item_counts if {i} in L1}

    def has_infrequent_subset(candidate, prev_level_sets):
        prev_set = set(map(frozenset, prev_level_sets))
        for s in combinations(candidate, len(candidate) - 1):
            if frozenset(s) not in prev_set:
                return True
        return False

    # 迭代生成更高阶频繁项集
    k = 2
    prev_level = L1
    while prev_level:
        candidates = []
        prev_level_sorted = [sorted(s) for s in prev_level]
        for i in range(len(prev_level_sorted)):
            for j in range(i + 1, len(prev_level_sorted)):
                c = sorted(set(prev_level_sorted[i]) | set(prev_level_sorted[j]))
                if len(c) == k:
                    c_set = set(c)
                    if not has_infrequent_subset(c_set, prev_level):
                        candidates.append(c_set)
        counts = {}
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[frozenset(c)] = counts.get(frozenset(c), 0) + 1
        Lk = [set(fs) for fs, cnt in counts.items() if cnt / n_trans >= min_support]
        for fs, cnt in counts.items():
            sup = cnt / n_trans
            if sup >= min_support:
                support_dict[fs] = sup
        if not Lk:
            break
        L.append(Lk)
        prev_level = Lk
        k += 1

    # 汇总频繁项集
    frequent_sets = set()
    for level in L:
        for s in level:
            frequent_sets.add(frozenset(s))

    # 生成规则并计算指标
    rules = []
    frequent_lookup = set(frequent_sets)
    for fs in frequent_sets:
        if len(fs) < 2:
            continue
        items = list(fs)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = fs.difference(A)
                if A in frequent_lookup and B in frequent_lookup:
                    sup_fs = support_dict.get(fs, 0.0)
                    sup_A = support_dict.get(A, 0.0)
                    sup_B = support_dict.get(B, 0.0)
                    if sup_A > 0 and sup_B > 0:
                        conf = sup_fs / sup_A
                        lift = conf / sup_B
                        rules.append({
                            "antecedent": sorted(list(A)),
                            "consequent": sorted(list(B)),
                            "support": sup_fs,
                            "confidence": conf,
                            "lift": lift,
                        })

    rules_df = pd.DataFrame(rules)
    if rules_df.empty:
        return rules_df
    rules_df = rules_df.sort_values(by=["lift", "confidence"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    return rules_df

def visualize_rules_heatmap(rules_df: pd.DataFrame, save_path: str = "关联规则_热力图.png") -> bool:
    p = rules_df[(rules_df["antecedent"].map(len) == 1) & (rules_df["consequent"].map(len) == 1)]
    if len(p) == 0:
        return False
    a = sorted(set(p["antecedent"].map(lambda x: x[0])))
    b = sorted(set(p["consequent"].map(lambda x: x[0])))
    M = np.zeros((len(a), len(b)))
    for _, r in p.iterrows():
        i = a.index(r["antecedent"][0])
        j = b.index(r["consequent"][0])
        M[i, j] = r["lift"]
    plt.figure(figsize=(6, 4))
    plt.imshow(M, cmap="Reds", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(b)), b, rotation=90)
    plt.yticks(range(len(a)), a)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True

def visualize_rules_network(rules_df: pd.DataFrame, save_path: str = "关联规则_网络图.png", max_edges: int = 50) -> bool:
    nodes = set()
    for _, r in rules_df.iterrows():
        nodes.update(r["antecedent"])
        nodes.update(r["consequent"])
    nodes = sorted(nodes)
    n = len(nodes)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {nodes[i]: (np.cos(ang[i]), np.sin(ang[i])) for i in range(n)}
    plt.figure(figsize=(8, 8))
    for k, (x, y) in pos.items():
        plt.scatter([x], [y], s=100, color="#4C78A8")
        plt.text(x, y, str(k), ha="center", va="center", fontsize=8, color="white")
    E = rules_df.sort_values(["lift", "confidence"], ascending=False).head(max_edges)
    for _, r in E.iterrows():
        lw = 1 + 2 * min(float(r["lift"]) / 100.0, 1.0)
        for a in r["antecedent"]:
            for b in r["consequent"]:
                xa, ya = pos[a]
                xb, yb = pos[b]
                plt.plot([xa, xb], [ya, yb], color="#E45756", linewidth=lw, alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True

def plot_feature_importance_bar(importance_df: pd.DataFrame, save_path: str = "特征重要性_条形图.png") -> None:
    d = importance_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(6, 4))
    plt.barh(d["feature"], d["importance"], color="#72B7B2")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix_from_model(df: pd.DataFrame, uf: pd.DataFrame, days: int = 30, n_estimators: int = 100, sample_ratio: float = 0.7, thresh: float = 0.5, save_path: str = "复购预测_混淆矩阵.png") -> dict:
    g = df.sort_values("InvoiceDate").groupby("CustomerID")["InvoiceDate"]
    mg = g.diff().dt.days.groupby(df.sort_values("InvoiceDate")["CustomerID"]).min()
    c = df.groupby("CustomerID")["InvoiceNo"].nunique()
    lab = ((mg <= days) & (c >= 2)).fillna(False).astype(int).rename("repurchase_30d").reset_index()
    m = uf.merge(lab, on="CustomerID", how="left").fillna({"repurchase_30d": 0})
    cols = ["recency_days", "frequency", "monetary", "morning_ratio", "night_ratio"]
    X = m[cols].astype(float).values
    y = m["repurchase_30d"].astype(int).values
    rng = np.random.default_rng(123)
    n = len(X)
    idx = rng.permutation(n)
    t = int(0.7 * n)
    tr, te = idx[:t], idx[t:]
    def gini(z):
        return 0.0 if len(z) == 0 else 2 * z.mean() * (1 - z.mean())
    trees = []
    for _ in range(n_estimators):
        s = max(1, int(sample_ratio * len(tr)))
        b = rng.integers(0, len(tr), size=s)
        Xb, yb = X[tr][b], y[tr][b]
        best_feat, best_thr, pl, pr, gain = -1, -1.0, 0.0, 0.0, -1e9
        base = gini(yb)
        for fi in range(len(cols)):
            qs = np.quantile(Xb[:, fi], q=np.linspace(0.1, 0.9, 9))
            for thr in np.unique(qs):
                L = yb[Xb[:, fi] <= thr]
                R = yb[Xb[:, fi] > thr]
                gl = gini(L)
                gr = gini(R)
                g0 = base - (len(L) / s) * gl - (len(R) / s) * gr
                if g0 > gain and len(L) > 0 and len(R) > 0:
                    best_feat, best_thr, pl, pr, gain = fi, float(thr), float(L.mean()), float(R.mean()), g0
        if best_feat < 0:
            best_feat, best_thr, pl, pr = 0, float(np.median(Xb[:, 0])), float(yb.mean()), float(yb.mean())
        trees.append((best_feat, best_thr, pl, pr))
    p = np.zeros(len(te))
    for fi, thr, pl, pr in trees:
        left = X[te, fi] <= thr
        p[left] += pl
        p[~left] += pr
    p = p / max(1, len(trees))
    yt = y[te]
    yp = (p >= thresh).astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    M = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4, 4))
    plt.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(M[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]) 
    plt.yticks([0, 1], ["True 0", "True 1"]) 
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

def cluster_users_rfm(user_features: pd.DataFrame, n_clusters: int = 3, save_prefix: str = "user_clusters") -> pd.DataFrame:
    """基于 RFM 特征进行 K-Means 聚类并可视化分群

    - 输入：`user_features` 包含 `recency_days/frequency/monetary`
    - 预处理：对 R/F/M 做 Min-Max 归一化，缓解量纲影响
    - 聚类：实现简易 K-Means（随机初始化，最多 100 次迭代，空簇重置）
    - 标签：按中心的价值分数进行 3 类标签映射（高价值活跃/成长/沉睡）
    - 可视化：保存 3D 散点图与雷达图到当前目录
    返回：包含 `CustomerID/cluster/label` 的 DataFrame
    """
    cols = ["recency_days", "frequency", "monetary"]
    X = user_features[cols].astype(float).values
    mins = X.min(axis=0)
    ptp = np.ptp(X, axis=0)
    ptp_safe = np.where(ptp == 0, 1.0, ptp)
    X_norm = (X - mins) / ptp_safe

    rng = np.random.default_rng(42)
    if len(X_norm) < n_clusters:
        n_clusters = max(1, len(X_norm))
    init_idx = rng.choice(len(X_norm), size=n_clusters, replace=False)
    centroids = X_norm[init_idx]

    for _ in range(100):
        dists = np.linalg.norm(X_norm[:, None, :] - centroids[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = centroids.copy()
        for k in range(n_clusters):
            points = X_norm[labels == k]
            if len(points) == 0:
                new_centroids[k] = X_norm[rng.integers(0, len(X_norm))]
            else:
                new_centroids[k] = points.mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids

    center_scores = centroids[:, 1] + centroids[:, 2] - centroids[:, 0]
    order = np.argsort(center_scores)[::-1]
    label_map = {}
    if n_clusters >= 3:
        label_map[order[0]] = "高价值活跃用户"
        label_map[order[-1]] = "沉睡用户"
        for k in range(n_clusters):
            if k not in label_map:
                label_map[k] = "成长用户"
    elif n_clusters == 2:
        label_map[order[0]] = "高价值活跃用户"
        label_map[order[1]] = "沉睡用户"
    else:
        label_map[order[0]] = "普通用户"

    result = user_features[["CustomerID"]].copy()
    result["cluster"] = labels
    result["label"] = result["cluster"].map(label_map)

    # 3D 散点图
    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        colors = plt.cm.get_cmap("tab10", n_clusters)
        for k in range(n_clusters):
            pts = X_norm[labels == k]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20, color=colors(k), label=f"Cluster {k} - {label_map.get(k, '')}")
        ax.set_xlabel("Recency (norm)")
        ax.set_ylabel("Frequency (norm)")
        ax.set_zlabel("Monetary (norm)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(f"{save_prefix}_三维散点图.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # 雷达图（簇中心）
    try:
        labels_axes = ["Recency", "Frequency", "Monetary"]
        angles = np.linspace(0, 2 * np.pi, len(labels_axes), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111, polar=True)
        for k in range(n_clusters):
            vals = np.concatenate([centroids[k], [centroids[k][0]]])
            ax2.plot(angles, vals, label=f"Cluster {k} - {label_map.get(k, '')}")
            ax2.fill(angles, vals, alpha=0.1)
        ax2.set_thetagrids(angles[:-1] * 180 / np.pi, labels_axes)
        ax2.set_title("RFM Cluster Centers (Radar)")
        ax2.legend(fontsize=8, bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        fig2.savefig(f"{save_prefix}_雷达图.png", dpi=150)
        plt.close(fig2)
    except Exception:
        pass

    return result


"""复购行为预测与营销策略"""
def predict_repurchase_within_30d(df: pd.DataFrame, user_features: pd.DataFrame, days: int = 30, n_estimators: int = 100, sample_ratio: float = 0.7) -> tuple:
    """训练并预测 30 天内复购概率（单函数完成打标、训练与预测）

    - 打标：按用户相邻购买间隔是否 ≤ `days` 生成 `repurchase_30d`
    - 训练：自助采样的随机森林桩（决策桩，按 Gini 减少选阈值）
    - 预测：输出每位用户的 30 天复购概率
    返回：(probs_df, importances)
    """
    g = df.sort_values("InvoiceDate").groupby("CustomerID")["InvoiceDate"]
    min_gap_days = g.diff().dt.days.groupby(df.sort_values("InvoiceDate")["CustomerID"]).min()
    counts = df.groupby("CustomerID")["InvoiceNo"].nunique()
    labels_df = ((min_gap_days <= days) & (counts >= 2)).fillna(False).astype(int).rename("repurchase_30d").reset_index()

    dfm = user_features.merge(labels_df, on="CustomerID", how="left").fillna({"repurchase_30d": 0})
    feature_cols = ["recency_days", "frequency", "monetary", "morning_ratio", "night_ratio"]
    X = dfm[feature_cols].astype(float).values
    y = dfm["repurchase_30d"].astype(int).values

    rng = np.random.default_rng(7)
    trees = []
    importances = {c: 0.0 for c in feature_cols}

    def gini_of(labels):
        if len(labels) == 0:
            return 0.0
        p1 = labels.mean()
        return 2 * p1 * (1 - p1)

    for _ in range(n_estimators):
        n = len(X)
        m = max(1, int(sample_ratio * n))
        idx = rng.integers(0, n, size=m)
        Xb, yb = X[idx], y[idx]

        best_feat, best_thr = None, None
        best_gain = -np.inf
        best_left_p, best_right_p = 0.0, 0.0
        base_gini = gini_of(yb)
        for fi, f in enumerate(feature_cols):
            col = Xb[:, fi]
            qs = np.quantile(col, q=np.linspace(0.1, 0.9, 9))
            for thr in np.unique(qs):
                left = yb[col <= thr]
                right = yb[col > thr]
                gl = gini_of(left)
                gr = gini_of(right)
                wgl = (len(left) / m) * gl + (len(right) / m) * gr
                gain = base_gini - wgl
                if gain > best_gain and len(left) > 0 and len(right) > 0:
                    best_gain = gain
                    best_feat = f
                    best_thr = float(thr)
                    best_left_p = float(left.mean())
                    best_right_p = float(right.mean())
        if best_feat is None:
            best_feat = feature_cols[0]
            best_thr = float(np.median(Xb[:, 0]))
            best_left_p = best_right_p = float(yb.mean())
            best_gain = 0.0

        trees.append({"feature": best_feat, "threshold": best_thr, "p_left": best_left_p, "p_right": best_right_p})
        importances[best_feat] += max(0.0, best_gain)

    Xdf = user_features.set_index("CustomerID")[feature_cols].astype(float)
    probs = np.zeros(len(Xdf))
    for t in trees:
        f = t["feature"]
        thr = t["threshold"]
        left_mask = Xdf[f] <= thr
        probs[left_mask.values] += t["p_left"]
        probs[(~left_mask).values] += t["p_right"]
    probs = probs / max(1, len(trees))
    probs_df = pd.DataFrame({"CustomerID": Xdf.index.values, "prob_30d": probs})

    return probs_df, importances

def feature_importance_report(importances: dict) -> pd.DataFrame:
    """输出特征重要性排名（Gini 减少的累积值）"""
    df_imp = pd.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
    return df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

def recommend_marketing_strategies(segments: pd.DataFrame, probs: pd.DataFrame, high_risk_thresh: float = 0.3, high_value_thresh: float = 0.7) -> pd.DataFrame:
    """基于分群与复购概率的营销策略建议

    - 高价值活跃用户：概率高（≥high_value_thresh）→ 会员积分/新品优先体验；概率低 → 唤醒礼包
    - 成长用户：概率中等 → 阶梯优惠券/捆绑促销；概率低 → 首购/复购引导券
    - 沉睡用户：概率低（<high_risk_thresh）→ 大力度折扣券/限时活动；概率高 → 关怀短信与复购奖励
    """
    merged = segments.merge(probs, on="CustomerID", how="left").fillna({"prob_30d": 0.0})
    def decide(row):
        p = row["prob_30d"]
        lab = row["label"]
        if lab == "高价值活跃用户":
            return "会员权益升级、专属新品试用；低概率时发唤醒礼包"
        if lab == "成长用户":
            return "阶梯优惠券与捆绑促销，提高复购频次"
        if lab == "沉睡用户":
            return "限时大力度折扣券与召回活动，配合短信关怀"
        # 其他标签
        if p < high_risk_thresh:
            return "折扣券+召回活动，提高复购概率"
        elif p >= high_value_thresh:
            return "会员激励与新品优先体验"
        else:
            return "常规优惠与个性化推荐"
    merged["strategy"] = merged.apply(decide, axis=1)
    return merged[["CustomerID", "label", "prob_30d", "strategy"]]


if __name__ == "__main__":
    # 加载 Excel 数据
    open_excel_path = r"./Online Retail.xlsx"
    df = pd.read_excel(open_excel_path)


    """数据预处理与特征工程"""
    # 数据清洗
    df_clean = clean_data(df)
    # 用户级特征（RFM 与时段偏好）
    user_features = compute_user_features(df_clean)
    # 商品级特征（频次与平均订单量）
    product_features = compute_product_features(df_clean)


    """关联规则与用户分群"""
    # 关联规则挖掘（前 5 条规则，最小支持度 0.01）
    rules_top5 = mine_association_rules(df_clean, min_support=0.01, top_n=5)
    # 基于 RFM 的 K-Means 分群并可视化
    segments = cluster_users_rfm(user_features, n_clusters=3, save_prefix="用户分群")


    """复购行为预测与营销策略"""
    # 预测 30 天复购概率（单函数完成打标、训练与预测）
    repurchase_probs, importances = predict_repurchase_within_30d(df_clean, user_features, days=30, n_estimators=100, sample_ratio=0.7)
    # 特征重要性排名
    importance_df = feature_importance_report(importances)
    # 分群驱动营销策略
    strategies = recommend_marketing_strategies(segments, repurchase_probs, high_risk_thresh=0.3, high_value_thresh=0.7)

    # 保存结果为中文命名文件
    df_clean.to_csv("1-1清洗数据.csv", index=False, encoding="utf-8-sig")
    user_features.to_csv("1-2用户特征.csv", index=False, encoding="utf-8-sig")
    product_features.to_csv("1-3商品特征.csv", index=False, encoding="utf-8-sig")
    rules_top5.to_csv("2-1关联规则Top5.csv", index=False, encoding="utf-8-sig")
    segments.to_csv("2-2用户分群.csv", index=False, encoding="utf-8-sig")
    repurchase_probs.to_csv("3-1复购概率.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv("3-2特征重要性.csv", index=False, encoding="utf-8-sig")
    strategies.to_csv("3-3营销策略.csv", index=False, encoding="utf-8-sig")

    ok = visualize_rules_heatmap(rules_top5, save_path="关联规则_热力图.png")
    if not ok:
        visualize_rules_network(rules_top5, save_path="关联规则_网络图.png")
    plot_feature_importance_bar(importance_df, save_path="特征重要性_条形图.png")
    plot_confusion_matrix_from_model(df_clean, user_features, days=30, n_estimators=100, sample_ratio=0.7, save_path="复购预测_混淆矩阵.png")

