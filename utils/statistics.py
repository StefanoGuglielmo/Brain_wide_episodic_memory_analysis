from scipy.linalg import orthogonal_procrustes
import numpy as np
import pandas as pd

def perform_pls_df(pls_data, n_bootstrap=1000, procrustes=False, threshold=2.58, random_state=42):

    rng = np.random.default_rng(random_state)

    encoded_data = pd.get_dummies(pls_data, columns=["group"])
    n_groups = pls_data["group"].nunique()

    x_data = encoded_data.drop(columns=["subject", "sex"] + list(encoded_data.columns[-n_groups:]))
    y_data = encoded_data.iloc[:, -n_groups:]

    x_matrix = x_data.to_numpy()
    y_matrix = y_data.to_numpy()

    def compute_pls(x, y):

        group_weights = np.diagflat(
            ((np.ones((1, y.shape[0])) @ y) ** (-1)).ravel()
        )

        cross_covariance = group_weights @ y.T @ x

        centered_covariance = (
            cross_covariance
            - np.ones((cross_covariance.shape[0], 1))
            @ ((np.ones((1, cross_covariance.shape[0])) @ cross_covariance)
               / cross_covariance.shape[0])
        )

        u, singular_values, v = np.linalg.svd(
            centered_covariance,
            full_matrices=False,
        )

        return u, singular_values, v

    def bootstrap_pls(x, y, reference_v, reference_u):

        n_subjects = x.shape[0]
        bootstrap_loadings = np.zeros(
            (n_bootstrap,) + reference_v.shape
        )

        for iteration in range(n_bootstrap):

            while True:
                indices = rng.integers(0, n_subjects, n_subjects)

                x_bootstrap = x[indices]
                y_bootstrap = y[indices]

                if not np.any(np.all(y_bootstrap == 0, axis=0)):
                    break

            boot_u, _, boot_v = compute_pls(
                x_bootstrap,
                y_bootstrap,
            )

            if procrustes:
                _, rotation = orthogonal_procrustes(
                    boot_v.T,
                    reference_v.T,
                )
                boot_v = (rotation @ boot_v.T).T

            bootstrap_loadings[iteration] = boot_v

        return bootstrap_loadings.std(axis=0)

    pls_u, _, pls_v = compute_pls(
        x_matrix,
        y_matrix,
    )

    loading_error = bootstrap_pls(
        x_matrix,
        y_matrix,
        pls_v,
        pls_u,
    )

    region_names = list(x_data.columns)

    bootstrap_ratio = pd.DataFrame(
        pls_v / loading_error,
        columns=region_names,
    )

    significant_regions = bootstrap_ratio.iloc[0][
        bootstrap_ratio.iloc[0] > threshold
    ]

    return bootstrap_ratio, significant_regions



def prepare_pls_dataframe(mid_level_data, group_prefixes, sex="M"):


    selected_subjects = [
        column
        for column in mid_level_data.columns
        if any(column.startswith(prefix) for prefix in group_prefixes)
    ]

    subject_data = mid_level_data[selected_subjects].T
    subject_data.index.name = None
    subject_data.reset_index(drop=True, inplace=True)

    subjects = []
    groups = []

    for prefix in group_prefixes:
        matching_subjects = [
            column
            for column in mid_level_data.columns
            if column.startswith(prefix)
        ]

        for index in range(1, len(matching_subjects) + 1):
            subjects.append(f"{index}{prefix[0]}")
            groups.append(prefix)

    subject_data["subject"] = subjects
    subject_data["sex"] = sex
    subject_data["group"] = groups

    return subject_data[
        ["subject", "sex", "group"] + list(mid_level_data.index)
    ]


import scikit_posthocs as sp
from scipy.stats import kruskal


def significance_label(p):

    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"



import scikit_posthocs as sp
from scipy.stats import kruskal

def kruskal_wallis_dunn(
    df,
    metric_col="metric",
    group_col="network",
    value_col="value",
    groups=("CNTX", "OCT", "OPCRT"),
    p_adjust="bonferroni",
):

    results_kw = []
    results_dunn = []

    comparisons = [
        (groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    ]

    for metric in df[metric_col].unique():

        df_sub = df[df[metric_col] == metric].copy()

        group_values = [
            df_sub.loc[
                df_sub[group_col] == group,
                value_col
            ].dropna()
            for group in groups
        ]

        H, p = kruskal(*group_values)

        results_kw.append({
            "metric": metric,
            "H": H,
            "p": p
        })

        dunn = sp.posthoc_dunn(
            df_sub,
            val_col=value_col,
            group_col=group_col,
            p_adjust=p_adjust
        )

        for group1, group2 in comparisons:

            p_value = dunn.loc[group1, group2]

            results_dunn.append({
                "metric": metric,
                "comparison": f"{group1} vs {group2}",
                "p": p_value,
                "significance": significance_label(p_value)
            })

    results_kw = pd.DataFrame(results_kw)
    results_dunn = pd.DataFrame(results_dunn)

    return results_kw, results_dunn