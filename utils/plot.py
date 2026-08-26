import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
import igraph as ig
from adjustText import adjust_text


def plot_covariance_ellipsoid(
    ax,
    center,
    covariance,
    color,
    n_std=2.5,
    resolution=60,
):

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = np.sqrt(eigenvalues) * n_std

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack((x, y, z), axis=-1)

    sphere *= radii
    ellipsoid = sphere @ eigenvectors.T
    ellipsoid += center

    ax.plot_surface(
        ellipsoid[..., 0],
        ellipsoid[..., 1],
        ellipsoid[..., 2],
        color=color,
        alpha=0.20,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=True,
        rasterized=True,
    )



def plot_grouped_heatmap(
    data,
    group_sizes,
    figsize=(14, 14),
    cmap="vlag",
    center=1,
    colorbar_label="",
    font_family="Arial",
):

    plt.rcParams["font.family"] = font_family

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=center,
        square=True,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={
            "fraction": 0.016,
            "pad": 0.02,
        },
    )

    ax.invert_yaxis()

    group_sizes_array = np.array(list(group_sizes.values()))

    group_centers = np.cumsum(group_sizes_array) - group_sizes_array / 2

    ax.set_yticks(group_centers)
    ax.set_yticklabels(
        group_sizes.keys(),
        rotation=0,
        fontsize=12,
        va="center",
    )

    group_boundaries = np.cumsum(group_sizes_array)[:-1]

    for boundary in group_boundaries:
        ax.hlines(
            boundary,
            *ax.get_xlim(),
            color="black",
            linewidth=0.4,
        )

    if colorbar_label:
        ax.collections[0].colorbar.set_label(
            colorbar_label,
            fontsize=12,
            rotation=90,
            labelpad=15,
        )

    ax.set_xlabel("")

    return fig, ax



def get_colors_for_regions(region_names, colors_df, default_color="gray"):

    colors = []

    for region in region_names:
        match = colors_df.loc[
            colors_df["target"] == region,
            "color",
        ]

        colors.append(
            match.iloc[0]
            if not match.empty
            else default_color
        )

    return colors



def plot_vpd(ax, vpd, title, colors_df, width=0.8, threshold=2.58):

    regions = vpd.columns
    values = vpd.iloc[0].values

    colors = get_colors_for_regions(
        regions,
        colors_df,
    )

    ax.bar(
        regions,
        values,
        color=colors,
        width=width,
    )

    ax.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"±{threshold}",
    )

    ax.axhline(
        -threshold,
        color="red",
        linestyle="--",
    )

    ax.set_title(title)
    ax.set_ylabel("Salience")

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(
        regions,
        rotation=90,
        fontsize=20,
    )



def plot_sig_regions(ax, significant_regions, title, colors_df, width=0.8):

    regions = significant_regions.index
    values = significant_regions.values

    colors = get_colors_for_regions(
        regions,
        colors_df,
    )

    ax.bar(
        regions,
        values,
        color=colors,
        width=width,
    )

    ax.set_title(
        f"Significant Regions: {title}"
    )

    ax.set_ylabel("Bootstrap ratio")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(
        regions,
        rotation=90,
    )


def save_figure(
    figure,
    filename,
    output_dir,
    dpi=900,
    rasterize=False,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()

    if rasterize and filename.lower().endswith(".eps"):
        for axis in figure.axes:
            for artist in axis.get_children():
                if hasattr(artist, "set_rasterized"):
                    artist.set_rasterized(True)

    figure.savefig(
        output_dir / filename,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_raw_psth(
    data,
    region,
    palette,
    label_names,
    ax=None,
):
    created_figure = ax is None

    if created_figure:
        _, ax = plt.subplots(figsize=(5, 4))

    for label in sorted(data["label"].unique()):
        label_data = data[data["label"] == label]

        bin_width = (
            np.diff(label_data["psth_x"]).mean()
            if len(label_data) > 1
            else 0.1
        )

        ax.bar(
            label_data["psth_x"],
            label_data["psth_raw"],
            width=bin_width,
            color=palette[label],
            alpha=0.5,
            label=label_names[label],
        )

    x_values = np.asarray(data["psth_x"])
    y_values = np.asarray(data["psth_raw"])

    ax.set(
        xlabel="Time (s)",
        ylabel="Firing Rate (Hz)",
        xlim=(x_values.min(), x_values.max()),
        ylim=(0, y_values.max() * 1.1),
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, linestyle="--", linewidth=1.5, color="black")

    return ax



def plot_normalized_psth(
    data,
    palette,
    label_names,
    ax=None,
):
    created_figure = ax is None

    if created_figure:
        _, ax = plt.subplots(figsize=(5, 4))

    for label in sorted(data["label"].unique()):
        label_data = data[data["label"] == label]

        grouped = (
            label_data
            .groupby("psth_x")["psth_norm"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        x = grouped["psth_x"]
        mean = grouped["mean"]
        sem = grouped["sem"]

        ax.plot(
            x,
            mean,
            color=palette[label],
            label=label_names[label],
        )

        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=palette[label],
            alpha=0.2,
        )

    x_values = np.asarray(data["psth_x"])
    y_values = np.asarray(data["psth_norm"])

    ax.set(
        xlabel="Time (s)",
        ylabel="Normalized Firing Rate",
        xlim=(x_values.min(), x_values.max()),
        ylim=(0, y_values.max() * 1.1),
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.legend(frameon=False, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, linestyle="--", linewidth=1.5, color="black")

    return ax



def plot_psth_heatmap(
    data,
    ax=None,
):
    created_figure = ax is None

    if created_figure:
        _, ax = plt.subplots(figsize=(6, 4))

    data = data.copy()

    data["peak_time_idx"] = data["psth_norm"].apply(np.argmax)

    data = data.sort_values(
        ["label", "peak_time_idx"]
    )

    matrix = np.stack(data["psth_raw"].values)

    x = np.asarray(data.iloc[0]["psth_x"])

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        extent=[
            x.min(),
            x.max(),
            0,
            len(data),
        ],
        origin="upper",
    )

    ax.axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.2,
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Mouse #",
        xlim=(x.min(), x.max()),
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.spines[["top", "right"]].set_visible(False)

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(
        "Firing Rate (Hz)",
        fontsize=15,
    )

    return ax



def plot_unit_raster(
    spike_times,
    epochs,
    unit_id,
    region,
    mouse_id,
    window_before=3.0,
    window_after=3.0,
    spike_color="black",
    spike_size=8,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    labels = sorted(epochs["label"].unique())

    y_offset = 0

    for label in labels:

        label_epochs = (
            epochs[epochs["label"] == label]
            .reset_index(drop=True)
        )

        for _, epoch in label_epochs.iterrows():

            exploration_start = (
                epoch["start_exploration_s"]
            )

            window_start = (
                exploration_start - window_before
            )

            window_end = (
                exploration_start + window_after
            )

            spikes_in_window = spike_times[
                (spike_times >= window_start)
                & (spike_times <= window_end)
            ]

            spikes_relative = (
                spikes_in_window - exploration_start
            )

            ax.scatter(
                spikes_relative,
                np.full(
                    len(spikes_relative),
                    y_offset,
                ),
                s=spike_size,
                color=spike_color,
                marker="|",
                linewidths=1,
            )

            y_offset += 1

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
    )

    ax.set_xlim(
        -window_before,
        window_after,
    )

    ax.set_xlabel(
        "Time from exploration start (s)"
    )

    ax.set_ylabel("Epoch")

    ax.set_title(
        f"{mouse_id} | {region} | Unit {unit_id}"
    )

    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_unit_classification(
    units_df,
    region_colors,
    spike_width_threshold=0.3,
    firing_rate_threshold=10,
    output_file=None,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    for region, color in region_colors.items():
        region_data = units_df[
            units_df["region"] == region
        ]

        ax.scatter(
            region_data["spike_width_ms"],
            region_data["firing_rate"],
            color=color,
            label=region,
            s=45,
        )

    ax.axvline(
        spike_width_threshold,
        linestyle="--",
        color="black",
        linewidth=1.5,
    )

    ax.axhline(
        firing_rate_threshold,
        linestyle="--",
        color="black",
        linewidth=1.5,
    )

    ax.set(
        xlabel="Spike width (ms)",
        ylabel="Firing rate (Hz)",
        xlim=(0.2, 0.8),
        ylim=(0, 12),
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    if output_file is not None:
        fig.savefig(
            output_file,
            bbox_inches="tight",
        )

    return fig, ax



def plot_connectivity_network(
    graph,
    ax=None,
    communities={0: ["ACA", "PL", "ILA", "ORB"], 1: ["RSP", "POST"]},
    layout_seed=42,
    jitter_seed=134,
    rotation_degrees=30,
    node_size=300,
    edge_scale=2000,
    edge_min_width=0.5,
    label_fontsize=15,
    figsize=(7, 5),
):
    community_assignment = {
        region: community
        for community, regions in communities.items()
        for region in regions
    }

    community_graph = nx.cycle_graph(len(communities))
    community_positions = nx.spring_layout(
        community_graph, scale=1.2, seed=layout_seed
    )

    rng = np.random.default_rng(jitter_seed)
    positions = {}

    for node in graph.nodes():
        community = community_assignment.get(node)
        if community is None:
            continue

        center = community_positions[community]
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.1, 0.5)

        positions[node] = center + radius * np.array(
            [np.cos(angle), np.sin(angle)]
        )

    angle_rad = np.deg2rad(rotation_degrees)
    rotation = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    positions = {key: rotation @ value for key, value in positions.items()}

    community_colors = {
        community_id: plt.cm.tab10(i)
        for i, community_id in enumerate(communities)
    }

    node_colors = [
        community_colors.get(
            community_assignment.get(node), (0.8, 0.8, 0.8, 0.3)
        )
        for node in graph.nodes()
    ]

    edge_weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    edge_widths = [w * edge_scale + edge_min_width for w in edge_weights]

    edge_colors = [
        community_colors.get(
            community_assignment.get(source), (0.5, 0.5, 0.5, 0.3)
        )
        for source, _ in graph.edges()
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nx.draw_networkx_nodes(
        graph, positions, node_color=node_colors, node_size=node_size, ax=ax
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        connectionstyle="arc3,rad=0.2",
        ax=ax,
    )

    labels = [
        ax.text(x, y, node, fontsize=label_fontsize, ha="left", va="top")
        for node, (x, y) in positions.items()
    ]

    adjust_text(
        labels,
        ax=ax,
        force_points=0.5,
        force_text=0.3,
        expand_points=(1.2, 1.4),
    )

    ax.axis("off")

    return fig, ax