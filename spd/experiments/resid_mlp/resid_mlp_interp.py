from typing import Any, cast

import einops
import fire
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import Config
from spd.experiments.resid_mlp.models import ResidualMLP
from spd.experiments.tms.models import TMSModel
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import plot_causal_importance_vals
from spd.utils.general_utils import get_device, set_seed
from spd.utils.run_utils import get_output_dir


def extract_ci_val_figures(run_id: str, input_magnitude: float = 0.75) -> dict[str, Any]:
    """Extract causal importances from a single run.

    Args:
        run_id: Wandb run ID to load model from
        input_magnitude: Magnitude of input features for causal importances plotting

    Returns:
        Dictionary containing causal importances data and metadata
    """
    model, config, _ = ComponentModel.from_pretrained(run_id)
    target_model = model.model
    assert isinstance(target_model, ResidualMLP | TMSModel), (
        "Target model must be a ResidualMLP or TMSModel"
    )
    n_features = target_model.config.n_features

    # Get components and gates from model
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(Gate | GateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    # Assume no position dimension
    batch_shape = (1, n_features)

    # Get device from model
    device = next(model.parameters()).device

    # Get mask values without plotting regular masks
    figures, all_perm_indices_ci_vals = plot_causal_importance_vals(
        model=model,
        components=components,
        gates=gates,
        batch_shape=batch_shape,
        device=device,
        input_magnitude=input_magnitude,
        plot_raw_cis=False,
        sigmoid_type=config.sigmoid_type,
    )

    return {
        "figures": figures,
        "all_perm_indices_ci_vals": all_perm_indices_ci_vals,
        "config": config,
        "components": components,
        "n_features": n_features,
    }


def plot_increasing_importance_minimality_coeff_ci_vals(
    run_ids: list[str], input_magnitude: float = 0.75, best_idx: list[int] | None = None
) -> plt.Figure:
    """Plot increasing importance minimality coeff for multiple runs in a combined figure.

    Args:
        run_ids: List of wandb run IDs to load models from
        input_magnitude: Magnitude of input features for causal importances plotting
        best_idx: List of indices indicating which runs are the best (for highlighting)

    Returns:
        Combined figure with causal importances from all runs
    """
    all_mask_data = {}
    all_components = []

    # Collect causal importances from all runs
    for run_id in run_ids:
        print(f"Loading model from {run_id}")

        # Extract causal importances using helper function
        extraction_result = extract_ci_val_figures(run_id, input_magnitude)
        figures = extraction_result["figures"]
        config = extraction_result["config"]
        assert isinstance(config, Config)

        # Extract causal importances_upper_leaky data from the figure
        ci_vals_fig = figures["causal_importances_upper_leaky"]

        # Get mask data from the figure axes
        mask_data = {}
        for ax in ci_vals_fig.axes[:-1]:  # Skip colorbar axis
            # Get the image data from the axis
            images = ax.get_images()
            if images:
                data = images[0].get_array()
                # Get component name from axis title
                title = ax.get_title()
                component_name = title.split(" (")[0]  # Extract component name
                mask_data[component_name] = data

                # Track all unique component names
                if component_name not in all_components:
                    all_components.append(component_name)

        all_mask_data[run_id] = {
            "mask_data": mask_data,
            "importance_minimality_coeff": config.importance_minimality_coeff,
        }
        plt.close(ci_vals_fig)  # Close the individual figure

    # Create combined figure
    n_runs = len(run_ids)
    n_components = len(all_components)

    fig, axs = plt.subplots(
        n_components,
        n_runs,
        figsize=(5 * n_runs, 5 * n_components),
        constrained_layout=False,
        squeeze=False,
        dpi=300,
    )

    # Plot all masks
    images = []
    vmin, vmax = float("inf"), float("-inf")

    component_name_map = {"layers.0.mlp_in": "$W_{in}$", "layers.0.mlp_out": "$W_{out}$"}
    for col_idx, run_id in enumerate(run_ids):
        for row_idx, component_name in enumerate(all_components):
            ax = axs[row_idx, col_idx]

            assert component_name in all_mask_data[run_id]["mask_data"]
            mask_data = all_mask_data[run_id]["mask_data"][component_name]
            im = ax.matshow(mask_data, aspect="auto", cmap="Reds")
            images.append(im)

            # Track min/max for unified colorbar
            vmin = min(vmin, mask_data.min())
            vmax = max(vmax, mask_data.max())

            component_name = component_name_map.get(component_name, component_name)
            # Add labels
            if col_idx == 0:
                ax.set_ylabel(f"{component_name}\nInput feature index", fontsize=14)
            else:
                ax.set_ylabel("")

            if row_idx == n_components - 1:
                ax.set_xlabel("Subcomponent index", fontsize=14)

            # Increase tick label font sizes
            ax.tick_params(axis="both", which="major", labelsize=12)
            # Move x-axis ticks to bottom
            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position("bottom")

            if row_idx == 0:
                # Add importance_minimality_coeff as column title
                imp_coeff = all_mask_data[run_id]["importance_minimality_coeff"]
                title_text = rf"$\beta_3$={imp_coeff:.0e}"

                # Add "BEST" indicator if this is one of the best runs
                if best_idx is not None and col_idx in best_idx:
                    title_text += " (BEST)"

                ax.set_title(title_text, fontsize=12, pad=11)

    # Highlight best runs with visual distinctions
    if best_idx is not None:
        for best_col_idx in best_idx:
            if 0 <= best_col_idx < n_runs:
                # Add colored borders and background to all subplots in the best column
                for row_idx in range(n_components):
                    ax = axs[row_idx, best_col_idx]

                    # Add a thick colored border around the subplot
                    for spine in ax.spines.values():
                        spine.set_edgecolor("darkblue")
                        spine.set_linewidth(3)
                        spine.set_visible(True)

                    # Add a subtle background color
                    ax.set_facecolor("#f0f8ff")  # Very light blue background

                # Make the title more prominent for the best column
                if n_components > 0:  # Ensure we have at least one component
                    top_ax = axs[0, best_col_idx]
                    current_title = top_ax.get_title()
                    # Update title with bold formatting and color
                    top_ax.set_title(current_title, fontsize=16, color="darkblue", pad=18)

    # Add unified colorbar
    if images:
        # Clip values at upper bound only (values should already be >= 0)
        vmax_clip = 1

        # Clip all values at the maximum
        for im in images:
            data = im.get_array()
            if data is not None:
                # Clip values at 1
                clipped_data = np.minimum(data, vmax_clip)
                im.set_array(clipped_data)

        # Use a simple linear normalization from 0 to 1
        norm = colors.Normalize(vmin=0, vmax=vmax_clip)
        for im in images:
            im.set_norm(norm)

        # Add colorbar
        cbar = fig.colorbar(images[0], ax=axs.ravel().tolist(), label="Importance value")
        cbar.set_label("Importance value", fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    return fig


def feature_contribution_plot(
    ax: plt.Axes,
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"],
    n_layers: int,
    n_features: int,
    d_mlp: int,
    pre_labelled_neurons: dict[int, list[int]] | None = None,
    legend: bool = True,
) -> dict[int, list[int]]:
    diag_relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = relu_conns.cpu().detach()

    # Define colors for different layers
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    distinct_colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#1B9E77",  # teal
        "#D95F02",  # dark orange
        "#7570B3",  # slate blue
        "#66A61E",  # lime green
    ]

    # Add legend if there are two layers
    if n_layers == 2 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.legend(loc="upper right")
    # Add legend if there are three layers
    if n_layers == 3 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.scatter([], [], c="green", alpha=0.3, marker=".", label="Third MLP")
        ax.legend(loc="upper right")
    labelled_neurons: dict[int, list[int]] = {i: [] for i in range(n_features)}

    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(n_features):
        # Split points by layer and plot separately
        for layer in range(n_layers):
            ax.scatter(
                [i] * d_mlp,
                diag_relu_conns[layer, i, :],
                alpha=0.3,
                marker=".",
                c=layer_colors[layer],
            )
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for layer in range(n_layers):
            for j in range(d_mlp):
                # Label the neuron if it's in the pre-labelled set or if no pre-labelled set is provided
                # and the neuron has a connection strength greater than 0.1
                if (
                    pre_labelled_neurons is not None
                    and layer * d_mlp + j in pre_labelled_neurons[i]
                ) or (pre_labelled_neurons is None and diag_relu_conns[layer, i, j].item() > 0.1):
                    color_idx = j % len(distinct_colors)
                    # Make the neuron label alternate between left and right (-0.1, 0.1)
                    # Add 0.05 or -0.05 to the x coordinate to shift the label left or right
                    ax.text(
                        i,
                        diag_relu_conns[layer, i, j].item(),
                        str(layer * d_mlp + j),
                        color=distinct_colors[color_idx],
                        ha="left" if (len(labelled_neurons[i]) + 1) % 2 == 0 else "right",
                    )
                    labelled_neurons[i].append(layer * d_mlp + j)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_xlabel("Features")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return labelled_neurons


def compute_target_weight_neuron_contributions(
    target_model: ResidualMLP, n_features: int | None = None
) -> Float[Tensor, "n_layers n_features d_mlp"]:
    """Compute per-neuron contribution strengths for a *trained* ResidualMLP.

    The returned tensor has shape ``(n_layers, n_features, d_mlp)`` recording – for
    every hidden layer and every input feature – the *virtual* weight connecting
    that feature to each neuron after the ReLU (i.e. the product ``W_in * W_out``)
    as described in the original script. Only the first ``n_features`` are kept
    (or all features if ``n_features is None``).
    """

    n_features = target_model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = target_model.W_E
    assert torch.equal(W_E, target_model.W_U.T)

    # Stack mlp_in / mlp_out weights across layers so that einsums can broadcast
    W_in: Float[Tensor, "n_layers d_mlp d_embed"] = torch.stack(
        [layer.mlp_in.weight for layer in target_model.layers], dim=0
    )
    W_out: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(
        [layer.mlp_out.weight for layer in target_model.layers], dim=0
    )

    # Compute connection strengths
    in_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_features d_embed, n_layers d_mlp d_embed -> n_layers n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_layers n_features d_mlp, n_layers d_mlp n_features -> n_layers n_features d_mlp",
    )

    # Truncate to the first *n_features* for visualisation
    return relu_conns[:, :n_features, :]


def compute_spd_weight_neuron_contributions(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = None,
) -> Float[Tensor, "n_layers n_features C d_mlp"]:
    """Compute per-neuron contribution strengths for the *SPD* factorisation.

    Returns a tensor of shape ``(n_layers, n_features, C, d_mlp)`` where *C* is
    the number of sub-components in the SPD decomposition.
    """

    n_layers: int = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = target_model.W_E

    # Build the *virtual* input weight matrices (V @ U) for every layer
    W_in_spd: Float[Tensor, "n_layers d_embed C d_mlp"] = torch.stack(
        [
            einops.einsum(
                components[f"layers.{i}.mlp_in"].V,
                components[f"layers.{i}.mlp_in"].U,
                "d_embed C, C d_mlp -> d_embed C d_mlp",
            )
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Output weights for every layer
    W_out_spd: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(
        [components[f"layers.{i}.mlp_out"].weight for i in range(n_layers)],
        dim=0,
    )

    # Connection strengths
    in_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        W_E,
        W_in_spd,
        "n_features d_embed, n_layers d_embed C d_mlp -> n_layers n_features C d_mlp",
    )
    out_conns_spd: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out_spd,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        in_conns_spd,
        out_conns_spd,
        "n_layers n_features C d_mlp, n_layers d_mlp n_features -> n_layers n_features C d_mlp",
    )

    return relu_conns_spd[:, :n_features, :, :]


def plot_spd_feature_contributions_truncated(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = 50,
):
    n_layers = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features
    d_mlp = target_model.config.d_mlp

    # Assert that there are no biases
    assert not target_model.config.in_bias and not target_model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # --- Compute neuron contribution tensors ---
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_target_weight_neuron_contributions(
            target_model=target_model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            components=components,
            target_model=target_model,
            n_features=n_features,
        )
    )

    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))
    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    n_rows = 2
    fig1, axes1 = plt.subplots(n_rows, 1, figsize=(10, 7), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # pyright: ignore[reportCallIssue,reportArgumentType]

    labelled_neurons = feature_contribution_plot(
        ax=axes1[0],
        relu_conns=relu_conns,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        legend=True,
    )
    axes1[0].set_ylabel("Neuron contribution")
    axes1[0].set_xlabel(f"Input feature index (first {n_features} shown)")
    axes1[0].set_title("Target model")
    axes1[0].set_xticks(range(n_features))  # Ensure all xticks have labels

    feature_contribution_plot(
        ax=axes1[1],
        relu_conns=max_component_contributions,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        pre_labelled_neurons=labelled_neurons,
        legend=False,
    )
    axes1[1].set_ylabel("Neuron contribution")
    axes1[1].set_xlabel("Subcomponent index")
    axes1[1].set_title("Individual SPD subcomponents")
    axes1[1].set_xticks(range(n_features))

    # Set the same y-axis limits for both plots
    y_min = min(axes1[0].get_ylim()[0], axes1[1].get_ylim()[0])
    y_max = max(axes1[0].get_ylim()[1], axes1[1].get_ylim()[1])
    axes1[0].set_ylim(y_min, y_max)
    axes1[1].set_ylim(y_min, y_max)

    # Label the x axis with the subnets that have the largest neuron for each feature
    axes1[1].set_xticklabels(max_component_indices[0].tolist())  # Labels are the subnet indices

    return fig1


def plot_neuron_contribution_pairs(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = 50,
) -> plt.Figure:
    """Create a scatter plot comparing target model and SPD component neuron contributions.

    Each point represents a (component, input_feature, neuron) combination across all layers.
    X-axis: neuron contribution from the target model
    Y-axis: neuron contribution from the SPD component
    """
    n_layers = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features

    # Assert that there are no biases
    assert not target_model.config.in_bias and not target_model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # Compute neuron contribution tensors
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_target_weight_neuron_contributions(
            target_model=target_model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            components=components,
            target_model=target_model,
            n_features=n_features,
        )
    )

    # For each layer and feature, find the component with the largest max value over d_mlp
    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))

    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Define colors for different layers (same as in plot_spd_feature_contributions_truncated)
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points separately for each layer with different colors
    for layer in range(n_layers):
        x_values = relu_conns[layer].flatten().cpu().detach().numpy()
        y_values = max_component_contributions[layer].flatten().cpu().detach().numpy()

        layer_label = {0: "First MLP", 1: "Second MLP", 2: "Third MLP"}.get(layer, f"Layer {layer}")

        ax.scatter(
            x_values,
            y_values,
            alpha=0.3,
            s=10,
            color=layer_colors[layer],
            edgecolors="none",
            label=layer_label if n_layers > 1 else None,
        )

    # Add y=x reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k--", alpha=0.2, zorder=0, label="y=x")

    # Labels and title
    ax.set_xlabel("Target Model Neuron Contribution", fontsize=12)
    ax.set_ylabel("Subcomponent Neuron Contribution (Max Subcomponent)", fontsize=12)
    ax.set_title(
        f"{n_features} input features, {n_layers} layer{'s' if n_layers != 1 else ''}", fontsize=12
    )

    # Make axes equal and square
    ax.set_aspect("equal", adjustable="box")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend if there are multiple layers
    if n_layers > 1:
        ax.legend(loc="lower right")

    # Add some statistics to the plot
    # Calculate correlation for all points combined
    all_x = relu_conns.flatten().cpu().detach().numpy()
    all_y = max_component_contributions.flatten().cpu().detach().numpy()
    correlation = np.corrcoef(all_x, all_y)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def main():
    out_dir = get_output_dir() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = get_device()

    paths: list[str] = [
        "wandb:spd-resid-mlp/runs/ziro93xq",  # 1 layer
        "wandb:spd-resid-mlp/runs/wau744ht",  # 2 layer
        "wandb:spd-resid-mlp/runs/qqdugze1",  # 3 layer
    ]

    for path in paths:
        wandb_id = path.split("/")[-1]

        model, config, _ = ComponentModel.from_pretrained(path)
        model.to(device)

        target_model = model.model
        assert isinstance(target_model, ResidualMLP)
        n_layers = target_model.config.n_layers

        components: dict[str, LinearComponent] = {
            k.removeprefix("components.").replace("-", "."): v
            for k, v in model.components.items()
            if isinstance(v, LinearComponent)
        }

        fig = plot_spd_feature_contributions_truncated(
            components=components,
            target_model=target_model,
            n_features=10,
        )
        fig.savefig(
            out_dir / f"resid_mlp_weights_{n_layers}layers_{wandb_id}.png",
            bbox_inches="tight",
            dpi=500,
        )
        print(f"Saved figure to {out_dir / f'resid_mlp_weights_{n_layers}layers_{wandb_id}.png'}")

        # Generate and save neuron contribution pairs plot
        fig_pairs = plot_neuron_contribution_pairs(
            components=components,
            target_model=target_model,
            n_features=None,  # Using same number of features as above
        )
        fig_pairs.savefig(
            out_dir / f"neuron_contribution_pairs_{n_layers}layers_{wandb_id}.png",
            bbox_inches="tight",
            dpi=500,
        )
        fig_name = f"neuron_contribution_pairs_{n_layers}layers_{wandb_id}.png"
        print(f"Saved figure to {out_dir / fig_name}")

        # Define a title formatter for ResidualMLP component names
        def format_resid_mlp_title(mask_name: str) -> str:
            """Convert 'layers.X.mlp_in/out' to 'Layer Y - $W_{in/out}$' with LaTeX formatting."""
            parts = mask_name.split(".")
            if len(parts) == 3 and parts[0] == "layers":
                layer_idx = int(parts[1]) + 1  # Convert to 1-based indexing
                weight_type = parts[2]
                if weight_type == "mlp_in":
                    return f"Layer {layer_idx} - $W_{{in}}$"
                elif weight_type == "mlp_out":
                    return f"Layer {layer_idx} - $W_{{out}}$"
            return mask_name  # Fallback to original if pattern doesn't match

        # Generate and save causal importance plots
        gates: dict[str, Gate | GateMLP] = {
            k.removeprefix("gates.").replace("-", "."): cast(Gate | GateMLP, v)
            for k, v in model.gates.items()
        }
        batch_shape = (1, target_model.config.n_features)
        figs_causal = plot_causal_importance_vals(
            model=model,
            components=components,
            gates=gates,
            batch_shape=batch_shape,
            device=device,
            input_magnitude=0.75,
            plot_raw_cis=False,
            orientation="vertical",
            title_formatter=format_resid_mlp_title,
            sigmoid_type=config.sigmoid_type,
        )[0]
        figs_causal["causal_importances_upper_leaky"].savefig(
            out_dir / f"causal_importance_upper_leaky_{n_layers}layers_{wandb_id}.png",
            bbox_inches="tight",
            dpi=500,
        )
        fig_name = f"causal_importance_upper_leaky_{n_layers}layers_{wandb_id}.png"
        print(f"Saved figure to {out_dir / fig_name}")

        ##### Resid_mlp 1-layer varying sparsity ####
        run_ids = [
            "wandb:spd-resid-mlp/runs/xh0qlbkj",  # 1e-6
            "wandb:spd-resid-mlp/runs/kkpzirac",  # 3e-6
            "wandb:spd-resid-mlp/runs/ziro93xq",  # Best. 1e-5
            "wandb:spd-resid-mlp/runs/pnxu3d22",  # 1e-4
            "wandb:spd-resid-mlp/runs/aahzg3zu",  # 1e-3
        ]
        best_idx = [2]

        # Create and save the combined figure
        fig = plot_increasing_importance_minimality_coeff_ci_vals(run_ids, best_idx=best_idx)
        out_dir = get_output_dir()
        fig.savefig(
            out_dir / "resid_mlp_varying_importance_minimality_coeff_ci_vals.png",
            bbox_inches="tight",
            dpi=400,
        )
        fig_name = "resid_mlp_varying_importance_minimality_coeff_ci_vals.png"
        print(f"Saved figure to {out_dir / fig_name}")


if __name__ == "__main__":
    fire.Fire(main)
