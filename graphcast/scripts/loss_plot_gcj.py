import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import defaultdict

# Group and merge records with the same label
merged_records = {}
merged_loss_curves = defaultdict(lambda: {"total_flops": [], "losses": [], "params": None})


def compute_graphcast_total_flops_corrected(mesh_level, gnn_msg_steps, latent_size):
    """
    Computes total FLOPs for the full GraphCast forward pass using the corrected parameter expressions,
    where:
    - Grid2Mesh and Mesh2Grid GNNs: total_params = 12 * lat + 9 * lat^2 (already includes norm)
    - Mesh GNN: total_params = step * (8 * lat + 7 * lat^2) (already includes norm)
    - Parameters are applied to one of the three: grid node, multi-mesh edge, or multi-mesh node

    Parameters:
    - mesh_level (int): Mesh refinement level (0 to 6)
    - gnn_msg_steps (int): Number of message passing steps in mesh GNN
    - latent_size (int): Latent feature dimension (lat)

    Returns:
    - total_flops (int): Total floating point operations (FLOPs)
    """

    # Mesh structure tables
    num_nodes_by_level = [12, 42, 162, 642, 2562, 10242, 40962]
    num_multilevel_edges_by_level = [60, 300, 1260, 5100, 20460, 81900, 327660]

    if not (0 <= mesh_level <= 6):
        raise ValueError("mesh_level must be between 0 and 6")

    num_mesh_nodes = num_nodes_by_level[mesh_level]
    num_edges = num_multilevel_edges_by_level[mesh_level]
    num_grid_nodes = 721*1440  # Fixed grid resolution

    # -----------------------------
    # Component 1: Grid2Mesh
    # -----------------------------
    params_grid2mesh_grid = 4 * latent_size + 2 * latent_size ** 2
    params_grid2mesh_mesh = 4 * latent_size + 3 * latent_size ** 2
    params_grid2mesh_edge = 4 * latent_size + 4 * latent_size ** 2
    grid2mesh_flops = 2 * (params_grid2mesh_mesh * num_mesh_nodes + \
        params_grid2mesh_grid * num_grid_nodes + params_grid2mesh_edge * num_edges)

    # -----------------------------
    # Component 2: Mesh GNN
    # -----------------------------
    params_mesh_gnn_node = gnn_msg_steps * (4 * latent_size + 3 * latent_size ** 2)
    params_mesh_gnn_edge = gnn_msg_steps * (4 * latent_size + 4 * latent_size ** 2)

    mesh_gnn_flops = 2 * (params_mesh_gnn_node * num_mesh_nodes + params_mesh_gnn_edge * num_edges)

    # -----------------------------
    # Component 3: Mesh2Grid
    # -----------------------------
    params_mesh2grid_grid = 4 * latent_size + 3 * latent_size ** 2
    params_mesh2grid_mesh = 4 * latent_size + 2 * latent_size ** 2
    params_mesh2grid_edge = 12 * latent_size + 4 * latent_size ** 2
    mesh2grid_flops = 2 * (params_mesh2grid_mesh * num_mesh_nodes + \
        params_mesh2grid_grid * num_grid_nodes + params_mesh2grid_edge * num_edges)

    # -----------------------------
    # Total
    # -----------------------------
    total_flops = grid2mesh_flops + mesh_gnn_flops + mesh2grid_flops
    total_flops *= 3 # back propagation
    return total_flops

def calculate_nonembedding_parameter(lat, gstep):
    """Calculate the number of non-embedding parameters for a model"""
    return 24*lat + 18 * lat ** 2 + gstep * (8 * lat + 7 * lat ** 2)

os.chdir("wandb")
# Scan files
file_pattern = re.compile(r".*_n(\d+)_mesh(\d+)_lat(\d+)_gstep(\d+).*")
log_files = [f for f in os.listdir() if file_pattern.match(f)]
log_files = sorted(log_files)

raw_records = []
records = []  # Store best_loss + FLOPs points
loss_curves = []  # Store each loss curve and its FLOPs info

# Iterate through files
for filename in log_files:
    match = file_pattern.match(filename)
    if not match:
        continue

    n_nodes = int(match.group(1))
    if n_nodes != 4:
        continue
    
    mesh_level = int(match.group(2))

    if mesh_level != 5:
        continue
    latent_size = int(match.group(3))
    if latent_size < 128:
        continue
    gnn_step = int(match.group(4))
    if gnn_step % 2 == 1:
        continue
    label = f"n{n_nodes}_mesh{mesh_level}_lat{latent_size}_gstep{gnn_step}"

    numparams = calculate_nonembedding_parameter(latent_size, gnn_step)
    print("numparams", numparams)
    
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except:
        print(filename)
        try:
            with open(filename+f"/files/loss_log_{label}.json", "r") as f:
                data = json.load(f)
        except:
            continue

    step_loss_pairs = [(item["val/global_step"], item["val/loss"]) for item in data 
                   if "val/global_step" in item and "val/loss" in item]
    step_loss_pairs.sort()  # Sort by step
    losses = [item["val/loss"] for item in data if "val/loss" in item]
    # step = [item["val/global_step"] for item in data if "val/global_step" in item]
    nstep = len(losses)
    if nstep == 0:
        continue

    # # Smoothing (adjustable window)
    # window = 10
    # if len(losses) > window:
    #     losses = np.convolve(losses, np.ones(window)/window, mode='valid')

    best_loss = min(losses)
    flops_per_step = compute_graphcast_total_flops_corrected(
        mesh_level=mesh_level,
        latent_size=latent_size,
        gnn_msg_steps=gnn_step
    )
    flops = flops_per_step * nstep

    raw_records.append({
        "step_loss_pairs": step_loss_pairs,
        "label": label,
        "best_loss": best_loss,
        "params": numparams,  # Store parameter count for coloring
        "flops_per_step": flops_per_step,
        "flops": flops
    })

for rec in raw_records:
    label = rec["label"]
    if label in merged_records:
        # Merge by keeping the lowest loss and summing the flops
        merged_records[label]["best_loss"] = min(merged_records[label]["best_loss"], rec["best_loss"])
        merged_records[label]["flops"] = max(merged_records[label]["flops"],rec["flops"])
    else:
        merged_records[label] = {
            "best_loss": rec["best_loss"],
            "flops": rec["flops"],
            "label": label,
            "params": rec["params"]
        }

for curve in raw_records:
    label = curve["label"]
    if label not in merged_loss_curves:
        merged_loss_curves[label] = {
            "flops_per_step": curve["flops_per_step"],
            "step_loss_pairs": curve["step_loss_pairs"],
            "params": curve["params"],
            "label": label
        }
    else:
        # merged_loss_curves[label]["flops_per_step"].extend(curve["flops_per_step"])
        merged_loss_curves[label]["step_loss_pairs"].extend(curve["step_loss_pairs"])
        merged_loss_curves[label]["params"] = curve["params"]
        merged_loss_curves[label]["step_loss_pairs"].sort()
        merged_loss_curves[label]["label"] = label 


    # loss_curves.append({
    #     "label": label,
    #     "total_flops": total_flops[10:],
    #     "losses": losses[10:],
    #     "params": numparams  # Store parameter count for coloring
    # })

os.chdir("..")

# Get parameter ranges for color mapping
all_params = [rec["params"] for rec in merged_records.values()]
min_params = min(all_params)
max_params = max(all_params)
norm = Normalize(vmin=min_params, vmax=max_params)
cmap = cm.viridis
# cmap = cm.viridis_r  # Use reversed viridis colormap so darker colors = more parameters

# Chart 1: Scaling Law with parameter-based coloring
plt.figure(figsize=(14, 8))

# Create scatter plot with colors based on parameter count
for rec in merged_records.values():
    color = cmap(norm(rec["params"]))
    plt.scatter(rec["flops"], rec["best_loss"], 
                color=color, s=100, alpha=0.8, 
                edgecolors='black', linewidths=0.5)
    
    # Improved label positioning
    # Adjust text position to avoid overlap with points
    # Place labels above points with small offset
    x_pos = rec["flops"]
    y_pos = rec["best_loss"] * 0.999  # Position slightly below the point
    
    plt.annotate(rec["label"], 
                xy=(x_pos, rec["best_loss"]),
                xytext=(x_pos, y_pos),
                fontsize=6,
                ha='center',
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Number of Parameters')

# Format the colorbar ticks to show actual parameter counts
cbar.ax.set_yticklabels([f'{int(t):,}' for t in cbar.get_ticks()])

plt.xlabel("Compute (FLOPs)")
plt.ylabel("Best validation loss")
plt.xscale("log")
plt.title("Scaling Law: FLOPs vs Best Loss (GraphCast)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_scaling_law_graphcast_colored.png", dpi=300)

# Chart 2: Loss curves with parameter-based coloring
plt.figure(figsize=(14, 8))

# Create line plots with colors based on parameter count
for curve in merged_loss_curves.values():
    color = cmap(norm(curve["params"]))
    step_loss_pairs = curve["step_loss_pairs"]
    step, losses = zip(*step_loss_pairs)
    step = np.array(step)
    flops_by_step = step * curve["flops_per_step"]
    plt.plot(flops_by_step, losses, 
             color=color, linewidth=2, alpha=0.8, 
             label=f"{curve['label']}")

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Number of Parameters')

# Format the colorbar ticks to show actual parameter counts
cbar.ax.set_yticklabels([f'{int(t):,}' for t in cbar.get_ticks()])

plt.xlabel("Total FLOPs")
plt.ylabel("Validation loss")
plt.xscale("log")
# plt.xlim(1*10**12.9,1*10**14.2)
plt.title("Loss Curve vs Total Compute")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig("loss_scaling_law_flops_curve_graphcast_colored.png", dpi=300)