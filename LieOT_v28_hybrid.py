import torch
import torch.optim as optim
import argparse
import os
import datetime
import json
import numpy as np
from scipy.spatial import cKDTree
import subprocess
import tempfile
from sklearn.neighbors import NearestNeighbors

# --- Constants ---
VDW_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'Default': 1.70
}
RESIDUE_CHARGES = {
    'ASP': {'OD1': -0.5, 'OD2': -0.5}, 'GLU': {'OE1': -0.5, 'OE2': -0.5},
    'LYS': {'NZ': 1.0}, 'ARG': {'NH1': 0.5, 'NH2': 0.5}, 'HIS': {'ND1': 0.5, 'NE2': 0.5}
}

# --- PDB Parsing ---
def parse_pdb(file_path, chain_id=None):
    coords, backbone_coords, radii, charges, lines = [], [], [], [], []
    seen_res_atoms = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                if chain_id and line[21].strip() != chain_id: continue
                if line[16] not in (' ', 'A'): continue
                
                atom_name = line[12:16].strip()
                res_seq = line[22:27]
                uid = (res_seq, atom_name)
                if uid in seen_res_atoms: continue
                seen_res_atoms.add(uid)

                element = line[76:78].strip().upper() or atom_name[0].upper()
                if element == 'H': continue

                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                lines.append(line)
                radii.append(VDW_RADII.get(element, VDW_RADII['Default']))
                res_name = line[17:20].strip()
                charges.append(RESIDUE_CHARGES.get(res_name, {}).get(atom_name, 0.0))
                if atom_name == "CA": 
                    backbone_coords.append(coords[-1])
    except FileNotFoundError:
        print(f"Error: PDB file not found at {file_path}")
        return [None]*5
    
    return (torch.tensor(coords, dtype=torch.float32), 
            torch.tensor(backbone_coords, dtype=torch.float32),
            torch.tensor(radii, dtype=torch.float32), 
            torch.tensor(charges, dtype=torch.float32), 
            lines)

# --- MSMS Surface Generation & Advanced Feature Detection (v23 inspired) ---
def _calculate_advanced_feature_score(points, normals, k=20):
    """Calculates a feature score for each point based on depth and wideness."""
    if torch.all(normals == 0): return torch.zeros(len(points))
    if len(points) <= k: k = len(points) - 1
    if k <= 0: return torch.zeros(len(points))

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points.cpu().numpy())
    _, indices = nbrs.kneighbors(points.cpu().numpy())
    
    feature_scores = []
    for i in range(len(points)):
        neighbor_points = points[indices[i][1:]]
        vectors_to_neighbors = neighbor_points - points[i]
        projections = torch.mv(vectors_to_neighbors, normals[i])
        depth_score = torch.mean(projections)
        wideness_score = torch.std(neighbor_points, dim=0).sum()
        feature_scores.append(depth_score * wideness_score)
        
    return torch.tensor(feature_scores, dtype=torch.float32)

def run_msms_and_get_features(coords, radii, msms_path, num_features, name, probe_radius=1.5, density=3.0):
    """
    Runs MSMS to generate a surface, then uses an advanced scoring algorithm
    (inspired by v23) to find concave and convex features.
    """
    print(f"  Running MSMS for {name} to generate surface...")
    with tempfile.TemporaryDirectory() as tmpdir:
        xyzr_path = os.path.join(tmpdir, f"{name}.xyzr")
        out_stem = os.path.join(tmpdir, f"{name}_out")
        vert_path = out_stem + ".vert"

        with open(xyzr_path, 'w') as f:
            for i in range(len(coords)):
                f.write(f"{coords[i][0]:.3f} {coords[i][1]:.3f} {coords[i][2]:.3f} {radii[i]:.3f}\n")

        command = [msms_path, "-if", xyzr_path, "-of", out_stem, "-probe_radius", str(probe_radius), "-density", str(density), "-no_header"]
        
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        except Exception as e:
            raise RuntimeError(f"MSMS failed for {name}: {e}")

        if not os.path.exists(vert_path) or os.path.getsize(vert_path) == 0:
            raise RuntimeError(f"MSMS did not produce a valid output file for {name}: {vert_path}")

        surface_points, surface_normals = [], []
        with open(vert_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9: continue
                surface_points.append([float(p) for p in parts[0:3]])
                surface_normals.append([float(p) for p in parts[3:6]])

    if not surface_points:
        raise ValueError(f"MSMS generated no surface vertices for {name}.")

    surface_points = torch.tensor(surface_points, dtype=torch.float32)
    surface_normals = torch.tensor(surface_normals, dtype=torch.float32)
    print(f"  MSMS generated {len(surface_points)} vertices for {name}.")

    print(f"  Calculating advanced feature scores for {name}...")
    feature_scores = _calculate_advanced_feature_score(surface_points, surface_normals)
    
    sorted_indices = torch.argsort(feature_scores)
    concave_indices = sorted_indices[:num_features]
    convex_indices = sorted_indices[-num_features:]
    
    print(f"  Identified {len(concave_indices)} concave and {len(convex_indices)} convex features.")
    return surface_points, surface_normals, concave_indices, convex_indices

# --- Geometric & Transformation Functions ---
def get_patch(surface_points, feature_index, radius):
    feature_point = surface_points[feature_index]
    dists = torch.norm(surface_points - feature_point, dim=1)
    patch_indices = torch.where(dists < radius)[0]
    return patch_indices

def get_rotation_matrix(v1, v2):
    v1, v2 = v1 / torch.norm(v1), v2 / torch.norm(v2)
    v = torch.cross(v1, v2)
    c = torch.dot(v1, v2)
    s = torch.norm(v)
    if s < 1e-6: return torch.eye(3)
    kmat = torch.zeros((3, 3)); kmat[0, 1], kmat[0, 2], kmat[1, 0], kmat[1, 2], kmat[2, 0], kmat[2, 1] = -v[2], v[1], v[2], -v[0], -v[1], v[0]
    return torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

def get_se3_exp_map(twist):
    v, w = twist[:3], twist[3:]
    theta = torch.norm(w)
    T = torch.eye(4, dtype=twist.dtype)
    if theta < 1e-6: 
        T[:3, 3] = v
        return T
    w_skew = torch.zeros((3, 3), dtype=twist.dtype)
    w_skew[0, 1], w_skew[0, 2], w_skew[1, 0], w_skew[1, 2], w_skew[2, 0], w_skew[2, 1] = -w[2], v[1], w[2], -w[0], -v[1], v[0]
    R = torch.eye(3, dtype=twist.dtype) + (torch.sin(theta)/theta)*w_skew + ((1-torch.cos(theta))/theta**2)*(w_skew@w_skew)
    V = torch.eye(3, dtype=twist.dtype) + ((1-torch.cos(theta))/theta**2)*w_skew + ((theta-torch.sin(theta))/theta**3)*(w_skew@w_skew)
    T[:3, :3], T[:3, 3] = R, V @ v
    return T

def transform_points(points, T):
    points_h = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    return (T @ points_h.T).T[:, :3]

# --- Scoring & Optimization ---
def sinkhorn_iterations(M, num_iters=10):
    P = M
    for _ in range(num_iters):
        P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
        P = P / (P.sum(dim=0, keepdim=True) + 1e-8)
    return P

def get_sinkhorn_alignment_score(points1, points2, d0=5.0, gamma=1.0, sinkhorn_iters=10, d_clash=1.0, clash_penalty_weight=10.0):
    if points1.shape[0] == 0 or points2.shape[0] == 0: return torch.tensor(0.0)
    dist_matrix = torch.cdist(points1, points2)
    
    reward_matrix = 1.0 / (1.0 + (dist_matrix / d0)**2)
    clash_penalty = -clash_penalty_weight * torch.relu(d_clash - dist_matrix)
    final_reward_matrix = reward_matrix + clash_penalty
    
    P = sinkhorn_iterations(torch.exp(gamma * final_reward_matrix), num_iters=sinkhorn_iters)
    score = torch.sum(P * final_reward_matrix) / len(points1)
    return score

def run_alignment_stage(initial_patch, target_patch, p):
    patch_centroid = initial_patch.mean(dim=0)
    centered_initial_patch = initial_patch - patch_centroid
    centered_target_patch = target_patch - patch_centroid

    twist = torch.zeros(6, requires_grad=True)
    optimizer = optim.AdamW([twist], lr=p['lr'])
    for _ in range(p['steps']):
        optimizer.zero_grad()
        T_local = get_se3_exp_map(twist)
        moved_patch_local = transform_points(centered_initial_patch, T_local)
        score = get_sinkhorn_alignment_score(
            moved_patch_local, centered_target_patch, 
            d0=p['d0'], gamma=p['gamma'], sinkhorn_iters=p['sinkhorn_iters'],
            d_clash=p['d_clash'], clash_penalty_weight=p['clash_penalty_weight']
        )
        loss = -score
        loss.backward()
        optimizer.step()
    
    T_local_final = get_se3_exp_map(twist.detach())

    T_translate_to_origin = torch.eye(4, dtype=torch.float32)
    T_translate_to_origin[:3, 3] = -patch_centroid
    T_translate_back = torch.eye(4, dtype=torch.float32)
    T_translate_back[:3, 3] = patch_centroid
    T_align_world = T_translate_back @ T_local_final @ T_translate_to_origin
    
    final_moved_patch = transform_points(initial_patch, T_align_world)
    final_score = get_sinkhorn_alignment_score(
        final_moved_patch, target_patch,
        d0=p['d0'], gamma=p['gamma'], sinkhorn_iters=p['sinkhorn_iters'],
        d_clash=p['d_clash'], clash_penalty_weight=p['clash_penalty_weight']
    )
    return T_align_world, final_score.item()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="LieOT_v28: MSMS surface + v23 feature scoring.")
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--reference", required=True, help="Reference PDB for RMSD calculation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--msms_path", default="/Users/huyue/LieOTdocking/msms_Arm64_2.6.1/msms", help="Path to the MSMS executable.")
    parser.add_argument("--num_features", type=int, default=50, help="Number of concave/convex features to sample.")
    parser.add_argument("--patch_radius", type=float, default=10.0)
    parser.add_argument("--probe_radius", type=float, default=1.5, help="Probe radius for MSMS.")
    parser.add_argument("--msms_density", type=float, default=3.0, help="Surface point density for MSMS.")
    parser.add_argument("--lr_align", type=float, default=1e-2)
    parser.add_argument("--steps_align", type=int, default=50)
    parser.add_argument("--align_d0", type=float, default=5.0)
    parser.add_argument("--align_gamma", type=float, default=2.0)
    parser.add_argument("--clash_threshold", type=float, default=1.0)
    parser.add_argument("--clash_penalty_weight", type=float, default=10.0)
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    print(f"--- LieOT_v28_hybrid --- Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print("Parameters:", json.dumps(vars(args), indent=2))

    l_all, l_bb, l_radii, _, _ = parse_pdb(args.ligand)
    r_all, r_bb, r_radii, _, _ = parse_pdb(args.receptor)
    if l_bb is None or r_bb is None: return

    ref_l_bb = parse_pdb(args.reference)[1][len(r_bb):]
    if ref_l_bb is None: return

    print("\nGenerating surfaces and features using MSMS + v23 scoring...")
    try:
        l_surf, l_normals, l_concave_idx, l_convex_idx = run_msms_and_get_features(
            l_all, l_radii, args.msms_path, args.num_features, "ligand", 
            probe_radius=args.probe_radius, density=args.msms_density)
        r_surf, r_normals, r_concave_idx, r_convex_idx = run_msms_and_get_features(
            r_all, r_radii, args.msms_path, args.num_features, "receptor",
            probe_radius=args.probe_radius, density=args.msms_density)
    except (RuntimeError, ValueError) as e:
        print(f"Error during feature generation: {e}")
        return
        
    initial_pairs = [(a,b) for a in l_concave_idx for b in r_convex_idx] + [(a,b) for a in l_convex_idx for b in r_concave_idx]
    print(f"Generated {len(initial_pairs)} initial patch pairs for alignment.")

    if not initial_pairs:
        print("Could not generate any initial pairs. Exiting.")
        return

    results = []
    for i, (idx_l, idx_r) in enumerate(initial_pairs):
        if (i+1) % 50 == 0: print(f"  Running trial {i+1}/{len(initial_pairs)}...")
        
        l_patch_indices = get_patch(l_surf, idx_l, args.patch_radius)
        r_patch_indices = get_patch(r_surf, idx_r, args.patch_radius)
        l_patch, r_patch = l_surf[l_patch_indices], r_surf[r_patch_indices]

        R_init = get_rotation_matrix(l_normals[idx_l], -r_normals[idx_r])
        t_init = r_surf[idx_r] - R_init @ l_surf[idx_l]
        T_init = torch.eye(4); T_init[:3, :3], T_init[:3, 3] = R_init, t_init
        l_patch_init = transform_points(l_patch, T_init)
        
        p_align = {
            'lr': args.lr_align, 'steps': args.steps_align, 'd0': args.align_d0,
            'gamma': args.align_gamma, 'sinkhorn_iters': 10,
            'd_clash': args.clash_threshold, 'clash_penalty_weight': args.clash_penalty_weight
        }
        T_align, alignment_score = run_alignment_stage(l_patch_init, r_patch, p_align)
        T_final = T_align @ T_init

        l_bb_transformed = transform_points(l_bb, T_final)
        current_rmsd = torch.sqrt(torch.mean((l_bb_transformed - ref_l_bb)**2)).item()
        
        results.append({'transform': T_final, 'score': alignment_score, 'rmsd': current_rmsd})

    # --- Analysis and Saving ---
    print("\n--- Final Analysis ---")
    if not results: 
        print("No results to analyze.")
        return
        
    results_by_score = sorted(results, key=lambda x: x['score'], reverse=True)
    results_by_rmsd = sorted(results, key=lambda x: x['rmsd'])

    print("\n--- Top 10 Candidates by Alignment Score ---")
    print("Rank | Score   | RMSD (Å)")
    print("--------------------------")
    for i, res in enumerate(results_by_score[:10]):
        print(f"{i+1:<4} | {res['score']:.5f} | {res['rmsd']:.4f}")

    print("\n--- Top 10 Candidates by RMSD ---")
    print("Rank | RMSD (Å) | Score")
    print("--------------------------")
    for i, res in enumerate(results_by_rmsd[:10]):
        print(f"{i+1:<4} | {res['rmsd']:.4f} | {res['score']:.5f}")

    best_res_by_score = results_by_score[0]
    best_score = best_res_by_score['score']
    rmsd_for_best_score = best_res_by_score['rmsd']
    
    min_rmsd_res = results_by_rmsd[0]
    min_rmsd_overall = min_rmsd_res['rmsd']
    score_for_min_rmsd = min_rmsd_res['score']
    best_rmsd_T = min_rmsd_res['transform']

    print(f"\n** Pose with Best Alignment Score ({best_score:.5f}) has RMSD: {rmsd_for_best_score:.4f} Å **")
    print(f"** Minimum RMSD found overall: {min_rmsd_overall:.4f} Å (with score: {score_for_min_rmsd:.5f}) **")

    output_filename = args.output.replace('.pdb', '_best_rmsd.pdb')
    print(f"\nSaving pose with best RMSD ({min_rmsd_overall:.4f} Å) to '{output_filename}'...")
    with open(output_filename, 'w') as f:
        remarks = (
            f"REMARK    Generated by LieOT_v28_hybrid.py\n"
            f"REMARK    Saved pose with the minimum RMSD.\n"
            f"REMARK    Minimum RMSD found: {min_rmsd_overall:.4f} Angstrom\n"
            f"REMARK    Score for this pose: {score_for_min_rmsd:.5f}\n"
            f"REMARK    --- (For comparison) ---\n"
            f"REMARK    Best alignment score in all candidates: {best_score:.5f}\n"
            f"REMARK    RMSD for best score pose: {rmsd_for_best_score:.4f} Angstrom\n"
            f"REMARK    Full params: {json.dumps(vars(args))}\n"
        )
        f.write(remarks)
        f.write("REMARK    RECEPTOR\n")
        with open(args.receptor, 'r') as r_f:
            for line in r_f:
                if line.startswith('ATOM') or line.startswith('TER'):
                    f.write(line)
        f.write("TER\n")
        f.write("REMARK    LIGAND\n")
        
        with open(args.ligand, 'r') as l_f:
            for line in l_f:
                if line.startswith('ATOM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coord_tensor = torch.tensor([[x, y, z]], dtype=torch.float32)
                    transformed_coord = transform_points(coord_tensor, best_rmsd_T)
                    tx, ty, tz = transformed_coord[0]
                    f.write(f"{line[:30]}{tx:8.3f}{ty:8.3f}{tz:8.3f}{line[54:].rstrip()}\n")
                elif line.startswith('TER') or line.startswith('END'):
                    f.write(line)

    end_time = datetime.datetime.now()
    print(f"--- Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Total time: {end_time - start_time} ---")

if __name__ == '__main__':
    main()