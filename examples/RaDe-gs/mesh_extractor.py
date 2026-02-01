import torch
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple, Optional, Sequence

import os

from splatkit.splat.training_state import SplatTrainingState
from splatkit.modules import SplatBaseModule, SplatRenderPayload
from splatkit.logger import SplatLogger
from splatkit.renderer import SplatRenderer
from splatkit.data_provider import SplatDataProvider, SplatDataItem
from splatkit.loss_fn import SplatLossFn
from splatkit.densification import SplatDensification

import trimesh
from trimesh import Trimesh

class SplatMeshExtractor(SplatBaseModule[SplatRenderPayload]):
    """
    Module for extracting meshes from splat models using marching tetrahedra.
    
    This implementation is adapted from Gaussian Opacity Fields (GOF):
    https://github.com/autonomousvision/gaussian-opacity-fields
    
    The extraction process:
    1. Generate tetrahedra from Gaussian centers and bounding boxes
    2. Evaluate signed distance function (SDF) at tetrahedra vertices by:
       - Integrating transmittance from each training camera to query points
       - Taking conservative estimate across all views (minimum alpha)
       - Converting to SDF (alpha > 0.5 = inside, < 0.5 = outside)
    3. Apply marching tetrahedra to extract initial surface mesh
    4. Refine surface location using binary search on SDF
    5. Filter mesh faces based on scale consistency
    
    Args:
        output_dir: Directory to save extracted meshes
        export_steps: Training steps at which to extract meshes
        kernel_size: Kernel size for alpha integration (3D filter radius)
        n_binary_steps: Number of binary search refinement steps
        extract_full_mesh: If True, extracts full mesh. If False, only saves splat model
    
    Example:
        >>> mesh_extractor = SplatMeshExtractor(
        ...     output_dir="./output",
        ...     export_steps=[7000, 15000, 30000],
        ...     kernel_size=0.1,
        ...     n_binary_steps=8,
        ... )
        >>> # Add to trainer modules
        >>> trainer.add_module(mesh_extractor)
    
    Notes:
        - Requires tetra-nerf for optimal performance: 
          pip install git+https://github.com/jkulhanek/tetra-nerf
        - Falls back to scipy Delaunay if tetra-nerf unavailable
        - Memory intensive for large point clouds (uses batching)
    """
    def __init__(
        self,
        output_dir: str,
        export_steps: list[int] = [],
        kernel_size: float = 0.1,
        n_binary_steps: int = 8,
        extract_full_mesh: bool = True,
    ):
        super().__init__()
        self._output_dir = os.path.join(output_dir, "meshes")
        self._export_steps = export_steps
        self._kernel_size = kernel_size
        self._n_binary_steps = n_binary_steps
        self._extract_full_mesh = extract_full_mesh

    def on_setup(
        self,
        logger: SplatLogger,
        renderer: SplatRenderer[SplatRenderPayload],
        data_provider: SplatDataProvider[SplatRenderPayload, SplatDataItem],
        loss_fn: SplatLossFn[SplatRenderPayload],
        densification: SplatDensification[SplatRenderPayload],
        modules: Sequence[SplatBaseModule[SplatRenderPayload]],
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        super().on_setup(
            logger, 
            renderer, 
            data_provider, 
            loss_fn, 
            densification, 
            modules, 
            max_steps, 
            world_rank, 
            world_size, 
            scene_scale
        )

        self._renderer = renderer
        self._data_provider = data_provider
        
        os.makedirs(self._output_dir, exist_ok=True)
        logger.info(f"Mesh extractor will save to: {self._output_dir}", module=self.module_name)
        logger.info(f"Successfully setup mesh extractor. Will extract meshes at steps: {sorted(self._export_steps)}", module=self.module_name)
    
    def post_step(
        self,
        logger: SplatLogger,
        step: int,
        max_steps: int,
        rendered_frames: torch.Tensor,
        target_frames: torch.Tensor,
        training_state: SplatTrainingState,
        rend_out: SplatRenderPayload,
        masks: torch.Tensor | None = None,
        world_rank: int = 0,
        world_size: int = 1,    
    ):
        if self._export_steps and step in self._export_steps:
            logger.info(f"Extracting mesh at step {step}...", module=self.module_name)
            
            try:
                # Extract full mesh using marching tetrahedra
                mesh_path = os.path.join(self._output_dir, f"{step}.ply")
                self._extract_mesh_marching_tetrahedra(
                    training_state=training_state,
                    output_path=mesh_path,
                    logger=logger,
                )
                logger.info(f"Successfully extracted mesh to {mesh_path}", module=self.module_name)
            except Exception as e:
                raise RuntimeError(f"Failed to extract mesh: {e}")
    
    @torch.no_grad()
    def _extract_mesh_marching_tetrahedra(
        self,
        training_state: SplatTrainingState,
        output_path: str,
        logger: SplatLogger,
    ):
        """
        Extract mesh using marching tetrahedra with binary search refinement.
        
        This is the core mesh extraction pipeline following GOF:
        
        Step 1: Generate tetrahedra points
            - For each Gaussian, create 8 box corners + 1 center = 9 points
            - Box is scaled by 3*sigma (covers 99.7% of Gaussian support)
            - Boxes are rotated/scaled according to Gaussian covariance
        
        Step 2: Triangulate points
            - Use Delaunay triangulation to create tetrahedra
            - Each tetrahedron is defined by 4 vertices
        
        Step 3: Evaluate SDF at vertices
            - For each vertex, integrate transmittance from all training cameras
            - Alpha > 0.5 = inside surface (SDF < 0)
            - Alpha < 0.5 = outside surface (SDF > 0)
            - Take minimum alpha across views (conservative)
        
        Step 4: Apply marching tetrahedra
            - Extract surface where SDF crosses zero
            - Creates edges that span the surface
            - Returns edge endpoints and their SDF values
        
        Step 5: Binary search refinement
            - Repeatedly bisect edges to find precise surface location
            - At each iteration, evaluate SDF at midpoint
            - Move left/right pointer based on SDF sign
            - Converges to SDF ≈ 0
        
        Step 6: Filter and export
            - Remove faces where edge length > combined Gaussian scales
            - This filters noise from distant/unrelated Gaussians
            - Export final mesh as PLY file
        
        Args:
            training_state: Current training state with Gaussian parameters
            output_path: Where to save the extracted mesh
            logger: Logger for progress messages
        """

        filter_3D = self.compute_3D_filter(self._data_provider, training_state, None, logger)
        
        # Step 1: Generate tetrahedra points
        logger.info("Generating tetrahedra points...", module=self.module_name)
        points, points_scale = self._get_tetra_points(training_state, filter_3D)
        logger.info(f"  Generated {points.shape[0]} tetrahedra vertices", module=self.module_name)
        
        # Step 2: Triangulate
        logger.info("Triangulating points...", module=self.module_name)
        cells = self._triangulate(points)
        logger.info(f"  Created {cells.shape[0]} tetrahedra", module=self.module_name)
        
        # Step 3: Evaluate SDF via alpha integration
        logger.info("Evaluating SDF via alpha integration...", module=self.module_name)
        sdf = self._evaluate_cull_alpha(points, self._data_provider, training_state, None, logger)
        
        # Step 4: Apply marching tetrahedra
        logger.info("Applying marching tetrahedra...", module=self.module_name)
        torch.cuda.empty_cache()
        verts_list, scale_list, faces_list = self._marching_tetrahedra(
            points.cpu()[None], 
            cells.cpu().long(), 
            sdf[None].cpu(), 
            points_scale[None].cpu()
        )
        
        del points
        del points_scale
        del cells
        
        end_points, end_sdf = verts_list[0]
        end_scales = scale_list[0]
        end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()
        
        faces = faces_list[0].cpu().numpy()
        logger.info(f"  Initial mesh: {end_points.shape[0]} edges, {faces.shape[0]} faces", module=self.module_name)
        
        # Step 5: Binary search refinement
        logger.info(f"Refining surface with {self._n_binary_steps} binary search steps...", module=self.module_name)
        left_points = end_points[:, 0, :]
        right_points = end_points[:, 1, :]
        left_sdf = end_sdf[:, 0, :]
        right_sdf = end_sdf[:, 1, :]
        left_scale = end_scales[:, 0, 0]
        right_scale = end_scales[:, 1, 0]
        distance = torch.norm(left_points - right_points, dim=-1)
        scale = left_scale + right_scale
        
        for step in range(self._n_binary_steps):
            logger.info(f"  Binary search step {step + 1}/{self._n_binary_steps}", module=self.module_name)
            mid_points = (left_points + right_points) / 2
            mid_sdf = self._evaluate_cull_alpha(mid_points, self._data_provider, training_state, None, logger)
            mid_sdf = mid_sdf.unsqueeze(-1)
            
            # Update pointers based on SDF sign
            ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
            
            left_sdf[ind_low] = mid_sdf[ind_low]
            right_sdf[~ind_low] = mid_sdf[~ind_low]
            left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
            right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
        points = (left_points + right_points) / 2
        
        # Step 6: Create and filter mesh
        logger.info("Creating and filtering mesh...", module=self.module_name)
        mesh = Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
        
        # Filter based on scale: remove edges longer than combined Gaussian radii
        vertice_mask = (distance <= scale).cpu().numpy()
        face_mask = vertice_mask[faces].all(axis=1)
        mesh.update_vertices(vertice_mask)
        mesh.update_faces(face_mask)
        
        # Export mesh
        mesh.export(output_path)
        logger.info(f"Mesh extracted: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces", module=self.module_name)
        logger.info(f"Saved to: {output_path}", module=self.module_name)
    
    @torch.no_grad()
    def _evaluate_cull_alpha(
        self,
        points: torch.Tensor,
        data_provider: SplatDataProvider,
        training_state: SplatTrainingState,
        masks: Optional[torch.Tensor],
        logger: SplatLogger,
    ) -> torch.Tensor:
        """
        Evaluate alpha values at points from multiple views with culling.
        
        This method follows GOF's approach:
        1. For each camera view, integrate transmittance to query points
        2. Only update SDF for points visible in the view (within frustum and valid depth)
        3. Take minimum alpha across all views (most conservative estimate)
        4. Convert to SDF: positive outside surface (alpha < 0.5), negative inside (alpha > 0.5)
        
        Returns SDF values where negative means inside the surface.
        """
        final_sdf = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
        weight = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")
        
        num_views = data_provider.get_train_data_size()
        
        for step in tqdm(range(num_views), desc="Integrating alpha from views", disable=False):
            torch.cuda.empty_cache()

            item = data_provider.next_train_data(step)
            
            # Integrate alpha at query points
            alpha_integrated, point_coordinate = self._integrate_alpha(
                points, item, training_state
            )
            
            # Check visibility: points must be within view frustum
            height, width = item.image.shape[:2]
            
            # Apply margin like GOF (15% border extension)
            margin_x = 0.15 * width
            margin_y = 0.15 * height
            
            valid_x = (point_coordinate[:, 0] >= -margin_x) & (point_coordinate[:, 0] <= width + margin_x)
            valid_y = (point_coordinate[:, 1] >= -margin_y) & (point_coordinate[:, 1] <= height + margin_y)
            valid_point = valid_x & valid_y
            
            # Optional: apply ground truth masks if available
            if masks is not None and step < len(masks):
                # Sample mask at point coordinates
                # Normalize to [-1, 1] for grid_sample
                point_coord_norm = point_coordinate.clone()
                point_coord_norm[:, 0] = (point_coord_norm[:, 0] * 2 + 1) / (width - 1) - 1
                point_coord_norm[:, 1] = (point_coord_norm[:, 1] * 2 + 1) / (height - 1) - 1
                
                mask = masks[step]
                if mask is not None:
                    mask_value = torch.nn.functional.grid_sample(
                        mask[None, None].float(),
                        point_coord_norm[None, None],
                        padding_mode='zeros',
                        align_corners=False
                    )
                    mask_valid = mask_value[0, 0, 0] > 0.5
                    valid_point = valid_point & mask_valid
            
            # Update SDF: take minimum alpha (conservative)
            final_sdf = torch.where(valid_point, torch.min(alpha_integrated, final_sdf), final_sdf)
            weight = torch.where(valid_point, weight + 1, weight)
        
        # Convert to SDF convention:
        # - Inside surface (high alpha > 0.5): negative SDF
        # - Outside surface (low alpha < 0.5): positive SDF  
        # - Unobserved points: large negative (will be inside)
        final_sdf = torch.where(weight > 0, 0.5 - final_sdf, -100)
        
        logger.info(f"  SDF range: [{final_sdf.min():.4f}, {final_sdf.max():.4f}]", module=self.module_name)
        logger.info(f"  Valid points: {(weight > 0).sum()}/{points.shape[0]}", module=self.module_name)
        
        return final_sdf
    
    @torch.no_grad()
    def _integrate_alpha(
        self,
        points: torch.Tensor,
        data_item: SplatDataItem,
        training_state: SplatTrainingState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate alpha values at 3D query points from a specific viewpoint.
        
        This method computes the integrated transmittance along rays from the camera
        to each query point by accumulating opacity from all Gaussians that contribute.
        
        Returns:
            alpha_integrated: Integrated alpha at each point [N] (1 - transmittance)
            point_coordinate: 2D screen coordinates [N, 2]
        """
        # Get camera parameters from data item
        cam_to_world = torch.from_numpy(data_item.cam_to_world).float().cuda()
        K = torch.from_numpy(data_item.K).float().cuda()
        height, width = data_item.image.shape[:2]
        
        # Transform points to camera space
        world_to_cam = torch.linalg.inv(cam_to_world)
        R = world_to_cam[:3, :3]
        t = world_to_cam[:3, 3]
        
        points_cam = points @ R.T + t[None, :]
        z_depth = points_cam[:, 2]  # Camera-space depth
        
        # Project to screen space
        points_screen = points_cam @ K.T
        points_screen_xy = points_screen[:, :2] / (points_screen[:, 2:3] + 1e-8)
        
        # Get Gaussian parameters
        means = training_state.params["means"]
        quats = training_state.params["quats"]
        scales = torch.exp(training_state.params["scales"])
        opacities = torch.sigmoid(training_state.params["opacities"])
        
        # Transform Gaussian centers to camera space
        means_cam = means @ R.T + t[None, :]
        gaussian_depths = means_cam[:, 2]  # [N_gaussians]
        
        # Compute 3D covariance matrices from quaternions and scales
        rots = self._build_rotation(quats)  # [N_gaussians, 3, 3]
        S = torch.diag_embed(scales)  # [N_gaussians, 3, 3]
        RS = rots @ S  # [N_gaussians, 3, 3]
        cov3d = RS @ RS.transpose(1, 2)  # [N_gaussians, 3, 3]
        
        # For each query point, compute contributions from all Gaussians
        # Only consider Gaussians in front of the query point (depth < point depth)
        N_points = points.shape[0]
        N_gaussians = means.shape[0]
        
        # Compute Mahalanobis distances from query points to Gaussians
        # For efficiency, process in batches
        batch_size = 10000  # Process points in batches to save memory
        alpha_integrated = torch.zeros(N_points, device=points.device)
        
        for i in range(0, N_points, batch_size):
            end_i = min(i + batch_size, N_points)
            batch_points = points[i:end_i]  # [B, 3]
            batch_depths = z_depth[i:end_i]  # [B]
            
            # Compute displacement vectors [B, N_gaussians, 3]
            diff = batch_points[:, None, :] - means[None, :, :]  # [B, N_gaussians, 3]
            
            # Compute Mahalanobis distance using inverse covariance
            # For numerical stability, use Cholesky decomposition
            try:
                # Add small regularization for numerical stability
                cov3d_reg = cov3d + torch.eye(3, device=cov3d.device) * 1e-6
                cov3d_inv = torch.linalg.inv(cov3d_reg)  # [N_gaussians, 3, 3]
            except:
                # Fallback to simple distance-based weighting
                dists_sq = (diff ** 2).sum(dim=-1)  # [B, N_gaussians]
                scale_avg = scales.mean(dim=-1, keepdim=True)  # [N_gaussians, 1]
                mahal_dist_sq = dists_sq / (scale_avg.squeeze() ** 2 + 1e-8)
            else:
                # Compute x^T Σ^-1 x
                temp = torch.einsum('bni,nij->bnj', diff, cov3d_inv)  # [B, N_gaussians, 3]
                mahal_dist_sq = torch.einsum('bni,bni->bn', temp, diff)  # [B, N_gaussians]
            
            # Apply depth culling: only consider Gaussians closer to camera than query point
            depth_mask = gaussian_depths[None, :] < batch_depths[:, None]  # [B, N_gaussians]
            
            # Compute Gaussian weights
            gaussian_weights = torch.exp(-0.5 * mahal_dist_sq) * depth_mask.float()
            
            # Compute opacity contribution
            alpha_contrib = opacities.squeeze() * gaussian_weights  # [B, N_gaussians]
            
            # Integrate: α_integrated = 1 - ∏(1 - α_i)
            transmittance = torch.prod(1 - alpha_contrib + 1e-10, dim=1)  # [B]
            batch_alpha = 1 - transmittance
            
            alpha_integrated[i:end_i] = batch_alpha
        
        return alpha_integrated, points_screen_xy
    
    def _build_rotation(self, r: torch.Tensor) -> torch.Tensor:
        """Build rotation matrices from quaternions (w, x, y, z format)."""
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R
    
    @torch.no_grad()
    def _get_tetra_points(
        self, 
        training_state: SplatTrainingState,
        filter_3d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tetrahedra points from Gaussian centers and scales.
        
        This method follows GOF's approach:
        1. For each Gaussian, create 8 corner points of a scaled/rotated box
        2. Add the Gaussian center as the 9th point
        3. These 9 points per Gaussian form tetrahedra that will be triangulated
        
        The scale factor of 3.0 is from GOF - it ensures the boxes cover the
        Gaussian's effective support (3 sigma ~ 99.7% of mass).
        
        Returns:
            vertices: All tetrahedra vertices [N_vertices, 3]
            vertices_scale: Scale at each vertex [N_vertices, 1] for filtering
        
        Note: GOF includes optional opacity filtering for bicycle scene.
        This is disabled by default but can be enabled for cleaner meshes.
        """
        # Create box template (8 corners)
        M = trimesh.creation.box()
        M.vertices *= 2  # Make it unit cube [-1, 1]
        
        # Get Gaussian parameters
        rots = self._build_rotation(training_state.params["quats"])
        xyz = training_state.params["means"]
        scale = torch.exp(training_state.params["scales"])
        scale = torch.sqrt(torch.square(scale) + torch.square(filter_3d)) * 3.0
        
        # Optional: Filter points with small opacity (for cleaner meshes)
        opacity = torch.sigmoid(training_state.params["opacities"])
        mask = (opacity > 0.1).squeeze(-1)
        xyz = xyz[mask]
        scale = scale[mask]
        rots = rots[mask]
        
        N_gaussians = xyz.shape[0]
        
        # Transform box vertices: scale -> rotate -> translate
        vertices = torch.from_numpy(M.vertices).T.float().cuda()  # [3, 8]
        vertices = vertices.unsqueeze(0).repeat(N_gaussians, 1, 1)  # [N, 3, 8]
        
        # Scale vertices first
        vertices = vertices * scale.unsqueeze(-1)  # [N, 3, 8]
        
        # Rotate and translate
        vertices = torch.bmm(rots, vertices) + xyz.unsqueeze(-1)  # [N, 3, 8]
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()  # [N*8, 3]
        
        # Concat center points (9th point per Gaussian)
        vertices = torch.cat([vertices, xyz], dim=0)  # [N*9, 3]
        
        # Assign scales for filtering (use max scale as representative)
        # "scale is not a good solution but use it for now"
        scale_max = scale.max(dim=-1, keepdim=True)[0]  # [N, 1]
        scale_corner = scale_max.repeat(1, 8).reshape(-1, 1)  # [N*8, 1]
        vertices_scale = torch.cat([scale_corner, scale_max], dim=0)  # [N*9, 1]
        
        return vertices, vertices_scale
    
    def _triangulate(self, points: torch.Tensor) -> torch.Tensor:
        """
        Triangulate points using tetra-nerf's cpp extension.
        
        NOTE: This requires tetra-nerf to be installed:
        pip install git+https://github.com/jkulhanek/tetra-nerf
        
        If you don't have tetra-nerf, this will fall back to scipy Delaunay.
        """
        try:
            from tetranerf.utils.extension import cpp
            cells = cpp.triangulate(points)
            return cells
        except ImportError:
            import warnings
            warnings.warn(
                "tetra-nerf not found, falling back to scipy Delaunay. "
                "For better performance, install: pip install git+https://github.com/jkulhanek/tetra-nerf"
            )
            from scipy.spatial import Delaunay
            points_np = points.cpu().numpy()
            tri = Delaunay(points_np)
            cells = torch.from_numpy(tri.simplices).long().cuda()
            return cells
    
    def _marching_tetrahedra(
        self,
        points: torch.Tensor,
        cells: torch.Tensor,
        sdf: torch.Tensor,
        points_scale: torch.Tensor,
    ) -> Tuple[list, list, list]:
        """
        Apply marching tetrahedra using tetra-nerf's implementation.
        
        NOTE: This requires tetra-nerf to be installed.
        If you don't have it, this will use a simplified fallback.
        """
        try:
            from utils.tetmesh import marching_tetrahedra
            verts_list, scale_list, faces_list, _ = marching_tetrahedra(
                points, cells, sdf, points_scale
            )
            return verts_list, scale_list, faces_list
        except ImportError:
            import warnings
            warnings.warn(
                "tetra-nerf tetmesh utils not found, using simplified marching tetrahedra. "
                "Results may be suboptimal. Install: pip install git+https://github.com/jkulhanek/tetra-nerf"
            )
            return self._marching_tetrahedra_fallback(points, cells, sdf, points_scale)
    
    def _marching_tetrahedra_fallback(
        self,
        points: torch.Tensor,
        cells: torch.Tensor,
        sdf: torch.Tensor,
        points_scale: torch.Tensor,
    ) -> Tuple[list, list, list]:
        """
        Simplified fallback marching tetrahedra implementation.
        """
        verts_list = []
        scale_list = []
        faces_list = []
        
        batch_size = points.shape[0]
        for b in range(batch_size):
            verts, scales, faces = self._marching_tetrahedra_single(
                points[b], cells, sdf[b], points_scale[b]
            )
            verts_list.append(verts)
            scale_list.append(scales)
            faces_list.append(faces)
        
        return verts_list, scale_list, faces_list
    
    def _marching_tetrahedra_single(
        self,
        points: torch.Tensor,
        cells: torch.Tensor,
        sdf: torch.Tensor,
        points_scale: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Simplified marching tetrahedra for a single batch.
        """
        # Get SDF values at tetrahedra vertices
        sdf_tet = sdf[cells]  # [M, 4]
        
        # Find tetrahedra that contain the isosurface (sign changes)
        sdf_pos = sdf_tet > 0
        sdf_neg = sdf_tet < 0
        has_surface = sdf_pos.any(dim=1) & sdf_neg.any(dim=1)
        
        # Extract edges that cross isosurface
        surface_cells = cells[has_surface]  # [M', 4]
        surface_sdf = sdf_tet[has_surface]  # [M', 4]
        
        # Tetrahedron edges (6 edges per tet)
        edges = torch.tensor([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]
        ], device=points.device)
        
        # Get edge vertices
        edge_verts = []
        edge_sdfs = []
        edge_scales = []
        face_indices = []
        
        for i, cell in enumerate(surface_cells):
            cell_sdf = surface_sdf[i]
            
            # Check each edge
            crossing_edges = []
            for e_idx, (v0, v1) in enumerate(edges):
                sdf0 = cell_sdf[v0]
                sdf1 = cell_sdf[v1]
                
                # Edge crosses isosurface if signs differ
                if (sdf0 > 0) != (sdf1 > 0):
                    idx0 = cell[v0]
                    idx1 = cell[v1]
                    
                    # Store edge endpoints
                    p0 = points[idx0]
                    p1 = points[idx1]
                    edge_verts.append(torch.stack([p0, p1]))
                    edge_sdfs.append(torch.stack([sdf0, sdf1]))
                    
                    s0 = points_scale[idx0]
                    s1 = points_scale[idx1]
                    edge_scales.append(torch.stack([s0, s1]))
                    
                    crossing_edges.append(len(edge_verts) - 1)
            
            # Create faces from crossing edges (simplified triangulation)
            if len(crossing_edges) >= 3:
                face_indices.append([crossing_edges[0], crossing_edges[1], crossing_edges[2]])
        
        # Stack results
        if len(edge_verts) > 0:
            edge_verts_t = torch.stack(edge_verts)  # [N_edges, 2, 3]
            edge_sdfs_t = torch.stack(edge_sdfs)  # [N_edges, 2]
            edge_scales_t = torch.stack(edge_scales)  # [N_edges, 2, 1]
            faces_t = torch.tensor(face_indices, dtype=torch.long)
        else:
            # Empty mesh
            edge_verts_t = torch.zeros((0, 2, 3))
            edge_sdfs_t = torch.zeros((0, 2))
            edge_scales_t = torch.zeros((0, 2, 1))
            faces_t = torch.zeros((0, 3), dtype=torch.long)
        
        return (edge_verts_t, edge_sdfs_t), edge_scales_t, faces_t

    @torch.no_grad()
    def compute_3D_filter(
        self, 
        data_provider: SplatDataProvider,
        training_state: SplatTrainingState,
        masks: Optional[torch.Tensor],
        logger: SplatLogger,
    ) -> torch.Tensor:
        # print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = training_state.params["means"]
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for step in range(data_provider.get_train_data_size()):
            item = data_provider.next_train_data(step)
            # focal_x = float(camera.intrinsic[0,0])
            # focal_y = float(camera.intrinsic[1,1])
            W, H = item.image.shape[1:3]
            K = item.K
            if K.dim() == 3:
                fx = K[:, 0, 0]
                fy = K[:, 1, 1]
                cx = K[:, 0, 2]
                cy = K[:, 1, 2]
            else:
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]

            focal_x = fx
            focal_y = fy

            # transform points to camera space
            R = torch.tensor(item.cam_to_world[:3, :3], device=xyz.device, dtype=torch.float32)
            T = torch.tensor(item.cam_to_world[:3, 3], device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * focal_x + W / 2.0
            y = y / z * focal_y + H / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * W, x <= W * 1.15), torch.logical_and(y >= -0.15 * H, y <= H * 1.15))
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < focal_x:
                focal_length = focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        return filter_3D[..., None]