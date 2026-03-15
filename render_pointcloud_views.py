"""
Headless rendering: render point cloud from 3 views and save as images.
No display required. Uses Open3D OffscreenRenderer.
"""
import open3d as o3d
import numpy as np


def render_views(pcd, width=640, height=480, output_prefix="view"):
    """
    Render point cloud from front, top, side and save to images.
    pcd: o3d.geometry.PointCloud
    """
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    render.scene.add_geometry("pcd", pcd, mat)
    render.scene.set_background([1, 1, 1, 1])

    # Camera poses: (center, eye, up) for each view
    center = pcd.get_center()
    extent = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    dist = extent * 2

    views = [
        ("front", center + [0, 0, dist], [0, 1, 0]),
        ("top", center + [0, dist, 0], [0, 0, -1]),
        ("side", center + [dist, 0, 0], [0, 1, 0]),
    ]

    for name, eye, up in views:
        render.setup_camera(60.0, center, eye, up)
        img = render.render_to_image()
        o3d.io.write_image(f"{output_prefix}_{name}.png", img, 9)

    print("Saved", [f"{output_prefix}_{v[0]}.png" for v in views])


if __name__ == "__main__":
    # Example: random point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.randn(5000, 3) * 0.5)
    pcd.paint_uniform_color([0.2, 0.5, 0.8])
    render_views(pcd, output_prefix="pcd")
