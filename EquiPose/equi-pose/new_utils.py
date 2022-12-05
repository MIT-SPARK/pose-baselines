
import open3d as o3d


def pos_tensor_to_o3d(pos, estimate_normals=True):
    """
    inputs:
    pos: torch.tensor of shape (3, N)

    output:
    open3d PointCloud
    """
    pos_o3d = o3d.utility.Vector3dVector(pos.transpose(0, 1).to('cpu').numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    if estimate_normals:
        object.estimate_normals()

    return object


def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (B, 3, n)
    pc2 : torch.tensor of shape (B, 3, m)
    """
    pc1 = pc1.detach()[0, ...].to('cpu')
    pc2 = pc2.detach()[0, ...].to('cpu')
    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None

