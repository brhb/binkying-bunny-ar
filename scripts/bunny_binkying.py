import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Functions
def project_points(mesh, intrinsic_matrix, view_matrix):
    vertices = np.asarray(mesh.vertices)
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    vertices_view = vertices_homogeneous @ view_matrix.T
    # vertices_proj = vertices_view @ intrinsic_matrix.T
    # Remove the homogeneous coordinate (last column) from vertices_view
    vertices_view_3d = vertices_view[:, :3]
    # Project the 3D points onto the image plane
    vertices_proj = vertices_view_3d @ intrinsic_matrix.T
    vertices_proj[:, :2] /= vertices_proj[:, 2].reshape(-1, 1)  # Perspective division
    
    # vertices_proj[:, :2] /= vertices_proj[:, 2].reshape(-1, 1)  # Perspective division
    return vertices_proj[:, :2]

def get_view_matrix(camera_position, camera_target, camera_up):
    z_axis = np.float64(camera_position - camera_target)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(camera_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    view_matrix = np.identity(4)
    view_matrix[:3, 0] = x_axis
    view_matrix[:3, 1] = y_axis
    view_matrix[:3, 2] = z_axis
    view_matrix[:3, 3] = -camera_position  # Negate for OpenGL convention
    return view_matrix

# Fonction pour superposer le maillage et les axes sur l'image
def overlay_mesh(image, projected_points, mesh):
    overlay = image.copy()
    triangles = np.asarray(mesh.triangles)
    for tri in triangles:
        pts = projected_points[tri].astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
    return overlay

def overlay_mesh_and_axes(image, projected_points, mesh, intrinsic_matrix, view_matrix):
    overlay = image.copy()
    try:
        import cv2
        # Projeter les axes x, y, z
        axis_length = 0.05
        axes_points = np.array([
            [0, 0, 0],  # Origine
            [axis_length, 0, 0],  # Axe X
            [0, axis_length, 0],  # Axe Y
            [0, 0, axis_length]   # Axe Z
        ])
        axes_homogeneous = np.hstack((axes_points, np.ones((axes_points.shape[0], 1))))
        axes_camera = axes_homogeneous @ view_matrix.T
        axes_camera = axes_camera[:, :3] / axes_camera[:, 3].reshape(-1, 1)
        axes_projected = axes_camera @ intrinsic_matrix.T
        axes_projected = axes_projected[:, :2] / axes_projected[:, 2].reshape(-1, 1)

        # Dessiner les axes
        origin = tuple(axes_projected[0].astype(int))
        x_axis = tuple(axes_projected[1].astype(int))
        y_axis = tuple(axes_projected[2].astype(int))
        z_axis = tuple(axes_projected[3].astype(int))

        cv2.arrowedLine(overlay, origin, x_axis, (0, 0, 255), 2, tipLength=0.1)  # Rouge pour X
        cv2.arrowedLine(overlay, origin, y_axis, (0, 255, 0), 2, tipLength=0.1)  # Vert pour Y
        cv2.arrowedLine(overlay, origin, z_axis, (255, 0, 0), 2, tipLength=0.1)  # Bleu pour Z

        # Récupérer les arêtes du maillage
        triangles = np.asarray(mesh.triangles)
        for triangle in triangles:
            pts = projected_points[triangle].astype(int)
            pts = pts.reshape((-1, 1, 2))
            overlay = cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    except ImportError:
        print("OpenCV is requis pour la visualisation de l'overlay.")
    return overlay

# Step 1: Load the Bunny Mesh
bunny_mesh = o3d.io.read_triangle_mesh("../data/bunny.obj")
bunny_mesh.compute_vertex_normals()

# Step 2: Load the Real Table Image
image_path = "../data/table.jpg"  # Replace with your image path
real_image = cv2.imread(image_path)
real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

# Step 3: Define Camera Parameters
# Intrinsic matrix (assume some calibrated camera parameters)
fx, fy = 1000, 1000  # Focal lengths
cx, cy = real_image.shape[1] // 2, real_image.shape[0] // 2  # Principal point
intrinsic_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
# Extrinsic matrix: Place camera and look at the table
camera_position = np.array([-0.15, 0.16, 0.4])  # 3 meters above the table
camera_target = np.array([0, 0, 0])  # Looking at the origin
camera_up = np.array([0, 1, 0])

view_matrix = get_view_matrix(camera_position, camera_target, camera_up)

# Step 4: Transform the Bunny onto the Table
bunny_mesh.scale(0.5, center=bunny_mesh.get_center())  # Adjust size
x_positions = np.linspace(0, -0.1, 10)  # Mouvement de 0 à 1 sur l'axe X
frames = []

for x in x_positions:
    bunny_mesh.translate(np.array([x, 0, 0]))  # Déplacer le lapin
    projected_points = project_points(bunny_mesh, intrinsic_matrix, view_matrix)
    frame = overlay_mesh(real_image, projected_points, bunny_mesh)
    frames.append(frame)#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bunny_mesh.translate(np.array([-x, 0, 0]))  # Réinitialiser la position du lapin

# Animation avec Matplotlib
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(frames[0])
ax.axis("off")

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
