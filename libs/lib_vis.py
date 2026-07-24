from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def get_colored_image_pca_sep(feature, name):
    # Reshape the features to [num_samples, num_features]
    w, h, d = feature.shape
    reshaped_features = feature.reshape((w * h, d))

    # Apply PCA to reduce dimensionality to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_features)

    # Normalize the PCA results to 0-1 range for visualization
    pca_result -= pca_result.min(axis=0)
    pca_result /= pca_result.max(axis=0)

    # Reshape back to the original image shape
    image_data = pca_result.reshape((w, h, 3))

    # Display and save the image
    plt.imshow(image_data)
    plt.axis('off')
    plt.savefig(f'img_{name}.jpg', bbox_inches='tight', pad_inches=0)


def get_colored_point_cloud_from_soft_labels(xyz, soft_labels, name):
    # Convert soft labels to hard labels
    hard_labels = np.argmax(soft_labels, axis=1)
    unique_labels = np.unique(hard_labels)
    # Generate a colormap with 21 distinct colors
    cmap = plt.get_cmap('tab20', len(unique_labels))  # 'tab20b' has 20 distinct colors, adjust as needed for 21
    # Map hard labels to colors using the colormap
    colors = np.array([cmap(i)[:3] for i in hard_labels])  # Extract RGB components
    # Create and color the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

    # Save the point cloud
    o3d.io.write_point_cloud(name + f'.ply', pcd)


