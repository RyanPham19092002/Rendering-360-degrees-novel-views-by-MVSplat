import open3d as o3d
import os
import numpy as np

def merge_point_clouds(ply_folder_path, output_ply_path):
    # Danh sách chứa tất cả các point cloud
    combined_points = []
    combined_colors = []

    # Duyệt qua tất cả các file PLY trong thư mục
    for filename in os.listdir(ply_folder_path):
        if filename.endswith(".ply"):
            file_path = os.path.join(ply_folder_path, filename)
            print(f"Đang đọc file: {file_path}")

            # Đọc point cloud từ file PLY
            pcd = o3d.io.read_point_cloud(file_path)

            # Lấy tọa độ các điểm và màu sắc từ point cloud
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # Thêm vào danh sách tổng hợp
            combined_points.append(points)
            combined_colors.append(colors)

    # Gộp tất cả các point cloud lại với nhau
    combined_points = np.vstack(combined_points)
    combined_colors = np.vstack(combined_colors)

    # Tạo một point cloud mới từ dữ liệu đã gộp
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # Lưu point cloud đã gộp thành một file PLY mới
    o3d.io.write_point_cloud(output_ply_path, combined_pcd)
    print(f"Point cloud đã ghép được lưu tại: {output_ply_path}")

# Thư mục chứa các file PLY cần ghép
ply_folder_path = "/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/file_ply/nf_100/depth"
# Đường dẫn lưu file PLY đã ghép
output_ply_path = "/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/file_ply/concate_point_cloud/nf_100/combined_point_cloud_depth.ply"

# Gọi hàm để ghép các point cloud và lưu kết quả
merge_point_clouds(ply_folder_path, output_ply_path)
