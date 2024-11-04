#include <opencv2/opencv.hpp>
#include <iostream>
#include <mpi.h>
#include <vector>

void preprocessImage(cv::Mat& image) {
    // Làm mịn hình ảnh để giảm nhiễu
    cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
}

int main(int argc, char** argv) {
    cv::setNumThreads(1);
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::Mat src;
    int rows, cols;

    if (rank == 0) {
        //Ham imread - doc anh 
        src = cv::imread("C:\\Users\\tri\\source\\repos\\Project1\\Project1\\1-46.png");
        if (src.empty()) {
            std::cerr << "Không thể mở ảnh!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        cv::resize(src, src, cv::Size(), 0.5, 0.5);
        preprocessImage(src);

        rows = src.rows;
        cols = src.cols;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = rows / size;
    int remaining_rows = rows % size;

    int local_rows = (rank < remaining_rows) ? (rows_per_proc + 1) : rows_per_proc;
    cv::Mat local_img(local_rows, cols, CV_8UC3);

    std::vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = (i < remaining_rows) ? (rows_per_proc + 1) * cols * 3 : rows_per_proc * cols * 3;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }
    }

    MPI_Scatterv(src.data, sendcounts.data(), displs.data(), MPI_BYTE,
        local_img.data, local_rows * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Tiền xử lý cho từng tiến trình
    preprocessImage(local_img);

    cv::Mat local_lab;
    cv::cvtColor(local_img, local_lab, cv::COLOR_BGR2Lab);

    cv::Mat data;
    local_lab.convertTo(data, CV_32F);
    data = data.reshape(1, local_rows * cols);

    // Phân đoạn với nhiều K
    std::vector<int> cluster_counts = { 2, 3, 4, 5 };
    for (int K : cluster_counts) {
        cv::Mat labels, centers;
        std::cout << "Đang thực hiện K-means trên tiến trình " << rank << " với K = " << K << "..." << std::endl;

        // Thực hiện K-Means
        cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
            3, cv::KMEANS_PP_CENTERS, centers);

        std::vector<int> all_labels;
        if (rank == 0) {
            all_labels.resize(rows * cols);
        }

        std::vector<int> recvcounts(size), recvdispls(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = (i < remaining_rows) ? (rows_per_proc + 1) * cols : rows_per_proc * cols;
            recvdispls[i] = (i == 0) ? 0 : recvdispls[i - 1] + recvcounts[i - 1];
        }

        MPI_Gatherv(labels.ptr<int>(), local_rows * cols, MPI_INT,
            all_labels.data(), recvcounts.data(), recvdispls.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Ghép ảnh phân đoạn
            cv::Mat segmented(rows, cols, CV_8UC3);
            centers = centers.reshape(3, centers.rows);
            centers.convertTo(centers, CV_8U);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    int clusterIdx = all_labels[i * cols + j];
                    segmented.at<cv::Vec3b>(i, j) = centers.at<cv::Vec3b>(clusterIdx);
                }
            }

            std::string filename = "segmented_output_K_" + std::to_string(K) + ".jpg";
            cv::imwrite(filename, segmented);
            std::cout << "Đã lưu ảnh phân đoạn với K = " << K << " vào " << filename << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
