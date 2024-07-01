#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <string>
#include <numeric>
#include <direct.h>

// Read a matrix from a file
void readMatrixFromFile(const std::string& filename, std::vector<std::vector<double>>& matrix) {
    std::ifstream file(filename);
    int rows, cols;
    file >> rows >> cols;
    matrix.resize(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }
}

// Write a matrix to a file
void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream file(filename);
    for (const auto& row : matrix) {
        for (double val : row) {
            file << std::fixed << std::setprecision(6) << val << " ";
        }
        file << "\n";
    }
}

// Write a report to a file
void writeReport(const std::string& filename, const std::string& report) {
    std::ofstream file(filename);
    file << report;
}

// Check if matrix Q is orthogonal
bool isOrthogonal(const std::vector<std::vector<double>>& Q) {
    int size = Q.size();
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            double dot = 0.0;
            for (int k = 0; k < size; ++k) {
                dot += Q[k][i] * Q[k][j];
            }
            if (std::abs(dot) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

// Check if A = QR
bool checkAequalsQR(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& Q, const std::vector<std::vector<double>>& R) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> QR(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                QR[i][j] += Q[i][k] * R[k][j];
            }
            if (std::abs(QR[i][j] - A[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

// Initialize Q as an identity matrix
void initializeQ(std::vector<std::vector<double>>& Q, int m) {
    Q.resize(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        Q[i][i] = 1.0;
    }
}

// Apply Householder transformation in parallel
void householderStep(std::vector<std::vector<double>>& R, std::vector<std::vector<double>>& Q, int k, int rank, int size) {
    int m = R.size();
    std::vector<double> x(m - k, 0.0);

    if (rank == k % size) {
        for (int i = k; i < m; i++) {
            x[i - k] = R[i][k];
        }
    }

    // Broadcasting x vector to all processes
    MPI_Bcast(x.data(), m - k, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

    double x_norm = std::sqrt(std::accumulate(x.begin(), x.end(), 0.0, [](double acc, double xi) { return acc + xi * xi; }));
    if (x_norm == 0) return;  // Skip if norm is zero

    std::vector<double> u(m - k);
    if (rank == k % size) {
        x[0] = x[0] > 0 ? x[0] + x_norm : x[0] - x_norm;
        for (int i = 0; i < m - k; i++) {
            u[i] = x[i] / std::sqrt(std::accumulate(x.begin(), x.end(), 0.0, [](double acc, double xi) { return acc + xi * xi; }));
        }
    }

    // Broadcasting u vector to all processes
    MPI_Bcast(u.data(), m - k, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

    // Apply transformation to R and Q matrices
    for (int i = 0; i < m; i++) {
        if (i % size != rank) continue;  // Each process only updates its part of R and Q

        double dot_product = 0.0;
        for (int j = k; j < m; j++) {
            dot_product += u[j - k] * R[i][j];
        }
        for (int j = k; j < m; j++) {
            R[i][j] -= 2 * dot_product * u[j - k];
        }
    }

    // Synchronize before updating Q
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < m; i++) {
        if (i % size != rank) continue;

        double dot_product = 0.0;
        for (int j = k; j < m; j++) {
            dot_product += u[j - k] * Q[j][i];
        }
        for (int j = k; j < m; j++) {
            Q[j][i] -= 2 * dot_product * u[j - k];
        }
    }
}

void householderQR(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R, int rank, int size) {
    int m = A.size(), n = A[0].size();
    R = A;
    initializeQ(Q, m);

    for (int k = 0; k < std::min(m, n); ++k) {
        householderStep(R, Q, k, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks finish updating before the next iteration
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::vector<double>> A, Q, R;
    int rows, cols;

    if (rank == 0) {
        readMatrixFromFile("GeneratedMatrix.txt", A);
        rows = static_cast<int>(A.size());
        cols = static_cast<int>(A[0].size());
    }

    // Broadcast the matrix dimensions to all processors
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local rows for each processor
    int localRows = rows / size + (rank < rows % size ? 1 : 0);

    std::vector<std::vector<double>> localA(localRows, std::vector<double>(cols));
    std::vector<std::vector<double>> localQ(localRows, std::vector<double>(cols));
    std::vector<std::vector<double>> localR(localRows, std::vector<double>(cols));

    // Flatten the matrices for MPI operations
    std::vector<double> flatA, flatLocalA(localRows * cols);
    if (rank == 0) {
        flatA.resize(rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                flatA[i * cols + j] = A[i][j];
            }
        }
    }

    // Scatter the matrix A to all processors
    MPI_Scatter(flatA.data(), localRows * cols, MPI_DOUBLE, flatLocalA.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Convert flatLocalA to localA
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            localA[i][j] = flatLocalA[i * cols + j];
        }
    }

    // Perform the Householder QR decomposition
    auto start = std::chrono::high_resolution_clock::now();
    householderQR(localA, localQ, localR, rank, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Gather Q and R matrices from all processors
    if (rank == 0) {
        Q.resize(rows, std::vector<double>(cols));
        R.resize(rows, std::vector<double>(cols));
    }
    std::vector<double> flatQ(rows * cols);
    std::vector<double> flatR(rows * cols);

    // Gather Q and R matrices
    MPI_Gather(localQ.data()->data(), localRows * cols, MPI_DOUBLE, flatQ.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(localR.data()->data(), localRows * cols, MPI_DOUBLE, flatR.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reconstruct the full matrices Q and R
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Q[i][j] = flatQ[i * cols + j];
                R[i][j] = flatR[i * cols + j];
            }
        }

        // Create a directory to store the results
        std::string directoryName = std::to_string(rows) + "x" + std::to_string(cols);
        _mkdir(directoryName.c_str());

        std::string originalMatrixPath = directoryName + "/GeneratedMatrix.txt";
        std::string qPath = directoryName + "/outputQ.txt";
        std::string rPath = directoryName + "/outputR.txt";
        std::string reportPath = directoryName + "/report.txt";

        // Write matrices and report to files
        writeMatrixToFile(originalMatrixPath, A);
        writeMatrixToFile(qPath, Q);
        writeMatrixToFile(rPath, R);

        std::stringstream report;
        report << "Elapsed Time: " << elapsed.count() << " seconds\n";
        report << "Is Q Orthogonal: " << (isOrthogonal(Q) ? "Yes" : "No") << "\n";
        report << "Does A = QR: " << (checkAequalsQR(A, Q, R) ? "Yes" : "No") << "\n";
        writeReport(reportPath, report.str());
    }

    MPI_Finalize();
    return 0;
}
