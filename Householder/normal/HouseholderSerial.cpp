#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <string>
#include <direct.h> // Include for _mkdir on Windows
#include <numeric>  // Include this for std::inner_product

// Utility functions
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

void writeMatrixToFile(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream file(filename);
    for (const auto& row : matrix) {
        for (double val : row) {
            file << std::fixed << std::setprecision(6) << val << " ";
        }
        file << "\n";
    }
}

void writeReport(const std::string& filename, const std::string& report) {
    std::ofstream file(filename);
    file << report;
}

bool isOrthogonal(const std::vector<std::vector<double>>& Q) {
    int size = Q.size();
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            double dot = 0.0;
            for (int k = 0; k < size; ++k) {
                dot += Q[k][i] * Q[k][j];
            }
            if (std::abs(dot) > 1e-5) { // Use a tolerance for floating point comparison
                return false;
            }
        }
    }
    return true;
}

bool checkAequalsQR(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& Q, const std::vector<std::vector<double>>& R) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> QR(rows, std::vector<double>(cols, 0.0));
    bool result = true;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                QR[i][j] += Q[i][k] * R[k][j];
            }
            double diff = std::abs(QR[i][j] - A[i][j]);
            if (diff > 1e-6) {
                std::cout << "Mismatch at (" << i << ", " << j << "): QR[" << i << "][" << j << "] = " << QR[i][j]
                          << ", A[" << i << "][" << j << "] = " << A[i][j] << ", diff = " << diff << std::endl;
                result = false;
            }
        }
    }
    return result;
}

// Apply Householder transformation to zero out below-diagonal entries
void householderStep(std::vector<std::vector<double>>& R, std::vector<std::vector<double>>& Q, int k) {
    int m = R.size();
    std::vector<double> x(m - k);
    for (int i = k; i < m; i++) {
        x[i - k] = R[i][k];
    }

    double x_norm = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
    std::vector<double> e(m - k, 0.0);
    e[0] = x_norm;

    std::vector<double> u(m - k);
    for (int i = 0; i < m - k; i++) {
        u[i] = x[i] - e[i];
    }

    double u_norm = std::sqrt(std::inner_product(u.begin(), u.end(), u.begin(), 0.0));
    if (u_norm == 0) return;  // Skip if u is a zero vector

    for (int i = 0; i < m - k; i++) {
        u[i] /= u_norm;
    }

    // Householder matrix Qk = I - 2 * v * v^T
    // Apply Q_k to R from the left (Q_k * R)
    for (int j = k; j < R[0].size(); j++) {
        double dot_product = 0.0;
        for (int i = 0; i < m - k; i++) {
            dot_product += u[i] * R[k + i][j];
        }
        for (int i = 0; i < m - k; i++) {
            R[k + i][j] -= 2 * dot_product * u[i];
        }
    }

    // Update Q accordingly (Q = Q * Q_k^T)
    for (int j = 0; j < Q.size(); j++) {
        double dot_product = 0.0;
        for (int i = 0; i < m - k; i++) {
            dot_product += u[i] * Q[j][k + i];
        }
        for (int i = 0; i < m - k; i++) {
            Q[j][k + i] -= 2 * dot_product * u[i];
        }
    }
}

// Function to perform QR decomposition using Householder reflections
void householderQR(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R) {
    int m = A.size(), n = A[0].size();
    R = A;
    Q.assign(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        Q[i][i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        householderStep(R, Q, k);
    }
}

int main() {
    std::vector<std::vector<double>> A, Q, R;
    readMatrixFromFile("GeneratedMatrix.txt", A);
    int dimension = static_cast<int>(A.size());
    std::string directoryName = std::to_string(dimension) + "dim";
    _mkdir(directoryName.c_str());

    auto start = std::chrono::high_resolution_clock::now();
    householderQR(A, Q, R);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::string originalMatrixPath = directoryName + "/GeneratedMatrix.txt";
    std::string qPath = directoryName + "/outputQ.txt";
    std::string rPath = directoryName + "/outputR.txt";
    std::string reportPath = directoryName + "/report.txt";

    writeMatrixToFile(originalMatrixPath, A);
    writeMatrixToFile(qPath, Q);
    writeMatrixToFile(rPath, R);

    std::stringstream report;
    report << "Elapsed Time: " << elapsed.count() << " seconds\n";
    report << "Is Q Orthogonal: " << (isOrthogonal(Q) ? "Yes" : "No") << "\n";
    report << "Does A = QR: " << (checkAequalsQR(A, Q, R) ? "Yes" : "No") << "\n";
    writeReport(reportPath, report.str());

    return 0;
}