#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <string> // Ensuring inclusion for std::stoi and std::to_string
#include <direct.h> // Include for _mkdir on Windows

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

// Gram-Schmidt Process
void gramSchmidt(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R) {
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(A[0].size());
    Q = A;
    R.resize(cols, std::vector<double>(cols, 0.0));

    for (int k = 0; k < cols; ++k) {
        double norm = 0.0;
        for (int i = 0; i < rows; ++i) {
            norm += Q[i][k] * Q[i][k];
        }
        norm = sqrt(norm);
        R[k][k] = norm;

        for (int i = 0; i < rows; ++i) {
            Q[i][k] /= norm;
        }

        for (int j = k + 1; j < cols; ++j) {
            double dotProduct = 0.0;
            for (int i = 0; i < rows; ++i) {
                dotProduct += Q[i][k] * Q[i][j];
            }
            R[k][j] = dotProduct;
            for (int i = 0; i < rows; ++i) {
                Q[i][j] -= dotProduct * Q[i][k];
            }
        }
    }
}

bool isOrthogonal(const std::vector<std::vector<double>>& Q) {
    int size = Q.size();
    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            double dot = 0.0;
            for (int k = 0; k < size; ++k) {
                dot += Q[k][i] * Q[k][j];
            }
            if (std::abs(dot) > 1e-6) { // Use a tolerance for floating point comparison
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

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                QR[i][j] += Q[i][k] * R[k][j];
            }
            if (std::abs(QR[i][j] - A[i][j]) > 1e-6) { // Use a tolerance for floating point comparison
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::vector<std::vector<double>> A, Q, R;
    readMatrixFromFile("GeneratedMatrix.txt", A);

    int dimension = static_cast<int>(A.size()); // Use static_cast to explicitly convert size_t to int
    std::string directoryName = std::to_string(dimension) + "dim";
    _mkdir(directoryName.c_str()); // Create directory using _mkdir from direct.h

    auto start = std::chrono::high_resolution_clock::now();
    gramSchmidt(A, Q, R);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::string originalMatrixPath = directoryName + "/GeneratedMatrix.txt";
    std::string qPath = directoryName + "/outputQ.txt";
    std::string rPath = directoryName + "/outputR.txt";
    std::string reportPath = directoryName + "/report.txt";

    writeMatrixToFile(originalMatrixPath, A); // Write the original matrix to the new directory
    writeMatrixToFile(qPath, Q);
    writeMatrixToFile(rPath, R);

    std::stringstream report;
    report << "Elapsed Time: " << elapsed.count() << " seconds\n";
    report << "Is Q Orthogonal: " << (isOrthogonal(Q) ? "Yes" : "No") << "\n";
    report << "Does A = QR: " << (checkAequalsQR(A, Q, R) ? "Yes" : "No") << "\n";
    writeReport(reportPath, report.str());

    return 0;
}
