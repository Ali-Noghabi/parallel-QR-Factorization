#include <chrono>
#include <sstream>
#include <string> // Ensuring inclusion for std::stoi and std::to_string
#include <direct.h> // Include for _mkdir on Windows

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

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

void printMatrix(const std::string& name, const std::vector<std::vector<double>>& matrix, int rank) {
    std::cout << "Matrix " << name << " at rank " << rank << ":\n";
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void parallelGramSchmidt(const std::vector<std::vector<double>>& localA, std::vector<std::vector<double>>& localQ, std::vector<std::vector<double>>& R, int rows, int cols, int rank, int size) {
    localQ = localA;
    R.resize(cols, std::vector<double>(cols, 0.0));

    for (int k = 0; k < cols; ++k) {
        double localNormSq = 0.0;
        for (int i = 0; i < rows; ++i) {
            localNormSq += localQ[i][k] * localQ[i][k];
        }

        double normSq;
        MPI_Allreduce(&localNormSq, &normSq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double norm = sqrt(normSq);
        R[k][k] = norm;

        for (int i = 0; i < rows; ++i) {
            localQ[i][k] /= norm;
        }

        for (int j = k + 1; j < cols; ++j) {
            double localDotProduct = 0.0;
            for (int i = 0; i < rows; ++i) {
                localDotProduct += localQ[i][k] * localQ[i][j];
            }

            double dotProduct;
            MPI_Allreduce(&localDotProduct, &dotProduct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            R[k][j] = dotProduct;

            for (int i = 0; i < rows; ++i) {
                localQ[i][j] -= dotProduct * localQ[i][k];
            }
        }
    }

    // printMatrix("Q (local)", localQ, rank);
    // printMatrix("R", R, rank);
}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::vector<double>> A, Q, R;

    int rows, cols;

    if (rank == 0) {
        readMatrixFromFile("GeneratedMatrix.txt", A);
        // printMatrix("A", A, rank);
        rows = A.size();
        cols = A[0].size();
    }

    // Broadcast the matrix dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localRows = rows / size;
    std::vector<std::vector<double>> localA(localRows, std::vector<double>(cols, 0.0));
    std::vector<std::vector<double>> localQ(localRows, std::vector<double>(cols, 0.0));

    // Scatter the matrix A to all processors
    std::vector<double> flatA(rows * cols, 0.0);
    if (rank == 0) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                flatA[i * cols + j] = A[i][j];
            }
        }
    }
    std::vector<double> flatLocalA(localRows * cols, 0.0);

    MPI_Scatter(flatA.data(), localRows * cols, MPI_DOUBLE, flatLocalA.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            localA[i][j] = flatLocalA[i * cols + j];
        }
    }

    // printMatrix("A (local)", localA, rank);

    auto start = std::chrono::high_resolution_clock::now();
    parallelGramSchmidt(localA, localQ, R, localRows, cols, rank, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Gather the Q matrix from all processors
    std::vector<double> flatQ(rows * cols, 0.0);
    std::vector<double> flatLocalQ(localRows * cols, 0.0);

    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatLocalQ[i * cols + j] = localQ[i][j];
        }
    }

    MPI_Gather(flatLocalQ.data(), localRows * cols, MPI_DOUBLE, flatQ.data(), localRows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Q.resize(rows, std::vector<double>(cols, 0.0));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Q[i][j] = flatQ[i * cols + j];
            }
        }

        // printMatrix("Q", Q, rank);

        std::string directoryName = std::to_string(rows) + "dim";
        _mkdir(directoryName.c_str());

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
    }

    MPI_Finalize();
    return 0;
}
