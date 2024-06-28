#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace Eigen;
using namespace std;

void householder_step(MatrixXd& R, MatrixXd& Q, int k) {
    int m = R.rows();
    int n = R.cols();
    VectorXd x = R.block(k, k, m - k, 1);
    VectorXd e = VectorXd::Zero(m - k);
    e(0) = x.norm();
    VectorXd u = x - e;
    if (u.norm() == 0) return;
    VectorXd v = u / u.norm();
    MatrixXd Q_k = MatrixXd::Identity(m, m);
    Q_k.block(k, k, m - k, m - k) -= 2.0 * v * v.transpose();
    R = Q_k * R;
    Q = Q * Q_k.transpose();
}

void parallel_householder_reflection(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    int m = A.rows();
    int n = A.cols();
    R = A;
    Q = MatrixXd::Identity(m, m);

    #pragma omp parallel for shared(R, Q)
    for (int k = 0; k < n; ++k) {
        MatrixXd R_local = R;
        MatrixXd Q_local = Q;
        householder_step(R_local, Q_local, k);
        
        #pragma omp critical
        {
            R = R_local;
            Q = Q_local;
        }
    }
}

MatrixXd load_matrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Unable to open file for reading: " << filename << endl;
        exit(1);
    }

    int rows, cols;
    file >> rows >> cols;
    MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix(i, j);

    file.close();
    return matrix;
}

int main() {
    // Load the matrix from the file
    MatrixXd A = load_matrix("matrix.txt");

    auto start = chrono::high_resolution_clock::now();
    MatrixXd Q, R;
    parallel_householder_reflection(A, Q, R);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Parallel Householder QR:" << endl;
    cout << "Is Q orthogonal? " << Q.transpose() * Q << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;
    cout << "Is A = QR? " << (A - Q * R).norm() << endl;

    return 0;
}
