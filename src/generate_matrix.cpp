#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

void save_matrix(const MatrixXd& matrix, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << matrix.rows() << " " << matrix.cols() << "\n";
        file << matrix << "\n";
        file.close();
    } else {
        cerr << "Unable to open file for writing: " << filename << endl;
    }
}

int main() {
    MatrixXd A = MatrixXd::Random(1000, 1000);  // Larger matrix for better benchmarking

    // Save the matrix to a file
    save_matrix(A, "matrix.txt");

    cout << "Matrix generated and saved to matrix.txt" << endl;

    return 0;
}
