#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <direct.h>  // Include for _mkdir on Windows

using namespace Eigen;
using namespace std;

void save_matrix(const MatrixXd& matrix, const string& folderName, const string& filename) {
    _mkdir(folderName.c_str());  // Use _mkdir for Windows
    ofstream file(folderName + "/" + filename);
    if (file.is_open()) {
        file << matrix.rows() << " " << matrix.cols() << "\n";
        file << matrix << "\n";
        file.close();
    } else {
        cerr << "Unable to open file for writing: " << folderName + "/" + filename << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <dimension>" << endl;
        return 1;
    }

    Eigen::Index dim = stoi(argv[1]);
    MatrixXd A = MatrixXd::Random(dim, dim);

    string folderName = to_string(dim) + "dim";
    save_matrix(A, folderName, "GeneratedMatrix.txt");

    cout << "Matrix generated and saved to " << folderName << "/GeneratedMatrix.txt" << endl;

    return 0;
}
