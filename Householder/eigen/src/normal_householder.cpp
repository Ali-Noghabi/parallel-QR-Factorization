#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <direct.h>
#include <sstream>
#include <Windows.h>
#include <TCHAR.h>
#include <pdh.h>
#include <psapi.h>

using namespace Eigen;
using namespace std;

// Global variables for CPU usage measurement
static PDH_HQUERY cpuQuery;
static PDH_HCOUNTER cpuTotal;
static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
static int numProcessors;
static HANDLE self;

// Initialize PDH query and counter for total CPU usage
void init_cpu_total_usage() {
    PdhOpenQuery(NULL, NULL, &cpuQuery);
    PdhAddEnglishCounterW(cpuQuery, L"\\Processor(_Total)\\% Processor Time", NULL, &cpuTotal);
    PdhCollectQueryData(cpuQuery);
}

// Get current total CPU usage percentage
double get_current_cpu_total_usage() {
    PDH_FMT_COUNTERVALUE counterVal;
    PdhCollectQueryData(cpuQuery);
    PdhGetFormattedCounterValue(cpuTotal, PDH_FMT_DOUBLE, NULL, &counterVal);
    return counterVal.doubleValue;
}

// Initialize CPU usage measurement for current process
void init_process_cpu_usage() {
    SYSTEM_INFO sysInfo;
    FILETIME ftime, fsys, fuser;

    GetSystemInfo(&sysInfo);
    numProcessors = sysInfo.dwNumberOfProcessors;

    GetSystemTimeAsFileTime(&ftime);
    memcpy(&lastCPU, &ftime, sizeof(FILETIME));

    self = GetCurrentProcess();
    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
    memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));
}

// Get current CPU usage percentage for current process
double get_current_process_cpu_usage() {
    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    double percent;

    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));

    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));
    percent = (sys.QuadPart - lastSysCPU.QuadPart) +
              (user.QuadPart - lastUserCPU.QuadPart);
    percent /= (now.QuadPart - lastCPU.QuadPart);
    percent /= numProcessors;
    lastCPU = now;
    lastUserCPU = user;
    lastSysCPU = sys;

    return percent * 100;
}

// Logging function with CPU usage
void log_step_info(stringstream& logStream, int step, const chrono::duration<double>& elapsed, const string& status) {
    double total_cpu_usage = get_current_cpu_total_usage();
    double process_cpu_usage = get_current_process_cpu_usage();

    auto now = chrono::system_clock::now();
    time_t now_c = chrono::system_clock::to_time_t(now);
    logStream << "Step " << step << ": Total CPU Usage = " << total_cpu_usage << "%, Process CPU Usage = " << process_cpu_usage << "%, Elapsed Time = " << elapsed.count() << " seconds, Status: " << status << "\n";
    cout << "Step " << step << ": Total CPU Usage = " << total_cpu_usage << "%, Process CPU Usage = " << process_cpu_usage << "%, Elapsed Time = " << elapsed.count() << " seconds, Status: " << status << "\n";
}

void householder_step(MatrixXd& R, MatrixXd& Q, Eigen::Index k, stringstream& logStream, const chrono::high_resolution_clock::time_point& start) {
    Eigen::Index m = R.rows();
    Eigen::Index n = R.cols();
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

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    log_step_info(logStream, k, elapsed, "Completed");
}

chrono::duration<double> householder_reflection(const MatrixXd& A, MatrixXd& Q, MatrixXd& R, stringstream& logStream) {
    auto start = chrono::high_resolution_clock::now();
    Eigen::Index m = A.rows();
    Eigen::Index n = A.cols();
    R = A;
    Q = MatrixXd::Identity(m, m);
    
    for (Eigen::Index k = 0; k < n; ++k) {
        householder_step(R, Q, k, logStream, start);
    }

    return chrono::high_resolution_clock::now() - start;
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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <dimension>" << endl;
        return 1;
    }
    string dimension = argv[1];
    string folderName = dimension + "dim";
    string normalFolder = folderName + "/normal";  // Path to the normal folder within the dimension-specific folder
    _mkdir(folderName.c_str());  // Create the dimension-specific folder
    _mkdir(normalFolder.c_str());  // Create the normal folder within the dimension-specific folder

    MatrixXd A = load_matrix(folderName + "/GeneratedMatrix.txt");
    MatrixXd Q, R;
    stringstream logStream;
    ofstream fileQ(normalFolder + "/Q_Matrix.txt"), fileR(normalFolder + "/R_Matrix.txt"), fileReport(normalFolder + "/Report.txt"), fileLog(normalFolder + "/log.txt");

    if (fileQ.is_open() && fileR.is_open() && fileReport.is_open() && fileLog.is_open()) {
        init_cpu_total_usage();
        init_process_cpu_usage();
        chrono::duration<double> total_elapsed = householder_reflection(A, Q, R, logStream);

        fileQ << Q;
        fileR << R;
        double norm = (A - Q * R).norm();
        fileReport << "Is Q orthogonal? " << (Q.transpose() * Q).isIdentity() << "\n";
        fileReport << "Is A = QR? " << norm << "\n";
        fileReport << "Matrix equality: " << (norm < 1e-10 ? "True" : "False") << "\n";
        fileReport << "Total Elapsed Time: " << total_elapsed.count() << " seconds\n";

        fileLog << logStream.rdbuf();
        fileLog.close();
        fileQ.close();
        fileR.close();
        fileReport.close();
    } else {
        cerr << "Error opening output files." << endl;
    }

    cout << "Processing completed. Output saved in " << normalFolder << "." << endl;

    return 0;
}