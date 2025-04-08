#include <iostream>
#include <Eigen/Eigen>
#include <vector>

using namespace std;
using namespace Eigen;

/// Funzione per il calcolo dell'errore relativo
double calcErrRelativo(const VectorXd& x, const VectorXd& xEsatto) {
    return (x - xEsatto).norm() / xEsatto.norm();
}

// Funzione per risoluzione via scomposizione QR
VectorXd solveQR(const MatrixXd& A, const VectorXd& b) {
    return A.colPivHouseholderQr().solve(b);
}

// Funzione per risoluzione via scomposizione PA=LU
VectorXd solvePALU(const MatrixXd& A, const VectorXd& b) {
    // pivoting parziale: A deve essere invertibile
    PartialPivLU<MatrixXd> lu(A);
    return lu.solve(b);
}

int main() {
    // Definisco la soluzione esatta
    VectorXd xTrue(2);
    xTrue << -1.0e+00, -1.0e+00;

    // Definisco le variabili per memorizzare i sistemi
    vector<MatrixXd> matrice(3, MatrixXd(2, 2));
    vector<VectorXd> vettore(3, VectorXd(2));

    // Inizializzo matrici e vettori
    matrice[0] << 5.547001962252291e-01, -3.770900990025203e-02,
                     8.320502943378437e-01, -9.992887623566787e-01;
    vettore[0] << -5.169911863249772e-01, 1.672384680188350e-01;

    matrice[1] << 5.547001962252291e-01, -5.540607316466765e-01,
                     8.320502943378437e-01, -8.324762492991313e-01;
    vettore[1] << -6.394645785530173e-04, 4.259549612877223e-04;

    matrice[2] << 5.547001962252291e-01, -5.547001955851905e-01,
                     8.320502943378437e-01, -8.320502947645361e-01;
    vettore[2] << -6.400391328043042e-10, 4.266924591433963e-10;


    for (size_t i = 0; i < matrice.size(); ++i) {
        const MatrixXd& A = matrice[i];     // Reference
        const VectorXd& b = vettore[i];     // Reference

        // QR
        VectorXd solutionQR = solveQR(A, b);
        double errorQR = calcErrRelativo(solutionQR, xTrue);

        // PALU
        VectorXd solutionPALU = solvePALU(A, b);
        double errorPALU = calcErrRelativo(solutionPALU, xTrue);
        
        // Stampa a schermo
        cout << "Sistema n. " << (i + 1) << endl;

        cout << "   ( QR ): x = " << solutionQR.transpose() 
             << ", errR = " << errorQR << endl;

        cout << "   (PALU): x = " << solutionPALU.transpose() 
             << ", errR = " << errorPALU << endl;

        cout << "   ErrR minore: " ;
            // Stampa il metodo con errore relativo minore
            if (errorQR < errorPALU) {
                cout << "QR (" << errorQR << ")" << endl;
            } else {
                cout << "PALU (" << errorPALU << ")" << endl;
            }
    }

    return 0;
}
