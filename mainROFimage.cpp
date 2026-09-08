#include <opencv2/core/core.hpp>                // OpenCV      
#include <opencv2/highgui/highgui.hpp>           
#include <iostream>                                   
#include <fstream> 
#include <cmath>                         // funciones matematicas
#include <float.h>                          // mathematical constants
#include <windows.h>                        // funciones de tiempo para Windows

// declara namespace
using namespace std;
using namespace cv;

long int renglones, columnas;
double* U0; //Variable global para imagen ruidosa
double TAO = 0.001, EPSILON = 1.0e-8, coefR = 15.0, BETA, LAMBDA;
int MAXITER = 200000, K = 5;
char imgnameOriginal[50];
char imgnameNoisy[50];

// funciones a utilizar en la optimizacion
double Funcional(double* P, double* Phase0);
void boundaryCond1(double* T, int renglones, int columnas);
void error_relativo(double* errorRe, double* Re, double* Reo);
void punto_fijo(double* U_h, double* U0, double* Uaux_h);
void call_punto_fijo(double* U_h, double* U0, double* Uaux_h);
void iteracion_Gauss_Seidel(double* U_h, double* U0, double* Uaux_h);
double MSE(double* P, double* Po);
double IFI(double* P, double* Po);
double NEI(double* P, double* Po);

//*************************************************************************
//                        inicia funcion principal
//*************************************************************************
int main(int argc, char** argv)
{
    //parametros desde consola
    if (argc == 5)
    {
        //version para lectura de imagen
        strcpy(imgnameOriginal, argv[1]);
        strcpy(imgnameNoisy, argv[2]);
        BETA = atof(argv[3]);
        LAMBDA = atof(argv[4]);
    }
    else
    {
        // Valores por defecto para pruebas en Windows
        strcpy(imgnameOriginal, "original.png");
        strcpy(imgnameNoisy, "noisy.png");
        BETA = 1.0;
        LAMBDA = 0.1;
    }

    // despliega informacion del proceso
    cout << endl << "Inicia procesamiento..." << endl << endl;
    cout << endl << "Lee datos ruidosos..." << endl << endl;

    // Leemos datos de la imagen
    Mat IMAGEN_Noisy = imread(imgnameNoisy, IMREAD_GRAYSCALE);
    if (!IMAGEN_Noisy.data)
    {
        cout << "Error en la lectura de la imagen inicial..." << endl;
        return -1;
    }
    renglones = IMAGEN_Noisy.rows;
    columnas = IMAGEN_Noisy.cols;

    Mat IMAGEN_Original = imread(imgnameOriginal, IMREAD_GRAYSCALE);
    if (!IMAGEN_Original.data)
    {
        cout << "Error en la lectura de la imagen original..." << endl;
        return -1;
    }

    // Arreglos para calculos numericos
    double* U_h, * Uo_h, * Uaux_h, * U_original;
    long int size_matrix = renglones * columnas;
    size_t size_matrix_bytes = size_matrix * sizeof(double);

    U_h = (double*)malloc(size_matrix_bytes);
    Uo_h = (double*)malloc(size_matrix_bytes);
    Uaux_h = (double*)malloc(size_matrix_bytes);
    U0 = (double*)malloc(size_matrix_bytes);
    U_original = (double*)malloc(size_matrix_bytes);

    // datos de entrada
    for (long int r = 0; r < renglones; r++)
        for (long int c = 0; c < columnas; c++)
        {
            long int idx_r_c = r * columnas + c;
            Uo_h[idx_r_c] = 1.0 * (double(IMAGEN_Noisy.at<unsigned char>(r, c))) / 255.0;
            U_original[idx_r_c] = 1.0 * (double(IMAGEN_Original.at<unsigned char>(r, c))) / 255.0;

            U0[idx_r_c] = Uo_h[idx_r_c];

            // valor inicial
            U_h[idx_r_c] = Uo_h[idx_r_c];
        }

    // ************************************************************************
    //             Inicia procesamiento - Punto Fijo
    // ************************************************************************
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    // Llama al método de punto fijo
    call_punto_fijo(U_h, U0, Uaux_h);

    // termina funcion, calcula y despliega valores indicadores del proceso  
    QueryPerformanceCounter(&end);

    // Calcula tiempo en milisegundos
    double ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << endl << "Tiempo empleado  : " << ms << " mili-segundos" << endl;

    // Guarda la imagen resultante
    Mat ImagenSalida(renglones, columnas, CV_64F);
    for (int r = 0; r < renglones; r++)
        for (int c = 0; c < columnas; c++)
        {
            long int idx_r_c = r * columnas + c;
            ImagenSalida.at<double>(r, c) = U_h[idx_r_c];
        }

    // Convierte a 8 bits para guardar
    Mat Imagen8U;
    ImagenSalida.convertTo(Imagen8U, CV_8U, 255.0);
    imwrite("estimacionFinal.png", Imagen8U);

    // Calculo de errores
    double mse = MSE(U_original, U_h);
    double nei = NEI(U_original, U_h);
    double ifi = IFI(U_original, U_h);

    cout << "Errores en las estimaciones" << endl;
    cout << "MSE : " << mse << endl;
    cout << "NEI : " << nei << endl;
    cout << "IFI : " << ifi << endl;

    // Libera memoria
    free(U_h);
    free(Uo_h);
    free(Uaux_h);
    free(U0);
    free(U_original);

    // termina ejecucion del programa
    return 0;
}

// ***************************************************************
//   Condiciones de frontera Neumann
// ***************************************************************
void boundaryCond1(double* T, int renglones, int columnas)
{
    // condiciones de frontera
    // T(0, all) = T(1, all);
    // T(renglones - 1, all) = T(renglones - 2, all);
    for (int c = 0; c < columnas; c++) {
        long int idx_0_c = c;
        long int idx_1_c = columnas + c;
        long int idx_rm1_c = (renglones - 1) * columnas + c;
        long int idx_rm2_c = (renglones - 2) * columnas + c;

        T[idx_0_c] = T[idx_1_c];
        T[idx_rm1_c] = T[idx_rm2_c];
    }

    // T(all, 0) = T(all, 1);
    // T(all, columnas - 1) = T(all, columnas - 2);
    for (int r = 0; r < renglones; r++) {
        long int idx_r_0 = r * columnas;
        long int idx_r_1 = r * columnas + 1;
        long int idx_r_cm1 = r * columnas + columnas - 1;
        long int idx_r_cm2 = r * columnas + columnas - 2;

        T[idx_r_0] = T[idx_r_1];
        T[idx_r_cm1] = T[idx_r_cm2];
    }

    // T(0, 0) = T(1, 1);
    T[0] = T[columnas + 1];
    // T(0, columnas - 1) = T(1, columnas - 2);
    T[columnas - 1] = T[columnas + columnas - 2];
    // T(renglones - 1, 0) = T(renglones - 2, 1);
    T[(renglones - 1) * columnas] = T[(renglones - 2) * columnas + 1];
    // T(renglones - 1, columnas - 1) = T(renglones - 2, columnas - 2);
    T[(renglones - 1) * columnas + columnas - 1] = T[(renglones - 2) * columnas + columnas - 2];
}

//***************************************************
// Error relativo
//***************************************************
void error_relativo(double* errorRe, double* Re, double* Reo)
{
    double sum_pow2difRRo = 0.0, sum_pow2Reo = 0.0;
    long int SizeImage = renglones * columnas;

    for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
    {
        double vRe = Re[idx_r_c];
        double vReo = Reo[idx_r_c];
        double difRe = vRe - vReo;
        sum_pow2difRRo += difRe * difRe;
        sum_pow2Reo += vReo * vReo;
    }

    *errorRe = sqrt(sum_pow2difRRo) / sqrt(sum_pow2Reo);
}

//********************************************************
// Normalized Error Index (NEI)
//********************************************************
double NEI(double* P, double* Po)
{
    double sum_pow2difPPo = 0.0, sum_pow2varP = 0.0, sum_pow2varPo = 0.0;
    long int SizeImage = renglones * columnas;

    for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
    {
        double difPPo = P[idx_r_c] - Po[idx_r_c];
        double varP = P[idx_r_c];
        double varPo = Po[idx_r_c];
        sum_pow2difPPo += difPPo * difPPo;
        sum_pow2varP += varP * varP;
        sum_pow2varPo += varPo * varPo;
    }

    double nei = sqrt(sum_pow2difPPo) / (sqrt(sum_pow2varP) + sqrt(sum_pow2varPo));
    return nei;
}

//***************************************************
// Image Fidelity Index
//***************************************************
double IFI(double* P, double* Po)
{
    double sum_pow2difPPo = 0.0, sum_pow2varP = 0.0;
    long int SizeImage = renglones * columnas;

    for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
    {
        double difPPo = P[idx_r_c] - Po[idx_r_c];
        double varP = P[idx_r_c];
        sum_pow2difPPo += difPPo * difPPo;
        sum_pow2varP += varP * varP;
    }

    double iqi = 1.0 - (sum_pow2difPPo / sum_pow2varP);
    return iqi;
}

//***************************************************
// MSE
//***************************************************
double MSE(double* P, double* Po)
{
    double sum_pow2difPPo = 0.0;
    long int SizeImage = renglones * columnas;

    for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
    {
        double difPPo = P[idx_r_c] - Po[idx_r_c];
        sum_pow2difPPo += difPPo * difPPo;
    }

    double mse = sum_pow2difPPo / (double(SizeImage));
    return mse;
}

//***************************************************
// Funcional
//*************************************************************************
double Funcional(double* P, double* Phase0)
{
    double lambda = LAMBDA;
    double hx = 1.0 / (double(renglones) - 1.0);
    double hy = 1.0 / (double(columnas) - 1.0);

    double dx, dy, v1;
    double suma = 0.0;

    for (long int r = 0; r < renglones; r++)
        for (long int c = 0; c < columnas; c++)
        {
            long int idx_r_c = r * columnas + c;
            long int idx_rp1_c = (r + 1) * columnas + c;
            long int idx_rm1_c = (r - 1) * columnas + c;
            long int idx_r_cp1 = r * columnas + c + 1;
            long int idx_r_cm1 = r * columnas + c - 1;

            // evalua derivadas, en x
            if (r == renglones - 1)
            {
                dx = P[idx_r_c] - P[idx_rm1_c];
            }
            else if (r == 0)
            {
                dx = P[idx_rp1_c] - P[idx_r_c];
            }
            else
            {
                dx = 0.5 * (P[idx_rp1_c] - P[idx_rm1_c]);
            }

            // evalua derivadas, en y
            if (c == columnas - 1)
            {
                dy = P[idx_r_c] - P[idx_r_cm1];
            }
            else if (c == 0)
            {
                dy = P[idx_r_cp1] - P[idx_r_c];
            }
            else
            {
                dy = 0.5 * (P[idx_r_cp1] - P[idx_r_cm1]);
            }

            // evalua terminos de similitud 
            v1 = P[idx_r_c] - Phase0[idx_r_c];

            // evalua funcional en (r,c)        
            suma += 0.5 * lambda * (v1 * v1) + sqrt(dx * dx + dy * dy);
        }

    return suma * hx * hy;
}

//***************************************************
// Iteracion de Gauss-Seidel para punto fijo
//***************************************************
void iteracion_Gauss_Seidel(double* U_h, double* U0, double* Uaux_h)
{
    long int SizeImage = renglones * columnas;
    double den, num, beta = BETA, lambda = LAMBDA;
    double A, B, D, Ux, Uy;

    for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
        Uaux_h[idx_r_c] = U_h[idx_r_c];

    for (long int r = 0; r < renglones; r++)
        for (long int c = 0; c < columnas; c++)
        {
            long int idx_r_c = r * columnas + c;
            long int idx_rp1_c = (r + 1) * columnas + c;
            long int idx_rm1_c = (r - 1) * columnas + c;
            long int idx_r_cp1 = r * columnas + c + 1;
            long int idx_r_cm1 = r * columnas + c - 1;
            long int idx_rm1_cp1 = (r - 1) * columnas + c + 1;
            long int idx_rp1_cm1 = (r + 1) * columnas + c - 1;

            // A = 1/sqrt(|∇u|^2 + β)
            Ux = Uaux_h[idx_rp1_c] - Uaux_h[idx_r_c];
            Uy = Uaux_h[idx_r_cp1] - Uaux_h[idx_r_c];
            A = 1.0 / sqrt(Ux * Ux + Uy * Uy + BETA);

            Ux = Uaux_h[idx_r_c] - Uaux_h[idx_rm1_c];
            Uy = Uaux_h[idx_rm1_cp1] - Uaux_h[idx_rm1_c];
            B = 1.0 / sqrt(Ux * Ux + Uy * Uy + BETA);

            Ux = Uaux_h[idx_rp1_cm1] - Uaux_h[idx_r_cm1];
            Uy = Uaux_h[idx_r_c] - Uaux_h[idx_r_cm1];
            D = 1.0 / sqrt(Ux * Ux + Uy * Uy + BETA);

            // Ecuación de Gauss-Seidel
            num = A * Uaux_h[idx_rp1_c] + B * U_h[idx_rm1_c] + A * Uaux_h[idx_r_cp1] + D * U_h[idx_r_cm1] + lambda * U0[idx_r_c];
            den = 2.0 * A + B + D + lambda;

            U_h[idx_r_c] = num / den;
        }
}

//***************************************************
// Metodo de punto fijo
//***************************************************
void punto_fijo(double* U_h, double* U0, double* Uaux_h)
{
    long int SizeImage = renglones * columnas;

    // Condiciones de frontera
    boundaryCond1(U_h, renglones, columnas);

    // K iteraciones de Gauss-Seidel
    for (int k = 0; k < K; k++)
        iteracion_Gauss_Seidel(U_h, U0, Uaux_h);
}

//***************************************************
// Funcion para llamar al Metodo de punto fijo
//***************************************************
void call_punto_fijo(double* U_h, double* U0, double* Uaux_h)
{
    // Inicia iteracion del algoritmo
    long int SizeImage = renglones * columnas;
    double errp, Fx0, Fx = Funcional(U_h, U0);
    double epsilon = EPSILON;
    unsigned iter = 0;

    bool flag = true;
    while (flag)
    {
        for (long int idx_r_c = 0; idx_r_c < SizeImage; idx_r_c++)
        {
            Uaux_h[idx_r_c] = U_h[idx_r_c];
        }

        punto_fijo(U_h, U0, Uaux_h);
        Fx0 = Fx;

        Fx = Funcional(U_h, U0);
        double difF = fabs(Fx0 - Fx);

        // Calcula error de la estimación
        error_relativo(&errp, U_h, Uaux_h);

        if ((iter % 100) == 0)
        {
            cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
        }

        // Criterios de paro
        if ((iter >= MAXITER) || (errp < epsilon) || (difF < epsilon * errp))
        {
            cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
            flag = false;
        }

        iter++;
    }
}