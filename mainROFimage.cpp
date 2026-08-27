#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOMINMAX

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <blitz/array.h>
#include <random>
#include <chrono>
#include <cmath>
#include <float.h>
#include <iostream>
#include <cstring>

// Windows específico para tiempo
#ifdef _WIN32
#include <windows.h>
// Definir estructura timeval para Windows
struct timeval {
    long tv_sec;
    long tv_usec;
};

// Implementación de gettimeofday para Windows
static inline int gettimeofday(struct timeval* tv, void* tz) {
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    unsigned long long time = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    time /= 10;  // Convertir a microsegundos
    time -= 11644473600000000ULL; // Convertir a Unix epoch
    tv->tv_sec = (long)(time / 1000000ULL);
    tv->tv_usec = (long)(time % 1000000ULL);
    return 0;
}
#else
#include <sys/time.h>
#endif

// declara namespace
using namespace std;
using namespace cv;
using namespace blitz;

// funciones a utilizar en la optimizacion
const double CTE = 0.2;
void Derivada(Array<double, 2>&, Array<double, 2>, Array<double, 2>);
double minMod(double, double);
double Funcional(Array<double, 2>, Array<double, 2>);
void Print3D(Array<double, 2>, FILE*, const char*);
void boundaryCond1(Array<double, 2>&);

// variables globales
double LAMBDA;
char imgname[50];

// ============================================================
//  FUNCIONES PARA MOSTRAR/GUARDAR SIN NORMALIZAR (Opción 3)
// ============================================================

// Mostrar imagen (escala fija: asume rango 0-2)
void mostrarImagen(const char* titulo, blitz::Array<double, 2>& datos, cv::Mat& img) {
    cv::Mat img_show;
    img.convertTo(img_show, CV_8U, 127.5);  // 255/2 = 127.5
    imshow(titulo, img_show);
}

// Guardar imagen (escala fija: asume rango 0-2)
void guardarImagen(const char* archivo, blitz::Array<double, 2>& datos, cv::Mat& img) {
    cv::Mat img_show;
    img.convertTo(img_show, CV_8U, 127.5);
    imwrite(archivo, img_show);
}

// Mostrar error (asume rango 0-1, porque es |P - Po|)
void mostrarError(const char* titulo, blitz::Array<double, 2>& datos, cv::Mat& img) {
    cv::Mat img_show;
    img.convertTo(img_show, CV_8U, 255.0);  // Escala directa si está en [0,1]
    imshow(titulo, img_show);
}

// Guardar error
void guardarError(const char* archivo, blitz::Array<double, 2>& datos, cv::Mat& img) {
    cv::Mat img_show;
    img.convertTo(img_show, CV_8U, 255.0);
    imwrite(archivo, img_show);
}

// Clase para generar números aleatorios (reemplazo de ranlib)
class NormalRandom {
private:
    mt19937 gen;
    normal_distribution<double> dist;
public:
    NormalRandom(double mean = 0.0, double variance = 1.0)
        : gen(random_device{}()), dist(mean, variance) {
    }

    void seed(unsigned int s) { gen.seed(s); }
    double random() { return dist(gen); }
};

class UniformRandom {
private:
    mt19937 gen;
    uniform_real_distribution<double> dist;
public:
    UniformRandom() : gen(random_device{}()), dist(0.0, 1.0) {}
    void seed(unsigned int s) { gen.seed(s); }
    double random() { return dist(gen); }
};

//*************************************************************************
//                        inicia funcion principal
//*************************************************************************
int main(int argc, char** argv)
{
    //parametros desde consola
    if (argc == 3) {
        strcpy_s(imgname, argv[1]);
        LAMBDA = atof(argv[2]);
    }
    else {
        cout << "Uso: " << argv[0] << " imagen.png LAMBDA" << endl;
        return -1;
    }

    // despliega informacion del proceso
    cout << endl << "Inicia procesamiento..." << endl << endl;

    // lectura de imagen
    Mat IMAGE = imread(imgname, IMREAD_GRAYSCALE);
    if (IMAGE.empty()) {
        cout << "Error: No se pudo cargar la imagen " << imgname << endl;
        return -1;
    }
    cout << "Imagen cargada: " << IMAGE.rows << "x" << IMAGE.cols << endl;
    // datos de la imagen
    int renglones = IMAGE.rows, columnas = IMAGE.cols;
    cout << "Dimensiones: " << renglones << "x" << columnas << endl;

    // inicializa funciones de ruido (versión Windows)
    NormalRandom ruidoInicial;
    ruidoInicial.seed((unsigned int)time(nullptr));
    NormalRandom ruidoImagen(0.0, CTE);
    ruidoImagen.seed((unsigned int)time(nullptr));

    // separa memoria para procesar con Blitz
    Array<double, 2> Is(renglones, columnas), Ic(renglones, columnas), Po(renglones, columnas),
        Gcx(renglones, columnas), Gcy(renglones, columnas),
        Gsx(renglones, columnas), Gsy(renglones, columnas),
        dummy(renglones, columnas);
    Array<double, 2> derivP(renglones, columnas), P(renglones, columnas), P0(renglones, columnas),
        P1(renglones, columnas), P2(renglones, columnas), Phase0(renglones, columnas);

    // datos de entrada
    double valMax = -DBL_MAX, valMin = DBL_MAX, num = 0.0, den = 0.0;
    for (int r = 0; r < IMAGE.rows; r++)
        for (int c = 0; c < IMAGE.cols; c++) {
            double ruido = ruidoImagen.random();
            Po(r, c) = 2.0 * double(IMAGE.at<uchar>(r, c)) / 255.0;
            // imagen ruidosa, se anade ruido aditivo
            Phase0(r, c) = Po(r, c) + 0.5 * ruido;
            // valor inicial
            P(r, c) = Phase0(r, c);

            // calcula el SNR, mide ruido en la imagen
            num += (P(r, c) * P(r, c));
            den += ((P(r, c) - Po(r, c)) * (P(r, c) - Po(r, c)));
        }

    // despliega diferencia entre la estimacion y el valor real
    cout << endl << "SNR = " << 20.0 * log10(num / den) << " db" << endl;

    // crea manejador de imagenes con openCV
    Mat Imagen(renglones, columnas, CV_64F, (unsigned char*)dummy.data());

    const char* win0 = "Fase original";
    namedWindow(win0, WINDOW_NORMAL);
    const char* win1 = "Estimaciones";
    namedWindow(win1, WINDOW_NORMAL);
    const char* win2 = "Imagen ruidosa";
    namedWindow(win2, WINDOW_NORMAL);

    // ===== Mostrar imagen ruidosa =====
    dummy = Phase0;
    mostrarImagen(win2, Phase0, Imagen);
    guardarImagen("imagenRuidosa.png", Phase0, Imagen);

    // ===== Mostrar fase original =====
    dummy = Po;
    mostrarImagen(win0, Po, Imagen);

    // ************************************************************************
    //             Inicia procesamiento
    // ************************************************************************
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // ===== Mostrar estimación inicial =====
    dummy = P;
    mostrarImagen(win1, P, Imagen);
    waitKey(1);

    // inicia iteracion del algoritmo
    double Fx0, Fx = Funcional(P, Phase0);
    double tao = 1.0e-3;
    unsigned iter = 0;
    P2 = P;
    bool flag = true;
    cout << "Fx inicial: " << Fx << endl;
    if (!isfinite(Fx)) {
        cout << "Error: Fx es infinito o NaN" << endl;
        return -1;
    }

    cout << "Antes del while..." << endl;


    const double TOL_ESTADO_ESTACIONARIO = 1.0e-6;   // tolerancia para u_t ≈ 0
    const unsigned int MAX_ITERACIONES = 50000;      // limite de seguridad

    while (flag) {
        P0 = P;    Fx0 = Fx;

        cout << "  P0 copiado" << endl;

        Derivada(derivP, P, Phase0);
        cout << "  Derivada calculada" << endl;
        P = P - tao * derivP;
        cout << "  P actualizado" << endl;
        Fx = Funcional(P, Phase0);
        cout << "  Fx calculado: " << Fx << endl;
        double difF = fabs(Fx0 - Fx);

        // Calculo del cambio relativo (aproximacion a u_t)
        double errp = sqrt(sum(pow2(P - P0))) / sqrt(sum(pow2(P0)));
        cout << "  errp: " << errp << endl;

        if ((iter % 100) == 0) {
            cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
            dummy = P;
            mostrarImagen(win1, P, Imagen);
            waitKey(1);
        }
        //  CONDICION DE PARO
 
        if (errp < TOL_ESTADO_ESTACIONARIO) {
            cout << "\n*** ESTADO ESTACIONARIO ALCANZADO ***" << endl;
            cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
            cout << "u_t ≈ 0, la solucion ha convergido" << endl;
            flag = false;
        }
        else if (iter >= MAX_ITERACIONES) {
            cout << "\n*** MAXIMO DE ITERACIONES ALCANZADO ***" << endl;
            cout << "iteracion : " << iter << " Fx= " << Fx << " ||d_p||= " << errp << endl;
            flag = false;
        }

        iter++;
        cout << "  Fin iteracion " << iter << endl;
    }
    cout << "SALI DEL WHILE" << endl;
    gettimeofday(&end, NULL);

    double startms = double(start.tv_sec) * 1000. + double(start.tv_usec) / 1000.;
    double endms = double(end.tv_sec) * 1000. + double(end.tv_usec) / 1000.;
    double ms = endms - startms;
    cout << endl << "Tiempo empleado  : " << ms << " mili-segundos" << endl;

    double error = sqrt(sum(pow2(Po - P))) / (sqrt(sum(pow2(Po))) + sqrt(sum(pow2(P))));
    cout << endl << "Normalized error : = " << error << endl << endl;

    // =====Mostrar y guardar estimación final =====
    dummy = P;
    mostrarImagen(win1, P, Imagen);
    guardarImagen("estimacionFinal.png", P, Imagen);
    waitKey(0);

    // ===== Mostrar y guardar error =====
    dummy = fabs(P - Po);
    mostrarError("Error", dummy, Imagen);
    guardarError("Error.png", dummy, Imagen);

    cout << "Resultados guardados en archivos PNG" << endl;
    cout << "Presiona cualquier tecla para salir..." << endl;
    waitKey(0);

    return 0;
}

const double beta = 0.1;
const double eps = sqrt(DBL_EPSILON);

void Derivada(Array<double, 2>& dP, Array<double, 2> P, Array<double, 2> Phase0) {
    int columnas = P.cols();
    int renglones = P.rows();
    double lambda = LAMBDA;
    double beta_local = beta;

    // Inicializar dP con el término de fidelidad
    dP = lambda * (P - Phase0);

    // Procesar píxel por píxel
    for (int r = 0; r < renglones; r++) {
        for (int c = 0; c < columnas; c++) {

            // ===== EJE X - SUPERIOR (r+1) =====
            double V31x = 0.0;
            if (r < renglones - 1) {
                double Ux = P(r + 1, c) - P(r, c);
                double Uy = 0.0;

                // Calcular Uy solo si tenemos vecinos válidos
                if (c > 0 && c < columnas - 1) {
                    double term1 = 0.5 * (P(r + 1, c + 1) - P(r + 1, c - 1));
                    double term2 = 0.5 * (P(r, c + 1) - P(r, c - 1));
                    Uy = minMod(term1, term2);
                }
                else if (c == 0 && c < columnas - 1) {
                    // Borde izquierdo: solo un lado
                    Uy = 0.5 * (P(r + 1, c + 1) - P(r + 1, c));
                }
                else if (c == columnas - 1 && c > 0) {
                    // Borde derecho: solo un lado
                    Uy = 0.5 * (P(r + 1, c) - P(r + 1, c - 1));
                }

                double denom = sqrt(Ux * Ux + Uy * Uy + beta_local);
                V31x = (denom > 1e-10) ? Ux / denom : 0.0;
            }

            // ===== EJE X - INFERIOR (r-1) =====
            double V32x = 0.0;
            if (r > 0) {
                double Ux = P(r, c) - P(r - 1, c);
                double Uy = 0.0;

                if (c > 0 && c < columnas - 1) {
                    double term1 = 0.5 * (P(r, c + 1) - P(r, c - 1));
                    double term2 = 0.5 * (P(r - 1, c + 1) - P(r - 1, c - 1));
                    Uy = minMod(term1, term2);
                }
                else if (c == 0 && c < columnas - 1) {
                    Uy = 0.5 * (P(r, c + 1) - P(r, c));
                }
                else if (c == columnas - 1 && c > 0) {
                    Uy = 0.5 * (P(r, c) - P(r, c - 1));
                }

                double denom = sqrt(Ux * Ux + Uy * Uy + beta_local);
                V32x = (denom > 1e-10) ? Ux / denom : 0.0;
            }

            // ===== EJE Y - DERECHA (c+1) =====
            double V31y = 0.0;
            if (c < columnas - 1) {
                double Uy = P(r, c + 1) - P(r, c);
                double Ux = 0.0;

                if (r > 0 && r < renglones - 1) {
                    double term1 = 0.5 * (P(r + 1, c + 1) - P(r - 1, c + 1));
                    double term2 = 0.5 * (P(r + 1, c) - P(r - 1, c));
                    Ux = minMod(term1, term2);
                }
                else if (r == 0 && r < renglones - 1) {
                    Ux = 0.5 * (P(r + 1, c + 1) - P(r + 1, c));
                }
                else if (r == renglones - 1 && r > 0) {
                    Ux = 0.5 * (P(r, c + 1) - P(r, c));
                }

                double denom = sqrt(Ux * Ux + Uy * Uy + beta_local);
                V31y = (denom > 1e-10) ? Uy / denom : 0.0;
            }

            // ===== EJE Y - IZQUIERDA (c-1) =====
            double V32y = 0.0;
            if (c > 0) {
                double Uy = P(r, c) - P(r, c - 1);
                double Ux = 0.0;

                if (r > 0 && r < renglones - 1) {
                    double term1 = 0.5 * (P(r + 1, c) - P(r - 1, c));
                    double term2 = 0.5 * (P(r + 1, c - 1) - P(r - 1, c - 1));
                    Ux = minMod(term1, term2);
                }
                else if (r == 0 && r < renglones - 1) {
                    Ux = 0.5 * (P(r + 1, c) - P(r + 1, c - 1));
                }
                else if (r == renglones - 1 && r > 0) {
                    Ux = 0.5 * (P(r, c) - P(r, c - 1));
                }

                double denom = sqrt(Ux * Ux + Uy * Uy + beta_local);
                V32y = (denom > 1e-10) ? Uy / denom : 0.0;
            }

            // Actualizar dP con la divergencia
            dP(r, c) -= ((V31x - V32x) + (V31y - V32y));
        }
    }
}

double Funcional(Array<double, 2> P, Array<double, 2> Phase0) {
    int columnas = P.cols();
    int renglones = P.rows();
    double lambda = LAMBDA;
    double hx = 1.0 / (double(renglones) - 1.0);
    double hy = 1.0 / (double(columnas) - 1.0);

    double dx, dy, v1;
    double suma = 0.0;

    for (int r = 0; r < renglones; r++) {
        for (int c = 0; c < columnas; c++) {
            if (r == renglones - 1)
                dx = P(r, c) - P(r - 1, c);
            else if (r == 0)
                dx = P(r + 1, c) - P(r, c);
            else
                dx = 0.5 * (P(r + 1, c) - P(r - 1, c));

            if (c == columnas - 1)
                dy = P(r, c) - P(r, c - 1);
            else if (c == 0)
                dy = P(r, c + 1) - P(r, c);
            else
                dy = 0.5 * (P(r, c + 1) - P(r, c - 1));

            v1 = P(r, c) - Phase0(r, c);
            suma += 0.5 * lambda * (v1 * v1) + sqrt(dx * dx + dy * dy);
        }
    }

    return suma * hx * hy;
}

void Print3D(Array<double, 2> Z, FILE* salida, const char* fileName) {
    fprintf(salida, "set terminal postscript eps enhanced rounded\n");
    fprintf(salida, "set output \"%s\"\n", fileName);
    fprintf(salida, "set style line 1 linetype -1 linewidth 1\n");
    fprintf(salida, "set xlabel \"columns (pixels)\" offset -1,-1\n");
    fprintf(salida, "set ylabel \"rows (pixels)\" offset -1,-1\n");
    fprintf(salida, "set zlabel \"phase\"\n");
    fprintf(salida, "set xrange [%f:%f]\n", 0., float(Z.cols()));
    fprintf(salida, "set yrange [%f:%f]\n", 0., float(Z.rows()));
    fprintf(salida, "set xtics 100 offset -0.5,-0.5\n");
    fprintf(salida, "set ytics 100 offset -0.5,-0.5\n");
    fprintf(salida, "set view 70, 210\n");
    fprintf(salida, "unset key\n");
    fprintf(salida, "unset colorbox\n");
    fprintf(salida, "set hidden3d front\n");
    fprintf(salida, "splot '-' using 1:2:3 title '' with lines lt -1 lw 0.1\n");
    for (int c = 0; c < Z.cols(); c += 8) {
        for (int r = 0; r < Z.rows(); r += 8)
            fprintf(salida, "%f %f %f\n", float(c), float(r), float(Z(r, c)));
        fprintf(salida, "\n");
    }
    fprintf(salida, "e\n");
    fflush(salida);
}

void boundaryCond1(Array<double, 2>& T) {
    int columnas = T.cols();
    int renglones = T.rows();
    blitz::Range all = blitz::Range::all();

    T(0, all) = T(1, all);
    T(renglones - 1, all) = T(renglones - 2, all);
    T(all, 0) = T(all, 1);
    T(all, columnas - 1) = T(all, columnas - 2);
    T(0, 0) = T(1, 1);
    T(0, columnas - 1) = T(1, columnas - 2);
    T(renglones - 1, 0) = T(renglones - 2, 1);
    T(renglones - 1, columnas - 1) = T(renglones - 2, columnas - 2);
}

double minMod(double a, double b) {
    double signa = (a > 0.0) ? 1.0 : ((a < 0.0) ? -1.0 : 0.0);
    double signb = (b > 0.0) ? 1.0 : ((b < 0.0) ? -1.0 : 0.0);
    double minim = (fabs(a) <= fabs(b)) ? fabs(a) : fabs(b);
    return ((signa + signb) * minim / 2.0);
}