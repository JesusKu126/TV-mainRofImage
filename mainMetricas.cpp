#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// MSE

double calcularMSE(const Mat& referencia, const Mat& imagen)
{
    Mat diferencia;
    absdiff(referencia, imagen, diferencia);

    diferencia.convertTo(diferencia, CV_64F);
    diferencia = diferencia.mul(diferencia);

    return mean(diferencia)[0];
}

// RMSE 

double calcularRMSE(const Mat& referencia, const Mat& imagen)
{
    return sqrt(calcularMSE(referencia, imagen));
}

// PSNR 

double calcularPSNR(const Mat& referencia, const Mat& imagen)
{
    double mse = calcularMSE(referencia, imagen);

    if (mse <= 1e-10)
        return INFINITY;

    const double MAX_I = 255.0;

    return 10.0 * log10((MAX_I * MAX_I) / mse);
}


// SSIM

double calcularSSIM(const Mat& referencia, const Mat& imagen)
{
    Mat I1, I2;

    referencia.convertTo(I1, CV_64F);
    imagen.convertTo(I2, CV_64F);

    // Constantes de SSIM
    const double C1 = pow(0.01 * 255.0, 2);
    const double C2 = pow(0.03 * 255.0, 2);

    // Medias
    Mat mu1, mu2;

    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    // Cuadrados de las medias
    Mat mu1_sq = mu1.mul(mu1);
    Mat mu2_sq = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    // Varianzas
    Mat sigma1_sq, sigma2_sq, sigma12;

    GaussianBlur(I1.mul(I1), sigma1_sq, Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    GaussianBlur(I2.mul(I2), sigma2_sq, Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    GaussianBlur(I1.mul(I2), sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    // Fórmula SSIM
    Mat numerador1 = 2 * mu1_mu2 + C1;
    Mat numerador2 = 2 * sigma12 + C2;

    Mat denominador1 = mu1_sq + mu2_sq + C1;
    Mat denominador2 = sigma1_sq + sigma2_sq + C2;

    Mat ssimMap;
    divide(numerador1.mul(numerador2),
        denominador1.mul(denominador2),
        ssimMap);

    return mean(ssimMap)[0];
}

// SNR

double calcularSNR(const Mat& referencia, const Mat& imagen)
{
    Mat ref64, diferencia;

    referencia.convertTo(ref64, CV_64F);

    absdiff(referencia, imagen, diferencia);
    diferencia.convertTo(diferencia, CV_64F);

    Mat refCuadrada = ref64.mul(ref64);
    Mat ruidoCuadrado = diferencia.mul(diferencia);

    double potenciaSenal = mean(refCuadrada)[0];
    double potenciaRuido = mean(ruidoCuadrado)[0];

    if (potenciaRuido <= 1e-10)
        return INFINITY;

    return 10.0 * log10(potenciaSenal / potenciaRuido);
}


// CNR
double calcularCNR(const Mat& imagen)
{
    int ancho = imagen.cols;
    int alto = imagen.rows;

    // ROI 1
    Rect roi1(
        ancho / 4,
        alto / 3,
        ancho / 8,
        alto / 6
    );

    // ROI 2
    Rect roi2(
        ancho * 5 / 8,
        alto / 3,
        ancho / 8,
        alto / 6
    );

    Mat region1 = imagen(roi1);
    Mat region2 = imagen(roi2);

    Scalar media1, desviacion1;
    Scalar media2, desviacion2;

    meanStdDev(region1, media1, desviacion1);
    meanStdDev(region2, media2, desviacion2);

    double contraste = abs(media1[0] - media2[0]);

    // Promedio de las desviaciones estándar
    double ruido = (desviacion1[0] + desviacion2[0]) / 2.0;

    if (ruido <= 1e-10)
        return INFINITY;

    return contraste / ruido;
}


// EPI

double calcularEPI(const Mat& referencia, const Mat& imagen)
{
    Mat ref64, img64;

    referencia.convertTo(ref64, CV_64F);
    imagen.convertTo(img64, CV_64F);

    Mat gxRef, gyRef;
    Mat gxImg, gyImg;

    Sobel(ref64, gxRef, CV_64F, 1, 0, 3);
    Sobel(ref64, gyRef, CV_64F, 0, 1, 3);

    Sobel(img64, gxImg, CV_64F, 1, 0, 3);
    Sobel(img64, gyImg, CV_64F, 0, 1, 3);

    Mat gradRef, gradImg;

    magnitude(gxRef, gyRef, gradRef);
    magnitude(gxImg, gyImg, gradImg);

    double mediaRef = mean(gradRef)[0];

    if (mediaRef <= 1e-10)
        return 0.0;

    double mediaImg = mean(gradImg)[0];

    return mediaImg / mediaRef;
}
// Mostrar resultados

void imprimirResultados(
    const string& nombre,
    double mse,
    double rmse,
    double psnr,
    double ssim,
    double snr,
    double cnr,
    double epi)
{
    cout << left << setw(15) << nombre
        << right << setw(15) << fixed << setprecision(4) << mse
        << setw(15) << rmse
        << setw(15) << psnr
        << setw(15) << ssim
        << setw(15) << snr
        << setw(15) << cnr
        << setw(15) << epi
        << endl;
}

int main()
{

    // Cargar imágenes


    Mat original = imread("original.png", IMREAD_GRAYSCALE);
    Mat ruidosa = imread("ruidosa.png", IMREAD_GRAYSCALE);
    Mat filtrada = imread("filtrada.png", IMREAD_GRAYSCALE);

    if (original.empty())
    {
        cerr << "Error: no se pudo cargar original.png" << endl;
        return -1;
    }

    if (ruidosa.empty())
    {
        cerr << "Error: no se pudo cargar ruidosa.png" << endl;
        return -1;
    }

    if (filtrada.empty())
    {
        cerr << "Error: no se pudo cargar filtrada.png" << endl;
        return -1;
    }


    if (original.size() != ruidosa.size() ||
        original.size() != filtrada.size())
    {
        cerr << "Error: las tres imagenes deben tener "
            << "las mismas dimensiones." << endl;

        return -1;
    }

    // Métricas de la imagen RUIDOSA
    double mseRuidosa = calcularMSE(original, ruidosa);
    double rmseRuidosa = calcularRMSE(original, ruidosa);
    double psnrRuidosa = calcularPSNR(original, ruidosa);
    double ssimRuidosa = calcularSSIM(original, ruidosa);
    double snrRuidosa = calcularSNR(original, ruidosa);
    double cnrRuidosa = calcularCNR(ruidosa);
    double epiRuidosa = calcularEPI(original, ruidosa);
    
    // Métricas de la imagen FILTRADA

    double mseFiltrada = calcularMSE(original, filtrada);
    double rmseFiltrada = calcularRMSE(original, filtrada);
    double psnrFiltrada = calcularPSNR(original, filtrada);
    double ssimFiltrada = calcularSSIM(original, filtrada);
    double snrFiltrada = calcularSNR(original, filtrada);
    double cnrFiltrada = calcularCNR(filtrada);
    double epiFiltrada = calcularEPI(original, filtrada);


    // IMPRIMIR TABLA

    cout << endl;

    cout << "================================================================================================================" << endl;
    cout << "                              EVALUACION DEL FILTRADO" << endl;
    cout << "================================================================================================================" << endl;

    cout << left
        << setw(15) << "Imagen"
        << right
        << setw(15) << "MSE"
        << setw(15) << "RMSE"
        << setw(15) << "PSNR(dB)"
        << setw(15) << "SSIM"
        << setw(15) << "SNR(dB)"
        << setw(15) << "CNR"
        << setw(15) << "EPI"
        << endl;

    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

    imprimirResultados(
        "Ruidosa",
        mseRuidosa,
        rmseRuidosa,
        psnrRuidosa,
        ssimRuidosa,
        snrRuidosa,
        cnrRuidosa,
        epiRuidosa
    );

    imprimirResultados(
        "Filtrada",
        mseFiltrada,
        rmseFiltrada,
        psnrFiltrada,
        ssimFiltrada,
        snrFiltrada,
        cnrFiltrada,
        epiFiltrada
    );

    cout << "================================================================================================================" << endl;

    // ========================================================
    // Mejoras
    // ========================================================

    cout << endl;
    cout << "                    CAMBIO DESPUES DEL FILTRADO" << endl;
    cout << "================================================================" << endl;

    cout << fixed << setprecision(4);

    cout << "Reduccion del MSE: "
        << ((mseRuidosa - mseFiltrada) / mseRuidosa) * 100
        << "%" << endl;

    cout << "Mejora del PSNR: "
        << psnrFiltrada - psnrRuidosa
        << " dB" << endl;

    cout << "Mejora del SSIM: "
        << ssimFiltrada - ssimRuidosa
        << endl;

    cout << "Mejora del SNR: "
        << snrFiltrada - snrRuidosa
        << " dB" << endl;

    cout << "Cambio del CNR: "
        << cnrFiltrada - cnrRuidosa
        << endl;

    cout << "Cambio del EPI: "
        << epiFiltrada - epiRuidosa
        << endl;

    cout << "================================================================" << endl;

    return 0;
}