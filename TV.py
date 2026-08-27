import sys
import time
import math
import numpy as np
import cv2

CTE = 0.2
BETA = 0.1


def minmod(a, b):
    signa = 1.0 if a > 0.0 else (-1.0 if a < 0.0 else 0.0)
    signb = 1.0 if b > 0.0 else (-1.0 if b < 0.0 else 0.0)
    minim = min(abs(a), abs(b))
    return (signa + signb) * minim / 2.0


#  FUNCIONES PARA MOSTRAR/GUARDAR SIN NORMALIZAR


def mostrar_imagen(titulo, P):
    """Muestra imagen asumiendo rango de datos [0, 2]."""
    img_show = np.clip(P * 127.5, 0, 255).astype(np.uint8)  # 255/2 = 127.5
    cv2.imshow(titulo, img_show)


def guardar_imagen(archivo, P):
    img_show = np.clip(P * 127.5, 0, 255).astype(np.uint8)
    cv2.imwrite(archivo, img_show)


def mostrar_error(titulo, D):
    """Muestra el mapa de error asumiendo rango [0, 1]."""
    img_show = np.clip(D * 255.0, 0, 255).astype(np.uint8)
    cv2.imshow(titulo, img_show)


def guardar_error(archivo, D):
    img_show = np.clip(D * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(archivo, img_show)



#  FUNCIONES DE TRABAJO (equivalentes a Derivada / Funcional)

def derivada(P, Phase0, lam):
    renglones, columnas = P.shape
    dP = lam * (P - Phase0)

    for r in range(renglones):
        for c in range(columnas):

            # EJE X - SUPERIOR (r+1)
            V31x = 0.0
            if r < renglones - 1:
                Ux = P[r + 1, c] - P[r, c]
                Uy = 0.0
                if 0 < c < columnas - 1:
                    term1 = 0.5 * (P[r + 1, c + 1] - P[r + 1, c - 1])
                    term2 = 0.5 * (P[r, c + 1] - P[r, c - 1])
                    Uy = minmod(term1, term2)
                elif c == 0 and c < columnas - 1:
                    Uy = 0.5 * (P[r + 1, c + 1] - P[r + 1, c])
                elif c == columnas - 1 and c > 0:
                    Uy = 0.5 * (P[r + 1, c] - P[r + 1, c - 1])
                denom = math.sqrt(Ux * Ux + Uy * Uy + BETA)
                V31x = Ux / denom if denom > 1e-10 else 0.0

            #  EJE X - INFERIOR (r-1)
            V32x = 0.0
            if r > 0:
                Ux = P[r, c] - P[r - 1, c]
                Uy = 0.0
                if 0 < c < columnas - 1:
                    term1 = 0.5 * (P[r, c + 1] - P[r, c - 1])
                    term2 = 0.5 * (P[r - 1, c + 1] - P[r - 1, c - 1])
                    Uy = minmod(term1, term2)
                elif c == 0 and c < columnas - 1:
                    Uy = 0.5 * (P[r, c + 1] - P[r, c])
                elif c == columnas - 1 and c > 0:
                    Uy = 0.5 * (P[r, c] - P[r, c - 1])
                denom = math.sqrt(Ux * Ux + Uy * Uy + BETA)
                V32x = Ux / denom if denom > 1e-10 else 0.0

            #  EJE Y - DERECHA (c+1)
            V31y = 0.0
            if c < columnas - 1:
                Uy = P[r, c + 1] - P[r, c]
                Ux = 0.0
                if 0 < r < renglones - 1:
                    term1 = 0.5 * (P[r + 1, c + 1] - P[r - 1, c + 1])
                    term2 = 0.5 * (P[r + 1, c] - P[r - 1, c])
                    Ux = minmod(term1, term2)
                elif r == 0 and r < renglones - 1:
                    Ux = 0.5 * (P[r + 1, c + 1] - P[r + 1, c])
                elif r == renglones - 1 and r > 0:
                    Ux = 0.5 * (P[r, c + 1] - P[r, c])
                denom = math.sqrt(Ux * Ux + Uy * Uy + BETA)
                V31y = Uy / denom if denom > 1e-10 else 0.0

            # EJE Y - IZQUIERDA (c-1) 
            V32y = 0.0
            if c > 0:
                Uy = P[r, c] - P[r, c - 1]
                Ux = 0.0
                if 0 < r < renglones - 1:
                    term1 = 0.5 * (P[r + 1, c] - P[r - 1, c])
                    term2 = 0.5 * (P[r + 1, c - 1] - P[r - 1, c - 1])
                    Ux = minmod(term1, term2)
                elif r == 0 and r < renglones - 1:
                    Ux = 0.5 * (P[r + 1, c] - P[r + 1, c - 1])
                elif r == renglones - 1 and r > 0:
                    Ux = 0.5 * (P[r, c] - P[r, c - 1])
                denom = math.sqrt(Ux * Ux + Uy * Uy + BETA)
                V32y = Uy / denom if denom > 1e-10 else 0.0

            dP[r, c] -= ((V31x - V32x) + (V31y - V32y))

    return dP


def funcional(P, Phase0, lam):
    renglones, columnas = P.shape
    hx = 1.0 / (renglones - 1.0)
    hy = 1.0 / (columnas - 1.0)

    suma = 0.0
    for r in range(renglones):
        for c in range(columnas):
            if r == renglones - 1:
                dx = P[r, c] - P[r - 1, c]
            elif r == 0:
                dx = P[r + 1, c] - P[r, c]
            else:
                dx = 0.5 * (P[r + 1, c] - P[r - 1, c])

            if c == columnas - 1:
                dy = P[r, c] - P[r, c - 1]
            elif c == 0:
                dy = P[r, c + 1] - P[r, c]
            else:
                dy = 0.5 * (P[r, c + 1] - P[r, c - 1])

            v1 = P[r, c] - Phase0[r, c]
            suma += 0.5 * lam * (v1 * v1) + math.sqrt(dx * dx + dy * dy)

    return suma * hx * hy


def boundary_cond1(T):
    """Copia el valor del borde interior al borde exterior (no usada en el
    flujo principal, se incluye por completitud, igual que en el original)."""
    T[0, :] = T[1, :]
    T[-1, :] = T[-2, :]
    T[:, 0] = T[:, 1]
    T[:, -1] = T[:, -2]
    T[0, 0] = T[1, 1]
    T[0, -1] = T[1, -2]
    T[-1, 0] = T[-2, 1]
    T[-1, -1] = T[-2, -2]
    return T

def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} imagen.png LAMBDA")
        sys.exit(-1)

    imgname = sys.argv[1]
    lam = float(sys.argv[2])

    print("\nInicia procesamiento...\n")

    image = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {imgname}")
        sys.exit(-1)

    renglones, columnas = image.shape
    print(f"Imagen cargada: {renglones}x{columnas}")
    print(f"Dimensiones: {renglones}x{columnas}")

    rng = np.random.default_rng()

    # datos de entrada
    Po = 2.0 * image.astype(np.float64) / 255.0
    ruido = rng.normal(0.0, CTE, size=(renglones, columnas))
    Phase0 = Po + 0.5 * ruido        # imagen ruidosa (ruido aditivo)
    P = Phase0.copy()                # valor inicial

    # calcula el SNR
    num = float(np.sum(P * P))
    den = float(np.sum((P - Po) ** 2))
    print(f"\nSNR = {20.0 * math.log10(num / den)} db")

    win0, win1, win2 = "Fase original", "Estimaciones", "Imagen ruidosa"
    cv2.namedWindow(win0, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)

    mostrar_imagen(win2, Phase0)
    guardar_imagen("imagenRuidosa.png", Phase0)

    mostrar_imagen(win0, Po)


    start = time.time()


    mostrar_imagen(win1, P)
    cv2.waitKey(1)

    Fx = funcional(P, Phase0, lam)
    epsilon = 1.0e-8
    tao = 1.0e-3
    iter_ = 0
    flag = True
    print(f"Fx inicial: {Fx}")
    if not math.isfinite(Fx):
        print("Error: Fx es infinito o NaN")
        sys.exit(-1)

    print("Antes del while...")

 
    #  CRITERIO DE PARO
    TOL_ESTADO_ESTACIONARIO = 1.0e-6  # tolerancia para u_t ≈ 0
    MAX_ITERACIONES = 50000            # limite

    while flag:
        P0 = P.copy()
        Fx0 = Fx

        print("  P0 copiado")

        dP = derivada(P, Phase0, lam)
        print("  Derivada calculada")
        P = P - tao * dP
        print("  P actualizado")
        Fx = funcional(P, Phase0, lam)
        print(f"  Fx calculado: {Fx}")
        difF = abs(Fx0 - Fx)

        # Calculo del cambio relativo (aproximacion a u_t)
        errp = math.sqrt(np.sum((P - P0) ** 2)) / math.sqrt(np.sum(P0 ** 2))
        print(f"  errp: {errp}")

        if iter_ % 100 == 0:
            print(f"iteracion : {iter_} Fx= {Fx} ||d_p||= {errp}")
            mostrar_imagen(win1, P)
            cv2.waitKey(1)


        #  CONDICION DE PARO: ESTADO ESTACIONARIO (u_t ≈ 0)
        if errp < TOL_ESTADO_ESTACIONARIO:
            print(f"\n*** ESTADO ESTACIONARIO ALCANZADO ***")
            print(f"iteracion : {iter_} Fx= {Fx} ||d_p||= {errp}")
            print(f"u_t ≈ 0, la solucion ha convergido")
            flag = False
        elif iter_ >= MAX_ITERACIONES:
            print(f"\n*** MAXIMO DE ITERACIONES ALCANZADO ***")
            print(f"iteracion : {iter_} Fx= {Fx} ||d_p||= {errp}")
            flag = False

        iter_ += 1
        print(f"  Fin iteracion {iter_}")

    print("SALI DEL WHILE")
    end = time.time()


    ms = (end - start) * 1000.0
    print(f"\nTiempo empleado  : {ms} mili-segundos")

    error = math.sqrt(np.sum((Po - P) ** 2)) / (
        math.sqrt(np.sum(Po ** 2)) + math.sqrt(np.sum(P ** 2))
    )
    print(f"\nNormalized error : = {error}\n")


    mostrar_imagen(win1, P)
    guardar_imagen("estimacionFinal.png", P)
    cv2.waitKey(0)


    diff = np.abs(P - Po)
    mostrar_error("Error", diff)
    guardar_error("Error.png", diff)

    print("Resultados guardados en archivos PNG")
    print("Presiona cualquier tecla para salir...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()