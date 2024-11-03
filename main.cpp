//g++ -I./include/opencv4 -L./lib main.cpp -o my_program -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
//export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    VideoCapture cap(0);

    if (!cap.isOpened())
	{
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    namedWindow("Control", WINDOW_AUTOSIZE);

    int iLastX = -1;
    int iLastY = -1;

    while (true) {
        Mat imgOriginal, imgGray, imgThresholded;

        bool bSuccess = cap.read(imgOriginal);

        if (!bSuccess) { // Si no se puede leer, romper el bucle
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        // Convertir la imagen a escala de grises
        cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);

        // Aplicar un umbral para detectar el objeto rectangular
        threshold(imgGray, imgThresholded, 100, 255, THRESH_BINARY); // Ajusta el valor del umbral según sea necesario

        // Encontrar contornos
        vector<vector<Point>> contours;
        findContours(imgThresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Dibujar el contorno del objeto más grande
        if (!contours.empty()) {
            size_t largestContourIndex = 0;
            double largestArea = 0;

            // Buscar el contorno más grande
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestContourIndex = i;
                }
            }

            // Calcular el rectángulo delimitador del contorno más grande
            Rect rect = boundingRect(contours[largestContourIndex]); // Cambiado a 'rect'
            rectangle(imgOriginal, rect.tl(), rect.br(), Scalar(0, 255, 0), 2); // Dibuja el rectángulo

            // Obtener las coordenadas del centro del objeto
            Point center = (rect.tl() + rect.br()) / 2;

            // Dibujar una línea si ya hay un último punto
            if (iLastX >= 0 && iLastY >= 0) {
                line(imgOriginal, Point(iLastX, iLastY), center, Scalar(0, 0, 255), 2);
            }

            // Actualizar las últimas coordenadas
            iLastX = center.x;
            iLastY = center.y;
        }

        imshow("Thresholded Image", imgThresholded); // Mostrar la imagen umbralizada
        imshow("Original", imgOriginal); // Mostrar la imagen original

        if (waitKey(30) == 27) { // Esperar la tecla 'esc' para salir
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }

    return 0;
}

