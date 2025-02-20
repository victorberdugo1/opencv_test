//g++ -L./lib main.cpp -o my_program -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
//export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
//g++ main.cpp -o my_program $(pkg-config --cflags --libs opencv4)


#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

/// Función para crear el pipeline GStreamer
std::string gstreamer_pipeline(int capture_width, int capture_height, int framerate, int display_width, int display_height) {
    return
            "libcamerasrc ! video/x-raw, "
            "width=(int)" + std::to_string(capture_width) + ","
            "height=(int)" + std::to_string(capture_height) + ","
	    "focus-mode=auto, "
            "framerate=(fraction)" + std::to_string(framerate) + "/1 ! "
            "videoconvert ! videoscale ! "
            "video/x-raw, "
            "width=(int)" + std::to_string(display_width) + ","
            "height=(int)" + std::to_string(display_height) + " ! appsink";
}

int main(int argc, char** argv)
{
    // Parámetros del pipeline
    int capture_width = 640; // Ancho de captura
    int capture_height = 480; // Altura de captura
    int framerate = 15; // Frecuencia de fotogramas (fps)
    int display_width = 640; // Ancho de visualización
    int display_height = 480; // Altura de visualización

    // Crea el pipeline
    std::string pipeline = gstreamer_pipeline(capture_width, capture_height, framerate, display_width, display_height);
    std::cout << "Usando pipeline: \n\t" << pipeline << "\n\n";

    // Abre la cámara usando OpenCV y el pipeline GStreamer
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Error: No se pudo abrir la cámara." << std::endl;
        return -1;
    }

    // Crea una ventana para mostrar el video
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    int iLastX = -1;
    int iLastY = -1;

    std::cout << "Presiona ESC para salir" << "\n";

    // Bucle de captura de frames
    while (true) {
        cv::Mat imgOriginal, imgGray, imgThresholded;

        // Captura un frame
        bool bSuccess = cap.read(imgOriginal);
        if (!bSuccess) {
            std::cout << "Error: No se pudo leer un frame" << std::endl;
            break;
        }

        // Convierte el frame a escala de grises
        cv::cvtColor(imgOriginal, imgGray, cv::COLOR_BGR2GRAY);

        // Aplica un umbral para obtener una imagen binaria
        cv::threshold(imgGray, imgThresholded, 100, 255, cv::THRESH_BINARY);

        // Encuentra los contornos en la imagen binaria
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(imgThresholded, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            size_t largestContourIndex = 0;
            double largestArea = 0;

            // Encuentra el contorno más grande
            for (size_t i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestContourIndex = i;
                }
            }

            // Dibuja un rectángulo alrededor del contorno más grande
            cv::Rect rect = cv::boundingRect(contours[largestContourIndex]);
            cv::rectangle(imgOriginal, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 2);

            // Dibuja una línea desde el centro del rectángulo a la última posición
            cv::Point center = (rect.tl() + rect.br()) / 2;
            if (iLastX >= 0 && iLastY >= 0) {
                cv::line(imgOriginal, cv::Point(iLastX, iLastY), center, cv::Scalar(0, 0, 255), 2);
            }
            iLastX = center.x;
            iLastY = center.y;
        }

        // Muestra las imágenes
        cv::imshow("Thresholded Image", imgThresholded);
        cv::imshow("Original", imgOriginal);

        // Sale si se presiona la tecla ESC (código 27)
        char esc = cv::waitKey(5);
        if (esc == 27) break;
    }

    // Libera la cámara y cierra la ventana
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

