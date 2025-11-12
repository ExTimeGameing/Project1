#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

// Функция для создания бинарной маски границ сегментов
void createSegmentBoundaryMask(const Mat& segmented, Mat& boundaryMask) {
    if (segmented.empty() || segmented.channels() != 3) {
        boundaryMask = Mat::zeros(segmented.size(), CV_8UC1);
        return;
    }

    boundaryMask = Mat::zeros(segmented.size(), CV_8UC1); // Создаем бинарное изображение (0 или 255)

    // Проходим по всем пикселям, кроме границ изображения
    for (int y = 1; y < segmented.rows - 1; ++y) {
        for (int x = 1; x < segmented.cols - 1; ++x) {
            Vec3b currentPixel = segmented.at<Vec3b>(y, x);

            // Проверяем соседей (вверх, вниз, влево, вправо)
            bool isBoundary = false;
            if (segmented.at<Vec3b>(y - 1, x) != currentPixel || // вверх
                segmented.at<Vec3b>(y + 1, x) != currentPixel || // вниз
                segmented.at<Vec3b>(y, x - 1) != currentPixel || // влево
                segmented.at<Vec3b>(y, x + 1) != currentPixel) { // вправо
                isBoundary = true;
            }

            // Если пиксель находится на границе сегмента, устанавливаем точку в маске в 255 (белый)
            if (isBoundary) {
                boundaryMask.at<uchar>(y, x) = 255; // Белый цвет в бинарной маске
            }
        }
    }
}

// Функция для отрисовки границ сегментов с использованием findContours/drawContours
void drawSegmentBoundariesWithContours(Mat& image, const Mat& segmented, Scalar boundaryColor = Scalar(0, 0, 0)) {
    if (image.empty() || segmented.empty() || image.size() != segmented.size() || image.channels() != segmented.channels()) {
        return; // Проверка на корректность входных данных
    }

    Mat boundaryMask;
    createSegmentBoundaryMask(segmented, boundaryMask);

    // Находим контуры на бинарной маске
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(boundaryMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Рисуем найденные контуры на исходном изображении
    drawContours(image, contours, -1, boundaryColor, 1, LINE_8); // Толщина 1, зеленый цвет
}

// Функция для обработки видеофайла
void processVideo(const string& inputPath, const string& outputPathSegmentsWithBoundaries, const string& outputPathOriginalWithCanny, const string& timeLogPath) {
    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл: " << inputPath << endl;
        return;
    }

    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    Size frameSize(static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Видео: MeanShift + визуализация границ сегментов (черные)
    VideoWriter writerSegmentsWithBoundaries(outputPathSegmentsWithBoundaries, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    // Видео: Оригинал + визуализация контуров Canny (зеленые)
    VideoWriter writerOriginalWithCanny(outputPathOriginalWithCanny, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);

    if (!writerSegmentsWithBoundaries.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для MeanShift с границами: " << outputPathSegmentsWithBoundaries << endl;
        return;
    }
    if (!writerOriginalWithCanny.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для оригинала с контурами: " << outputPathOriginalWithCanny << endl;
        return;
    }

    ofstream timeLogFile(timeLogPath);
    if (!timeLogFile.is_open()) {
        cerr << "Ошибка: не удалось открыть файл для лога времени: " << timeLogPath << endl;
        return;
    }

    Mat frame, originalFrame, segmented, gray, edges, segmentsWithBoundaries, originalWithCanny;
    int frameCount = 0;

    vector<double> processingTimes;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Сохраняем оригинальный кадр
        originalFrame = frame.clone();

        auto start = chrono::high_resolution_clock::now();

        // Применить MeanShift к оригинальному кадру
        pyrMeanShiftFiltering(frame, segmented, 20, 30, 2);

        // Преобразовать ОРИГИНАЛЬНЫЙ кадр в оттенки серого и применить Canny
        cvtColor(originalFrame, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, 50, 150);

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0; // в мс
        processingTimes.push_back(duration);
        timeLogFile << duration << endl;

        // --- Видео 1: MeanShift + границы сегментов ---
        segmentsWithBoundaries = segmented.clone(); // Копируем сегментированное изображение
        // Нарисовать границы сегментов (черные линии) с использованием findContours/drawContours
        drawSegmentBoundariesWithContours(segmentsWithBoundaries, segmented, Scalar(0, 0, 0)); // Черный цвет
        writerSegmentsWithBoundaries.write(segmentsWithBoundaries);

        // --- Видео 2: Оригинал + контуры Canny ---
        originalWithCanny = originalFrame.clone(); // Копируем оригинальный кадр
        // Попиксельно наложить бинарные контуры Canny на оригинальный цветной кадр
        for (int y = 0; y < originalWithCanny.rows; ++y) {
            for (int x = 0; x < originalWithCanny.cols; ++x) {
                if (edges.at<uchar>(y, x) != 0) { // Если пиксель является частью контура
                    originalWithCanny.at<Vec3b>(y, x) = Vec3b(0, 255, 0); // Установить зеленый цвет
                }
            }
        }
        writerOriginalWithCanny.write(originalWithCanny);

        frameCount++;
        if (frameCount % 30 == 0) { // Прогресс каждые 30 кадров
            cout << "Обработано кадров: " << frameCount << " из " << totalFrames << endl;
        }
    }

    cap.release();
    writerSegmentsWithBoundaries.release();
    writerOriginalWithCanny.release();
    timeLogFile.close();

    cout << "Видео " << inputPath << " обработано." << endl;
    cout << "Выходные файлы:" << endl;
    cout << "  MeanShift + границы сегментов: " << outputPathSegmentsWithBoundaries << endl;
    cout << "  Оригинал + контуры Canny: " << outputPathOriginalWithCanny << endl;
    cout << "Лог времени: " << timeLogPath << endl;

    // Вывод статистики по времени
    if (!processingTimes.empty()) {
        double sum = 0, sumSquares = 0;
        double minTime = processingTimes[0], maxTime = processingTimes[0];
        for (double t : processingTimes) {
            sum += t;
            sumSquares += t * t;
            if (t < minTime) minTime = t;
            if (t > maxTime) maxTime = t;
        }
        double mean = sum / processingTimes.size();
        double variance = (sumSquares / processingTimes.size()) - (mean * mean);
        double stddev = sqrt(variance);

        cout << "--- Статистика по времени обработки (мс) для " << inputPath << " ---" << endl;
        cout << "Среднее: " << mean << endl;
        cout << "Минимум: " << minTime << endl;
        cout << "Максимум: " << maxTime << endl;
        cout << "Стандартное отклонение: " << stddev << endl;
    }
}

int main(int argc, char* argv[]) {
    // Обработка первого видеофайла
    processVideo("city.mp4",
        "city_meanshift_boundaries.avi",    // MeanShift + границы сегментов (черные)
        "city_original_with_canny.avi",     // Оригинал + контуры Canny (зеленые)
        "city_timing_log.txt");

    // Обработка второго видеофайла
    processVideo("winter.mp4",
        "winter_meanshift_boundaries.avi",  // MeanShift + границы сегментов (черные)
        "winter_original_with_canny.avi",   // Оригинал + контуры Canny (зеленые)
        "winter_timing_log.txt");

    cout << "Обработка завершена. Проверьте выходные файлы." << endl;

    return 0;
}