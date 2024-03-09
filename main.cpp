#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QDebug>
#include <QPainter>
#include <iostream>

using namespace  cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    /*
    cv::Mat InImg = cv::imread("C:/Users/Elery/Desktop/img100.jpg");
    cv::Mat ouImg;
    cv::resize(InImg, ouImg,Size(), 0.2, 0.2);
    cv::cvtColor(ouImg, ouImg, CV_BGR2RGB);
    QImage image(ouImg.data, ouImg.cols, ouImg.rows, QImage::Format_RGB888);
    if(image.allGray())
    {
        qDebug() << "all gray";
    }
    */


    /*
    QImage img("C:/Users/Elery/Desktop/img100.jpg");
    img = img.convertToFormat(QImage::Format_RGB888);
    int size = img.byteCount();
    int byteperline = img.bytesPerLine();
    qDebug() << "size:" << size << "byteperline: "<< byteperline;

    QImage img2 = img.copy();
    img2.save("img2.jpg");
    int depth = img2.depth();
    qDebug() << "depth: " << depth;
    img2.fill(168);
    img2.save("fill.jpg");

    int format = img2.format();
    qDebug() << "format: " << format;

    if(img2.hasAlphaChannel())
    {
        qDebug() << "hasAlphaChannel: ";
    }

    QImage mirrorImge = img.mirrored(true, false);
    mirrorImge.save("mirrorImg.jpg");

    QRgb rgb = mirrorImge.pixel(20, 50);
    QColor color = mirrorImge.pixelColor(20, 50);
    mirrorImge.save("setcolor.jpg");
    qDebug() << "rgb: " << rgb << "color: " << color;

    QImage bgr = mirrorImge.rgbSwapped();
    mirrorImge.save("swapImg.jpg");

    for(int i = 0; i < mirrorImge.width(); i++)
    {
        for(int j = 0; j < mirrorImge.height(); j++)
        {
            mirrorImge.setPixel(i, j,qRgb(20, 168, 50));
        }
    }
    mirrorImge.save("setpixel.jpg");

    for(int i = 0; i < 50; i++)
    {
        for(int j = 0; j < mirrorImge.height(); j++)
        {
            mirrorImge.setPixelColor(i, j, QColor(0, 255, 0));
        }
    }
    mirrorImge.save("setpixelcolor.jpg");

    QImage img3("mirrorImg.jpg");
    QTransform trans;
    trans.rotate(90);
    img3 = img3.transformed(trans);
    img3.save("transform.jpg");
*/


/*
    QImage img(320, 240, QImage::Format_RGB888);
    QPainter painter;
    painter.begin(&img);
    painter.fillRect(img.rect(), Qt::white);
    painter.drawLine(0, 0, 100, 100);
    painter.drawEllipse(QPoint(100,100), 50, 80);
    painter.end();
    Mat mat(img.height(), img.width(), CV_8UC3, (void*)img.constBits(), img.bytesPerLine());
    cv::cvtColor(mat, mat, CV_BGR2RGB);

    cv::imshow("001", mat);
*/

/*
    cv::Mat InImg = cv::imread("C:/Users/Elery/Desktop/img100.jpg");
    cv::Mat ouImg;
    cv::resize(InImg, ouImg,Size(), 0.2, 0.2);
    Mat outPut;
    cv::bilateralFilter(ouImg, outPut, 15, 200, 200);
    cv::imshow("bilateralFilter", outPut);

    Size kernelSize(5, 5);
    cv::blur(ouImg, outPut, kernelSize);
    cv::imshow("blur", outPut);

    int depth = -1;
    Size kernelsize(10, 10);
    Point anchorPoint(-1, -1);
    bool normalize = true;
    boxFilter(ouImg, outPut, depth, kernelsize, anchorPoint, normalize);
    cv::imshow("boxFilter", outPut);

    Mat output2;
    cv::GaussianBlur(ouImg, output2, Size(13, 13), 10, 10);
    cv::imshow("GaussianBlur", output2);

    cv::medianBlur(ouImg, output2, 13);
    cv::imshow("medianBlur", output2);

    cv::Mat kernel = (Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1,  0,-1,0);
    filter2D(ouImg, output2, ouImg.depth(),kernel);
    cv::imshow("filter2D", output2);

*/

    /*
    cv::Mat inImg = cv::imread("D:/QtCode/OpencvTest/OpencvTest/test.jpg");
    cv::imshow("raw", inImg);
    Mat outPut;
    Sobel(inImg, outPut, -1, 1, 1);
    cv::imshow("Sobel", outPut);

    Sobel(inImg, outPut, -1, 1, 1, 3, 5, 220);
    cv::imshow("Sobel2", outPut);

    Sobel(inImg, outPut, -1, 1, 1, 3, 5, 100);
    cv::imshow("Sobel3", outPut);

    Sobel(inImg, outPut, -1, 1, 1, 3, 5, 150);
    cv::imshow("Sobel4", outPut);

    Scharr(inImg, outPut, -1, 1, 0, 1, 100);
    cv::imshow("Scharr", outPut);

    Laplacian(inImg, outPut, -1, 3, 3);
    cv::imshow("Laplacian", outPut);

    Laplacian(inImg, outPut, -1, 3, 3, 15);
    cv::imshow("Laplacian2", outPut);
    */


//#kernel = np.ones((3,3), np.uint8)
    cv::Mat inImg = cv::imread("D:/QtCode/OpencvTest/OpencvTest/test.jpg");
    Mat outPut, outPut2;
//    cv::imshow("raw", inImg);
/*
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    cv::dilate(inImg, outPut, kernel);
    cv::imshow("dilate", outPut);

    cv::erode(outPut, outPut2, kernel);
    cv::imshow("erode", outPut2);

    cv::morphologyEx(inImg, outPut, MORPH_OPEN, kernel);
    cv::imshow("morphologyExMORPH_OPEN", outPut);

    cv::morphologyEx(inImg, outPut, MORPH_CLOSE, kernel);
    cv::imshow("morphologyExMORPH_CLOSE", outPut);

    cv::morphologyEx(inImg, outPut, MORPH_GRADIENT, kernel);
    cv::imshow("morphologyExMORPH_GRADIENT", outPut);

    cv::morphologyEx(inImg, outPut, MORPH_TOPHAT, kernel);
    cv::imshow("morphologyExMORPH_TOPHAT", outPut);
*/


//    resize(inImg, outPut, Size(), 0.5, 0.5, INTER_LANCZOS4);
//    cv::imshow("resize", outPut);

/*
    Point2f triangleA[3] = {Point2f(0, 0), Point2f(1,0), Point2f(0,1)};
    Point2f triangleB[3] = {Point2f(0, 0.5), Point2f(1,0.5), Point2f(0.5,1)};
    Mat affineMat = getAffineTransform(triangleA, triangleB);
    warpAffine(inImg, outPut, affineMat, inImg.size(), INTER_CUBIC, BORDER_WRAP);
    cv::imshow("warpAffine", outPut);


    std::vector<Point2f> cornersA(4);
    std::vector<Point2f> cornersB(4);
    cornersA[0] = Point2f(0,0);
    cornersA[1] = Point2f(inImg.cols, 0);
    cornersA[2] = Point2f(inImg.cols, inImg.rows);
    cornersA[3] = Point2f(0, inImg.rows);

    cornersB[0] = Point2f(inImg.cols * 0.25, 0);
    cornersB[1] = Point2f(inImg.cols * 0.9, 0);
    cornersB[2] = Point2f(inImg.cols, inImg.rows);
    cornersB[3] = Point2f(0, inImg.rows * 0.80);

    Mat homo = getPerspectiveTransform(cornersA, cornersB);
    warpPerspective(inImg, outPut, homo, inImg.size(), INTER_LANCZOS4, BORDER_CONSTANT,Scalar(5, 5, 5));
    cv::imshow("warpPerspective", outPut);


    Mat mapX, mapY;
    mapX.create(inImg.size(), CV_32FC(1));
    mapY.create(inImg.size(), CV_32FC(1));
    for(int i = 0; i < inImg.rows; i++)
        for(int j = 0; j < inImg.cols; j++)
        {
            mapX.at<float>(i, j) = inImg.cols - j;
            mapX.at<float>(i, j) = inImg.rows - i;
        }
    remap(inImg, outPut, mapX, mapY, INTER_LANCZOS4, BORDER_REPLICATE);
    cv::imshow("remap", outPut);
*/

/*
    cvtColor(inImg, outPut, COLOR_RGB2BGR);
    imshow("COLOR_BGR2RGB", outPut);

    cvtColor(inImg, outPut2, COLOR_RGBA2GRAY);
    imshow("COLOR_RGBA2GRAY", outPut2);
    QImage img(outPut2.data, outPut2.cols, outPut2.rows, QImage::Format_Grayscale8);
    img.save("COLOR_RGBA2GRAY.jpg");

    Mat source = cv::imread("COLOR_RGBA2GRAY.jpg", cv::IMREAD_GRAYSCALE);
    Mat colorMap;
    applyColorMap(inImg, colorMap, COLORMAP_HOT);
    imshow("COLORMAP_JET", colorMap);

    applyColorMap(inImg, colorMap, COLORMAP_HSV);
    imshow("COLORMAP_HSV", colorMap);

    applyColorMap(inImg, colorMap, COLORMAP_JET);
    imshow("COLORMAP_JET", colorMap);

    applyColorMap(inImg, colorMap, COLORMAP_BONE);
    imshow("COLORMAP_BONE", colorMap);

    applyColorMap(inImg, colorMap, COLORMAP_COOL);
    imshow("COLORMAP_COOL", colorMap);

    applyColorMap(inImg, colorMap, COLORMAP_AUTUMN);
    imshow("COLORMAP_AUTUMN", colorMap);
*/

    /*
    Mat grayscale;
    cvtColor(inImg, grayscale, CV_BGR2GRAY);
    threshold(inImg, grayscale, 38, 255, THRESH_BINARY_INV);
    imshow("THRESH_BINARY_INV", grayscale);

    threshold(inImg, grayscale, 38, 255, THRESH_BINARY);
    imshow("THRESH_BINARY", grayscale);


    cvtColor(inImg, grayscale, CV_BGR2GRAY);
    adaptiveThreshold(grayscale, outPut, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 0);
    imshow("adaptiveThreshold5", outPut);
    adaptiveThreshold(grayscale, outPut, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 9, 0);
    cvtColor(outPut, outPut, CV_GRAY2BGR);
    imshow("adaptiveThreshold7", outPut);
*/

//    cv::line(inImg, Point(0,0),Point(inImg.cols-1, inImg.rows-1),Scalar(0,0,255),3, LINE_AA);
//    cv::line(inImg, Point(inImg.cols-1, 0), Point(0,inImg.rows-1),Scalar(0,0,255),3, LINE_AA);
//    cv::imshow("line", inImg);

/*
    Mat temp = cv::imread("D:/QtCode/OpencvTest/OpencvTest/template.jpg");

    double angle = 45;
    // get the center coordinates of the image to create the 2D rotation matrix
    Point2f center((inImg.cols - 1) / 2.0, (inImg.rows - 1) / 2.0);
    // using getRotationMatrix2D() to get the rotation matrix
    Mat rotation_matix = getRotationMatrix2D(center, angle, 1.0);

    // we will save the resulting image in rotated_image matrix
    Mat rotated_image;
    // rotate the image using warpAffine
    warpAffine(inImg, rotated_image, rotation_matix, inImg.size());
    imshow("Rotated image", rotated_image);
    cvtColor(rotated_image, rotated_image, CV_BGR2RGB);
    QImage img(rotated_image.data, rotated_image.cols, rotated_image.rows, QImage::Format_RGB888);
    img.save("rotate45.jpg");

    matchTemplate(rotated_image, temp, outPut, TM_CCORR_NORMED);
    imshow("matchTemplate", outPut);
    double minVal, maxVal;
    Point minLoc,maxLoc;
    minMaxLoc(outPut, &minVal, &maxVal, &minLoc, &maxLoc);
    rectangle(rotated_image, Rect(maxLoc.x, maxLoc.y, temp.cols, temp.rows), Scalar(0,0,255),2);
    imshow("minMaxLoc", rotated_image);
*/

    /*
   cv::TickMeter  meter;
   meter.start();
    Mat M(2, 2, CV_8UC3, Scalar(0, 0, 101));
    cout << "M = " << endl << " " << M << endl << endl;

    Mat test;
    test.create(4, 4, CV_8U);
    cout << "test = " << endl << " " << test << endl << endl;

    Mat roi(test, Rect( 0, 0, 3, 3));
    cout << "roi = " << endl << " " << roi << endl << endl;
    meter.stop();
    meter.getTimeMicro();
    meter.getTimeMilli();
    meter.getTimeSec();
    meter.getTimeTicks();
    cout << meter.getTimeMicro() <<"微秒 ;"<< meter.getTimeMilli() << "毫秒 ;" <<
            meter.getTimeSec() << "秒 ;"<< meter.getTimeTicks();
    */

    /*
    vector<Mat> planes;
    split(inImg, planes);

    cvtColor(inImg, outPut, CV_BGR2GRAY);
    int bins = 256;
    int channels[] = {0};
    int histsize[] = {bins};
    float rangeGray[] = {0, 255};
    const float* ranges[] = {rangeGray};
    Mat histogram[3];
    double maxVal[3] = {0};
    for(int i=0; i<3; i++)
    {
        calcHist(&outPut, 1, channels, Mat(), histogram[i], 1, histsize, ranges, true, false);
        minMaxLoc(histogram[i], Q_NULLPTR, &maxVal[i], Q_NULLPTR, Q_NULLPTR);
    }

    outPut2.create(640, 640, CV_8UC3);
    outPut2 = Scalar::all(128);
    Point p1[3], p2[3];
    for(int i = 0; i < bins; i++)
    {
        for(int j=0; j<3; j++)
        {
            float value = histogram[j].at<float>(i, 0);
            value =  maxVal[j] - value;
            value = value / maxVal[j] * outPut2.rows;
    //        p1.y = value;
    //        p2.x = float(i+1) * float(outPut2.cols) / float(bins);
            line(outPut2, p1[j], Point(p1[j].x, value),Scalar(j==0 ? 255:0,j==1 ? 255:0,j==2 ? 255:0),2);
            p1[j].y = p2[j].y = value;
            p2[j].x = float(i + 1) * float(outPut2.cols) / float(bins);
    //        rectangle(outPut2, p1, p2, Scalar::all(0), FILLED);
            line(outPut2, p1[j], p2[j], Scalar(j==0 ? 255:0,j==1 ? 255:0,j==2 ? 255:0),2);
            p1[j].x = p2[j].x;
        }
    }
    cv::imshow("00", outPut2);
    */

    /*
    int bins = 256;
    Mat histogram[3];
    int channels[] = {0};
    int histSize[] = {bins};; // number of bins

    float range[] = {0,255}; // range of colors
    const float* ranges[] = { range };

    Mat histograms[3];

    vector<Mat> planes;
    split(inImg, planes);

    double maxVal[3] = {0,0,0};

    for(int i = 0; i < 3; i++)
    {
        calcHist(&planes[i],
                 1, // number of images
                 channels,
                 Mat(), // no masks, an empty Mat
                 histograms[i],
                 1, // dimensionality
                 histSize,
                 ranges);

        minMaxLoc(histograms[i],
                  Q_NULLPTR, // don't need min
                  &maxVal[i],
                  Q_NULLPTR, // don't need index min
                  Q_NULLPTR // don't need index max
                  );
    }

    outPut.create(640, // any image width
                       640, // any image height
                       CV_8UC(3));

    outPut = Scalar::all(0); // empty black image

    Point p1[3], p2[3];
    for(int i=0; i< bins; i++)
    {
        for(int j=0; j<3; j++)
        {
            float value = histograms[j].at<float>(i,0);
            value = maxVal[j] - value; // invert
            value = value / maxVal[j] * outPut.rows;
            line(outPut,
                 p1[j],
                 Point(p1[j].x,value),
                 Scalar(j==0 ? 255:0,
                        j==1 ? 255:0,
                        j==2 ? 255:0),
                        2);
            p1[j].y = p2[j].y = value;
            p2[j].x = float(i+1) * float(outPut.cols) / float(bins);
            line(outPut,
                 p1[j], p2[j],
                 Scalar(j==0 ? 255:0,
                        j==1 ? 255:0,
                        j==2 ? 255:0),
                 2);
            p1[j].x = p2[j].x;
        }
    }
    cv::imshow("00", outPut);
    */

    /*
    Mat image(25, 180, CV_8UC3);
    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            image.at<Vec3b>(i, j)[0] = j;
            image.at<Vec3b>(i, j)[1] = 255;
            image.at<Vec3b>(i, j)[2] = 255;
        }
    }

    cvtColor(image, image, CV_HSV2BGR);
    imshow("002", image);
*/

    /*
    cvtColor(inImg, outPut, CV_BGR2GRAY);
    imshow("CV_BGR2GRAY", outPut);
    cv::equalizeHist(outPut, outPut2);
    imshow("equalizeHist", outPut2);
    waitKey(0);
    */


    //Mat ROI
    // read the image
    cv::Mat image=  cv::imread("D:\\QtCode\\OpencvTest\\OpencvTest\\dota_pa.jpg");

    // read the logo
    cv::Mat logo=  cv::imread("D:\\QtCode\\OpencvTest\\OpencvTest\\dota_logo.jpg");

    cv::Mat imageROI(image, cv::Rect(0,
                                     0,
                                     logo.cols, logo.rows));
    logo.copyTo(imageROI);

//    cv::imshow("001", image);
//    cv::imshow("002", imageROI);

    cv::resize(image, image, cv::Size(), 2 , 2, cv::INTER_NEAREST);
    cv::imshow("003", image);

    return a.exec();
}
