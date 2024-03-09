QT += gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        main.cpp

#INCLUDEPATH += D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\include \

#LIBS += D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_core455.dll \
#        D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_highgui455.dll \
#        D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_imgproc455.dll \
#        D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_calib3d455.dll \
#        D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_imgcodecs455.dll \
#        D:\OpenCV4.5.5\OpenCV-MinGW-Build-OpenCV-4.5.5-x64\x64\mingw\bin\libopencv_videoio455.dll

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../OpenCV4.5.5/opencv/build/x64/vc14/lib/ -lopencv_world455
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../OpenCV4.5.5/opencv/build/x64/vc14/lib/ -lopencv_world455d

INCLUDEPATH += $$PWD/../../../OpenCV4.5.5/opencv/build/x64/vc14
DEPENDPATH += $$PWD/../../../OpenCV4.5.5/opencv/build/x64/vc14


INCLUDEPATH += D:\OpenCV4.5.5\opencv\build\include \
               D:\OpenCV4.5.5\opencv\build\include\opencv \
               D:\OpenCV4.5.5\opencv\build\include\opencv2
