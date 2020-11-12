# Basic
import numpy as np
import math
import sys
import time
import os
import datetime
import cv2
import copy

# PySide2
from PySide2 import QtCore
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel
from PySide2.QtCore import QFile, QTimer, SIGNAL
from PySide2.QtGui import QPixmap, QImage
from widget import Ui_Widget

# PySide2 and matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# ROS
import rospy
from sensor_msgs.msg import Imu
from ads_b_read.msg import Traffic_Report_Array

#######################################
# Define Parameter
WGS84_a_m = 6378137.0
WGS84_e = 0.08181919

d2r = math.pi / 180
r2d = 180 / math.pi

Ownship_ICAO = 7463479
Intruder_ICAO = 7463475

# Hanseo Univ
std_lat = 36.593133
std_lon = 126.295440
std_alt = 1000

scale_100m = 0.4444444444444
scale_200m = 0.2222222222222
scale_400m = 0.1111111111111
scale_800m = 0.0555555555555
scale_1600m = 0.0277777777777

map_row = 540
map_col = 800

icon_row_col = 26

#######################################
std_sinLat = math.sin(std_lat * d2r)
std_sinLon = math.sin(std_lon * d2r)
std_cosLat = math.cos(std_lat * d2r)
std_cosLon = math.cos(std_lon * d2r)

N = WGS84_a_m / math.sqrt(1 - WGS84_e * WGS84_e * std_sinLat * std_sinLat)
ref_ECEF_x = (N + std_alt) * std_cosLat * std_cosLon
ref_ECEF_y = (N + std_alt) * std_cosLat * std_sinLon
ref_ECEF_z = (N * (1 - WGS84_e * WGS84_e) + std_alt) * std_sinLat

#######################################
# help function
def read_img(img_path, row, col):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(col, row), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_add_function(background, add_image, roi_row, roi_col):
    height, width, channel = add_image.shape
    check_image_size = int(height/2)

    roi = background[roi_row - check_image_size:roi_row + check_image_size, roi_col - check_image_size : roi_col + check_image_size]

    img2gray = cv2.cvtColor(add_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    img2_fg = cv2.bitwise_and(add_image, add_image, mask = mask)

    dst = cv2.add(img1_bg,img2_fg)
    ret_image = copy.copy(background)
    ret_image[roi_row - check_image_size:roi_row + check_image_size, roi_col - check_image_size : roi_col + check_image_size] = dst
    return ret_image

class ros_class():
    def __init__(self):
        self.ROS_TIME = 0
        self.ads_callback_time = []
        self.distance_list = []
        self.plot_cnt = 10
        self.plot_time_table = [-i for i in range(self.plot_cnt)]
        self.temp = self.plot_time_table.reverse()

        ####################################
        # Pos information (LLA)
        self.Ownship_lat = 0
        self.Ownship_lon = 0
        self.Ownship_alt = 0

        self.Intruder_lat = 0
        self.Intruder_lon = 0
        self.Intruder_alt = 0

        ####################################
        # ECEF Pos
        self.Ownship_ECEF_x = 0
        self.Ownship_ECEF_y = 0
        self.Ownship_ECEF_z = 0

        self.Intruder_ECEF_x = 0
        self.Intruder_ECEF_y = 0
        self.Intruder_ECEF_z = 0

        ####################################
        # Local Pos (Origin position is Hansae Univ)
        self.Ownship_local_x = 0
        self.Ownship_local_y = 0
        self.Ownship_local_z = 0

        self.Intruder_local_x = 0
        self.Intruder_local_y = 0
        self.Intruder_local_z = 0

        self.distance = 0

        ####################################
        # Heading / Hor / Ver Velocity
        self.Ownship_heading = 0
        self.Intruder_heading = 0

        self.Ownship_hor_vel = 0
        self.Ownship_ver_vel = 0

        self.Intruder_hor_vel = 0
        self.Intruder_ver_vel = 0

    def lla_to_ECEF(self, lat, lon, alt):
        sinLat = math.sin(lat * d2r)
        sinLon = math.sin(lon * d2r)
        cosLat = math.cos(lat * d2r)
        cosLon = math.cos(lon * d2r)

        N = WGS84_a_m / math.sqrt(1 - WGS84_e * WGS84_e * sinLat * sinLat)
        ECEF_x = (N + alt) * cosLat * cosLon
        ECEF_y = (N + alt) * cosLat * sinLon
        ECEF_z = (N * (1 - WGS84_e * WGS84_e) + alt) * sinLat
        return [ECEF_x, ECEF_y, ECEF_z]

    def ECEF_to_Local(self, x, y, z):
        ROT_MAT = [ [-std_sinLat * std_cosLon,       -std_sinLat * std_sinLon,        std_cosLat],
                    [             -std_sinLon,                     std_cosLon,               0.0],
                    [-std_cosLat * std_cosLon,       -std_cosLat * std_sinLon,       -std_sinLat]]

        diff_ECEF = [x - ref_ECEF_x, y - ref_ECEF_y, z - ref_ECEF_z]

        local_pos = np.matmul(np.array(ROT_MAT), np.array(diff_ECEF))
        return local_pos

    def ADS_callback(self, data):
        if (len(self.ads_callback_time) < self.plot_cnt):
            self.ads_callback_time.append(self.ROS_TIME)
        else:
            self.ads_callback_time.pop(0)
            self.ads_callback_time.append(self.ROS_TIME)

        Traffic_Reports_list = data.Traffic_Reports

        for i in range(len(Traffic_Reports_list)):
            if (Traffic_Reports_list[i].ICAO_address == Ownship_ICAO):
                self.Ownship_lat = Traffic_Reports_list[i].lat / 10000000.0
                self.Ownship_lon = Traffic_Reports_list[i].lon / 10000000.0
                self.Ownship_alt = Traffic_Reports_list[i].altitude / 1000.0
                self.Ownship_heading = Traffic_Reports_list[i].heading / 100.0
                self.Ownship_hor_vel = Traffic_Reports_list[i].hor_velocity / 100.0
                self.Ownship_ver_vel = Traffic_Reports_list[i].ver_velocity / 100.0
                self.Ownship_ECEF_x, self.Ownship_ECEF_y, self.Ownship_ECEF_z = self.lla_to_ECEF(self.Ownship_lat, self.Ownship_lon, self.Ownship_alt)
                self.Ownship_local_x, self.Ownship_local_y, self.Ownship_local_z = self.ECEF_to_Local(self.Ownship_ECEF_x, self.Ownship_ECEF_y, self.Ownship_ECEF_z)
                
                self.Ownship_local_x = round(self.Ownship_local_x, 3)
                self.Ownship_local_y = round(self.Ownship_local_y, 3)
                self.Ownship_local_z = round(self.Ownship_local_z, 3)

            if (Traffic_Reports_list[i].ICAO_address == Intruder_ICAO):
                self.Intruder_lat = Traffic_Reports_list[i].lat / 10000000.0
                self.Intruder_lon = Traffic_Reports_list[i].lon / 10000000.0
                self.Intruder_alt = Traffic_Reports_list[i].altitude / 1000.0
                self.Intruder_heading = Traffic_Reports_list[i].heading / 100.0
                self.Intruder_hor_vel = Traffic_Reports_list[i].hor_velocity / 100.0
                self.Intruder_ver_vel = Traffic_Reports_list[i].ver_velocity / 100.0
                self.Intruder_ECEF_x, self.Intruder_ECEF_y, self.Intruder_ECEF_z = self.lla_to_ECEF(self.Intruder_lat, self.Intruder_lon, self.Intruder_alt)
                self.Intruder_local_x, self.Intruder_local_y, self.Intruder_local_z = self.ECEF_to_Local(self.Intruder_ECEF_x, self.Intruder_ECEF_y, self.Intruder_ECEF_z)
        
                self.Intruder_local_x = round(self.Intruder_local_x, 3)
                self.Intruder_local_y = round(self.Intruder_local_y, 3)
                self.Intruder_local_z = round(self.Intruder_local_z, 3)

        self.distance = math.sqrt(   (self.Ownship_local_x - self.Intruder_local_x) ** 2 + \
                                (self.Ownship_local_y - self.Intruder_local_y) ** 2 + \
                                (self.Ownship_local_z - self.Intruder_local_z) ** 2)

        self.distance = round(self.distance, 2)

        if (len(self.distance_list) < self.plot_cnt):
            self.distance_list.append(self.distance)
        else:
            self.distance_list.pop(0)
            self.distance_list.append(self.distance)

    def timer_callback(self, data):
        self.ROS_TIME = data.header.stamp.secs + (data.header.stamp.nsecs) * 0.000000001
        self.ROS_TIME = math.floor(self.ROS_TIME * 10) / 10

ros = ros_class()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        ####################################
        # Figure
        self.distance_fig = Figure()
        self.distance_canvas = FigureCanvas(self.distance_fig)
        self.ui.DistanceLayout.addWidget(self.distance_canvas)

        self.visual_fig = Figure()
        self.visual_canvas = FigureCanvas(self.visual_fig)
        self.ui.visualization_layout.addWidget(self.visual_canvas)

        self.visual_cnt = 0
        
        ####################################
        # Button Interactive
        self.ui.map_change_Down.clicked.connect(self.map_change_down_clicked)
        self.ui.map_change.clicked.connect(self.mpa_change_clicked)

        ####################################
        # Img Path
        self.map_100_path = "img/icon/map_100m.png"
        self.map_200_path = "img/icon/map_200m.png"
        self.map_400_path = "img/icon/map_400m.png"
        self.map_800_path = "img/icon/map_800m.png"
        self.map_1600_path = "img/icon/map_1600m.png"
        self.Scale_factor = scale_100m

        self.intruder_img_path = "img/icon/Intruder Symbol.png"
        self.ownship_img_path = "img/icon/Ownship Symbol.png"

        ####################################
        # Map Parameter
        self.map_size = 100
        self.map_img = read_img(self.map_100_path, map_row, map_col)
        self.Scale_factor = scale_100m

        ####################################
        # Check aircraft is in MAP
        self.check_Ownship_in_map = 0
        self.check_Intruder_in_map = 0

        ####################################
        # Update GUI
        self.timer = QTimer()
        self.connect(self.timer, SIGNAL("timeout()"), self.visual_UI)
        self.timer.start(33)
        self.prev_time = 0

        ####################################
        # Debug Part
        self.check_Hz = 0
        self.plot_visual = 10

    def map_change_down_clicked(self):
        if (self.map_size == 200):
            self.map_size = 100
            self.Scale_factor = scale_100m
        elif (self.map_size == 400):
            self.map_size = 200
            self.Scale_factor = scale_200m
        elif (self.map_size == 800):
            self.map_size = 400
            self.Scale_factor = scale_400m
        elif (self.map_size == 1600):
            self.map_size = 800
            self.Scale_factor = scale_800m

        self.visual_map()
        
    def mpa_change_clicked(self):
        if (self.map_size == 100):
            self.map_size = 200
            self.Scale_factor = scale_200m
        elif (self.map_size == 200):
            self.map_size = 400
            self.Scale_factor = scale_400m
        elif (self.map_size == 400):
            self.map_size = 800
            self.Scale_factor = scale_800m
        elif (self.map_size == 800):
            self.map_size = 1600
            self.Scale_factor = scale_1600m

        self.visual_map()

    def visual_UI(self):
        if (self.check_Hz == 1):
            now = time.time()
            print("Hz : " + str(1 / (now - self.prev_time)))
            self.prev_time = now

        self.visual_img()
        self.visual_map()
        self.visual_txt()
        self.visual_Plot()
        if (self.visual_cnt % self.plot_visual == 0):
            self.visual_visual()
        self.visual_cnt += 1

    def visual_visual(self):
        self.visual_ax = self.visual_fig.add_subplot(111, projection='3d')
        self.visual_ax.cla()

        self.visual_ax.scatter([ros.Ownship_lat], [ros.Ownship_lon], [ros.Ownship_alt], color = 'b')
        self.visual_ax.scatter([ros.Intruder_lat], [ros.Intruder_lon], [ros.Intruder_alt], color = 'r')
        
        self.visual_ax.set_xlim(36.5, 36.68)
        self.visual_ax.set_ylim(126.2, 126.4)
        self.visual_ax.set_zlim(0, 2000)
    
        self.visual_ax.grid()
        self.visual_canvas.draw()


    def visual_Plot(self):
        self.distance_ax = self.distance_fig.add_subplot(111)
        self.distance_ax.cla()
        if (len(ros.distance_list) < ros.plot_cnt):
            temp_distance_list = [0 for i in range(ros.plot_cnt - len(ros.distance_list))] + ros.distance_list
        else:
            temp_distance_list = ros.distance_list
        self.distance_ax.plot(ros.plot_time_table, temp_distance_list, 'r')
        self.distance_ax.set_ylim(0, 5000)
        self.distance_ax.axhspan(0, 2000, facecolor='0.5', alpha = 0.5)
        self.distance_ax.grid()
        self.distance_canvas.draw()

    def visual_map(self):
        map_path_list = [self.map_100_path, self.map_200_path, self.map_400_path, self.map_800_path, self.map_1600_path]
        self.map_img = read_img(map_path_list[int(math.log(self.map_size/100, 2))], map_row, map_col)

        ownship_img = read_img(self.ownship_img_path, icon_row_col, icon_row_col)
        intruder_img = read_img(self.intruder_img_path, icon_row_col, icon_row_col)

        height, width, channel = ownship_img.shape
        matrix = cv2.getRotationMatrix2D((width/2, height/2), -ros.Ownship_heading, 1)
        rot_ownship_img = cv2.warpAffine(ownship_img, matrix, (width, height))

        matrix = cv2.getRotationMatrix2D((width/2, height/2), -ros.Intruder_heading, 1)
        rot_intruder_img = cv2.warpAffine(intruder_img, matrix, (width, height))

        ownship_row_in_map = map_row / 2 - int(ros.Ownship_local_x * self.Scale_factor)
        ownship_col_in_map = map_col / 2 + int(ros.Ownship_local_y * self.Scale_factor)

        Intruder_row_in_map = map_row / 2 - int(ros.Intruder_local_x * self.Scale_factor)
        Intruder_col_in_map = map_col / 2 + int(ros.Intruder_local_y * self.Scale_factor)

        height, width, channel = rot_intruder_img.shape
        check_image_size = int(height/2)
        if (Intruder_row_in_map < map_row - check_image_size and Intruder_row_in_map > check_image_size and Intruder_col_in_map < map_col - check_image_size and Intruder_col_in_map > check_image_size):
            self.overwrite_map_img = image_add_function(self.map_img, rot_intruder_img, Intruder_row_in_map, Intruder_col_in_map)
            self.check_Intruder_in_map = 1
        else:
            self.check_Intruder_in_map = 0

        if (ownship_row_in_map < map_row - check_image_size and ownship_row_in_map > check_image_size and ownship_col_in_map < map_col - check_image_size and ownship_col_in_map > check_image_size):
            if (self.check_Intruder_in_map):
                self.overwrite_map_img = image_add_function(self.overwrite_map_img, rot_ownship_img, ownship_row_in_map, ownship_col_in_map)
            else:
                self.overwrite_map_img = image_add_function(self.map_img, rot_ownship_img, ownship_row_in_map, ownship_col_in_map)
            self.check_Ownship_in_map = 1
        else:
            self.check_Ownship_in_map = 0

        # Add indicator Text
        # map, text, location, font, fontsclae, color, thickness
        if (self.check_Intruder_in_map):
            cv2.putText(self.overwrite_map_img, "Intruder", (Intruder_col_in_map - 30, Intruder_row_in_map - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if (self.check_Ownship_in_map):
            cv2.putText(self.overwrite_map_img, "Ownship", (ownship_col_in_map - 30, ownship_row_in_map - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if (self.check_Intruder_in_map == 0 and self.check_Ownship_in_map == 0):
            height, width, channel = self.map_img.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.map_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        else:
            height, width, channel = self.overwrite_map_img.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.overwrite_map_img.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap(qImg)
        pixmap = pixmap.scaled(map_col, map_row, QtCore.Qt.KeepAspectRatio) 
        self.ui.MAP.setPixmap(pixmap)

    def visual_txt(self):
        self.ui.IN_Ownship_lat.setText(str(ros.Ownship_lat))
        self.ui.IN_Ownship_lon.setText(str(ros.Ownship_lon))
        self.ui.IN_Ownship_Alt.setText(str(ros.Ownship_alt))
        self.ui.IN_Ownship_Head.setText(str(ros.Ownship_heading))
        self.ui.IN_Ownship_Hor_vel.setText(str(ros.Ownship_hor_vel))
        self.ui.IN_Ownship_Ver_vel.setText(str(ros.Ownship_ver_vel))
        self.ui.IN_Ownship_x.setText(str(ros.Ownship_local_x))
        self.ui.IN_Ownship_y.setText(str(ros.Ownship_local_y))

        self.ui.IN_Intruder_lat.setText(str(ros.Intruder_lat))
        self.ui.IN_Intruder_lon.setText(str(ros.Intruder_lon))
        self.ui.IN_Intruder_Alt.setText(str(ros.Intruder_alt))
        self.ui.IN_Intruder_Head.setText(str(ros.Intruder_heading))
        self.ui.IN_Intruder_Hor_vel.setText(str(ros.Intruder_hor_vel))
        self.ui.IN_Intruder_Ver_vel.setText(str(ros.Intruder_ver_vel))
        self.ui.IN_Intruder_x.setText(str(ros.Intruder_local_x))
        self.ui.IN_Intruder_y.setText(str(ros.Intruder_local_y))

        self.ui.IN_distance.setText(str(ros.distance))

    def visual_img(self):
        self.ui.IN_ROS_TIME.setText(str(ros.ROS_TIME))
        
        year = datetime.datetime.fromtimestamp(ros.ROS_TIME).year
        month = datetime.datetime.fromtimestamp(ros.ROS_TIME).month
        if (len(str(month)) == 1):
            month = "0" + str(month)
        day = datetime.datetime.fromtimestamp(ros.ROS_TIME).day
        if (len(str(day)) == 1):
            day = "0" + str(day)
        hour = datetime.datetime.fromtimestamp(ros.ROS_TIME).hour
        if (len(str(hour)) == 1):
            hour = "0" + str(hour)
        minute = datetime.datetime.fromtimestamp(ros.ROS_TIME).minute
        if (len(str(minute)) == 1):
            minute = "0" + str(minute)
        second = datetime.datetime.fromtimestamp(ros.ROS_TIME).second
        if (len(str(second)) == 1):
            second = "0" + str(second)

        ROS_datetime = str(year) + '-' + str(month) + "-" + str(day) + " " + str(hour) + ":" + str(minute) + ":" + str(second)
        self.ui.CurrentTime.setText(str(ROS_datetime))

        for i in range(10):
            img_path1 = "/home/usrg-asus/Civil/bag_file/200926_06/video/frame0/" + str(ros.ROS_TIME) + str(i) + ".jpg"
            if (os.path.exists(img_path1)):
                pixmap = QPixmap(QImage(img_path1))
                pixmap = pixmap.scaled(480, 270, QtCore.Qt.KeepAspectRatio) 
                self.ui.Img_view1.setPixmap(pixmap)

            img_path2 = "/home/usrg-asus/Civil/bag_file/200926_06/video/frame1/" + str(ros.ROS_TIME) + str(i) + ".jpg"
            if (os.path.exists(img_path2)):
                pixmap = QPixmap(QImage(img_path2))
                pixmap = pixmap.scaled(480, 270, QtCore.Qt.KeepAspectRatio) 
                self.ui.Img_view3.setPixmap(pixmap)

            img_path3 = "/home/usrg-asus/Civil/bag_file/200926_06/video/frame2/" + str(ros.ROS_TIME) + str(i) + ".jpg"
            if (os.path.exists(img_path3)):
                pixmap = QPixmap(QImage(img_path3))
                pixmap = pixmap.scaled(480, 270, QtCore.Qt.KeepAspectRatio) 
                self.ui.Img_view2.setPixmap(pixmap)

app = QApplication(sys.argv)
window = MainWindow()
# window.setCentralWidget(window.canvas)

window.ui.IN_Ownship_ICAO.setText(str(Ownship_ICAO))
window.ui.IN_Intruder_ICAO.setText(str(Intruder_ICAO))

if __name__ == "__main__":
    ros = ros_class()
    rospy.init_node('widget', anonymous = True)

    rospy.Subscriber("/ADS_B/Traffic_Report_Array", Traffic_Report_Array, ros.ADS_callback, queue_size = 1)
    rospy.Subscriber("mavros/imu/data", Imu, ros.timer_callback, queue_size = 1)

    window.show()
    sys.exit(app.exec_())

