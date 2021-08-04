import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QToolTip
from PyQt5.QtGui import QIcon, QPixmap, QFont
import time
from PyQt5.QtCore import QTimer,QDateTime
import os
import psutil
from pathlib import Path


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'TF Lite Demo'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 800
        self.initUI()
    
    def initUI(self):

        self.count = 0
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #QToolTip.setFont(QFont('LiberationMono', 20))
        # Create widget
        self.label = QLabel(self)

        pixmap = QPixmap('/home/root/mit-bih_ecg_recognition/imx8mp_result.png')
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())

        label_cpu = QLabel('CPU(%):', self)
        label_cpu.move(400,20)
        label_cpu.setFont(QFont('LiberationMono', 20))
        self.label_cpu_usage = QLabel('1.23', self)
        self.label_cpu_usage.setFont(QFont('LiberationMono', 20))
        self.label_cpu_usage.move(550,20)

        label_power = QLabel('POWER(W):', self)
        label_power.move(400,50)
        label_power.setFont(QFont('LiberationMono', 20))  
        self.label_power_vaule = QLabel('1.23', self)
        self.label_power_vaule.setFont(QFont('LiberationMono', 20))
        self.label_power_vaule.move(550,50)

        self.timer=QTimer()
        self.timer.timeout.connect(self.showSysInfo)
        self.timer.start(1000)
        
        
    def showSysInfo(self):
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load1/os.cpu_count()) * 100
        self.label_cpu_usage.setText(str(cpu_usage))

        file_ina219 = open('/home/root/mit-bih_ecg_recognition/ecg.py', 'r')
        #file_test = Path('/sys/class/hwmon/hwmon0/power1_input')
        if Path('/sys/class/hwmon/hwmon0/power1_input').is_file():
            file_ina219 = open('/sys/class/hwmon/hwmon0/power1_input', 'r')
        elif Path('/sys/class/hwmon/hwmon1/power1_input').is_file():
            file_ina219 = open('/sys/class/hwmon/hwmon1/power1_input', 'r')
        else:
            os.system("reboot")

        power_ina219 = int(file_ina219.readline())/1000000
        self.label_power_vaule.setText(str(power_ina219))

        self.count = self.count + 1
        if self.count == 30:
            new_pixmap = QPixmap('/home/root/mit-bih_ecg_recognition/imx8mp_result.png')
            self.label.setPixmap(new_pixmap)
            self.count = 0



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())