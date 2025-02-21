import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import StockPredictionApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_()) 