from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, models, training_data, start_date, end_date):
        super().__init__()
        self.models = models
        self.training_data = training_data
        self.start_date = start_date
        self.end_date = end_date
        
    def run(self):
        # ... Training implementation ... 