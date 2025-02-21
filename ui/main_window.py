from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from .menu_bar import create_menu_bar
from .tool_bar import create_tool_bar
from .model_selection import ModelSelectionWidget
from .trained_models import TrainedModelsWidget
from .data_table import DataTableWidget
from .charts import CandlestickWidget, PredictionWidget

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction Analysis System")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Create menu and toolbar
        self.create_menu_bar()
        self.create_tool_bar()
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        
        # Create widgets
        self.model_selection = ModelSelectionWidget()
        self.trained_models = TrainedModelsWidget() 
        self.data_table = DataTableWidget()
        self.candlestick = CandlestickWidget()
        self.prediction = PredictionWidget()
        
        # Layout widgets
        self.setup_layout()
        
    def setup_layout(self):
        # ... Layout code here ... 