import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QMenuBar, QMenu, QAction, QToolBar, 
                           QListWidget, QLabel, QFileDialog, QSplitter,
                           QTableWidget, QTableWidgetItem, QCheckBox,
                           QDateEdit, QPushButton, QGroupBox, QSizePolicy,
                           QHeaderView, QTreeWidget, QTreeWidgetItem,
                           QDialog, QMessageBox, QProgressBar, QProgressDialog,
                           QDateTimeEdit, QFormLayout)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QBrush
import pyqtgraph as pg
import os
import joblib
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import tensorflow as tf

# Add CandlestickItem class - moved to the front
class CandlestickItem(pg.GraphicsObject):
    def __init__(self, x, open, close, high, low, width=0.6, color='r'):
        pg.GraphicsObject.__init__(self)
        self.x = x
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.width = width
        self.color = color
        self.picture = None
        self.generatePicture()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen(self.color))
        p.setBrush(pg.mkBrush(self.color))
        
        # Draw candlestick body
        if self.open > self.close:
            p.setBrush(pg.mkBrush('w'))  # Hollow for bearish
        p.drawRect(pg.QtCore.QRectF(self.x - self.width/2, self.open, 
                                   self.width, self.close - self.open))
        
        # Draw upper and lower shadows
        p.drawLine(pg.QtCore.QLineF(self.x, self.high, self.x, self.open))
        p.drawLine(pg.QtCore.QLineF(self.x, self.close, self.x, self.low))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, str)  # Progress signal
    training_finished = pyqtSignal()         # Completion signal
    training_error = pyqtSignal(str)         # Error signal

    def __init__(self, models, training_data, start_date, end_date):
        super().__init__()
        self.models = models
        self.training_data = training_data
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            from train_model import ModelTrainer
            trainer = ModelTrainer()
            
            for i, model_name in enumerate(self.models):
                try:
                    self.progress_updated.emit(i, f"Training {model_name}...")
                    trainer.train_single_model(model_name, self.training_data)
                except Exception as e:
                    self.training_error.emit(f"Failed to train {model_name}: {str(e)}")
                
            self.progress_updated.emit(len(self.models), "Training completed")
            self.training_finished.emit()
            
        except Exception as e:
            self.training_error.emit(f"Training process error: {str(e)}")

class PredictionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Prediction Date")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Date selection
        date_layout = QHBoxLayout()
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        date_layout.addWidget(QLabel("Prediction Date:"))
        date_layout.addWidget(self.date_edit)
        layout.addLayout(date_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

class StockPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction Analysis System")
        self.setGeometry(100, 100, 1600, 1000)  # Increased window size
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_tool_bar()
        
        # Create main window widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        main_layout = QVBoxLayout(self.main_widget)
        
        # Create top-bottom splitter
        top_bottom_splitter = QSplitter(Qt.Vertical)
        
        # Create top container
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Create top left-right splitter
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Create left model selection area
        self.create_model_selection()
        
        # Create middle trained models area
        self.create_trained_models()
        
        # Create right data table area
        self.create_data_table()
        
        # Add to top splitter
        top_splitter.addWidget(self.model_selection_group)
        top_splitter.addWidget(self.trained_models_group)
        top_splitter.addWidget(self.data_table_group)
        
        # Set split ratio 1:1:2
        top_splitter.setSizes([400, 400, 800])
        top_layout.addWidget(top_splitter)
        
        # Create bottom container
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Create bottom left-right splitter
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Create candlestick chart area
        self.create_candlestick_widget()
        
        # Create prediction chart area
        self.create_prediction_widget()
        
        # Add to bottom splitter
        bottom_splitter.addWidget(self.candlestick_group)
        bottom_splitter.addWidget(self.prediction_group)
        bottom_layout.addWidget(bottom_splitter)
        
        # Add to main splitter
        top_bottom_splitter.addWidget(top_widget)
        top_bottom_splitter.addWidget(bottom_widget)
        
        # Set top-bottom ratio
        top_bottom_splitter.setSizes([400, 600])
        
        main_layout.addWidget(top_bottom_splitter)
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        open_action = QAction('Open Data File', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Model menu
        model_menu = menubar.addMenu('Model')
        train_action = QAction('Train Model', self)
        model_menu.addAction(train_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
    def create_tool_bar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add toolbar buttons
        import_action = QAction('Import Data', self)
        import_action.triggered.connect(lambda: self.load_data('./TSLA.csv'))
        toolbar.addAction(import_action)
        
        predict_action = QAction('Predict', self)
        predict_action.triggered.connect(self.predict_stock)
        toolbar.addAction(predict_action)
        
        toolbar.addAction('Analyze')
        
    def create_model_selection(self):
        self.model_selection_group = QGroupBox("Prediction Model Selection")
        layout = QVBoxLayout()
        
        # Create training date range selection
        date_range_group = QGroupBox("Training Data Range")
        date_layout = QFormLayout()
        
        self.train_start_date = QDateTimeEdit()
        self.train_start_date.setDisplayFormat("yyyy-MM-dd")
        self.train_start_date.setCalendarPopup(True)
        self.train_start_date.setDate(QDate(2014, 1, 1))
        
        self.train_end_date = QDateTimeEdit()
        self.train_end_date.setDisplayFormat("yyyy-MM-dd")
        self.train_end_date.setCalendarPopup(True)
        self.train_end_date.setDate(QDate(2014, 12, 31))
        
        date_layout.addRow("Start Date:", self.train_start_date)
        date_layout.addRow("End Date:", self.train_end_date)
        date_range_group.setLayout(date_layout)
        layout.addWidget(date_range_group)
        
        # Create tree widget
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabel("Available Models")
        
        # Add model categories and specific models
        categories = {
            "Statistical Learning Models": [
                "ARIMA - Autoregressive Integrated Moving Average",
                "SARIMA - Seasonal ARIMA",
                "VAR - Vector Autoregression",
                "Exponential Smoothing"
            ],
            "Machine Learning Models": [
                "Prophet - Facebook Time Series Forecasting",
                "XGBoost - Extreme Gradient Boosting",
                "Random Forest - Time Series Forecasting"
            ],
            "Deep Learning Models": [
                "LSTM - Long Short-Term Memory",
                "GRU - Gated Recurrent Unit"
            ],
            "Other Models": [
                "Kalman Filter"
            ]
        }
        
        for category, models in categories.items():
            category_item = QTreeWidgetItem([category])
            for model in models:
                model_item = QTreeWidgetItem([model])
                model_item.setCheckState(0, Qt.Unchecked)
                category_item.addChild(model_item)
            self.model_tree.addTopLevelItem(category_item)
        
        self.model_tree.expandAll()
        layout.addWidget(self.model_tree)
        
        # Add train button
        train_button = QPushButton("Train Selected Models")
        train_button.clicked.connect(self.train_models)
        layout.addWidget(train_button)
        
        self.model_selection_group.setLayout(layout)
    
    def create_trained_models(self):
        self.trained_models_group = QGroupBox("Trained Models")
        layout = QVBoxLayout()
        
        # Create trained models list
        self.trained_models_tree = QTreeWidget()
        self.trained_models_tree.setHeaderLabel("Model Files")
        
        # Load trained models
        self.load_trained_models()
        
        layout.addWidget(self.trained_models_tree)
        self.trained_models_group.setLayout(layout)
    
    def load_trained_models(self):
        self.trained_models_tree.clear()
        
        if os.path.exists('models'):
            models = {}
            for filename in os.listdir('models'):
                try:
                    # Get model type
                    if filename.endswith('.keras'):  # LSTM/GRU models
                        model_type = filename.split('_')[0]
                    elif filename.endswith('.joblib'):  # Other models
                        model_type = filename.split('_')[0]
                    else:
                        continue  # Skip unknown files
                    
                    # Add to corresponding category
                    if model_type not in models:
                        models[model_type] = []
                    models[model_type].append(filename)
                    
                except Exception as e:
                    print(f"Failed to load model file {filename}: {str(e)}")
            
            # Display models by category
            for model_type, files in models.items():
                type_item = QTreeWidgetItem([model_type])
                for file in sorted(files, reverse=True):  # Newest first
                    file_item = QTreeWidgetItem([file])
                    type_item.addChild(file_item)
                self.trained_models_tree.addTopLevelItem(type_item)
            
            self.trained_models_tree.expandAll()
        
    def create_data_table(self):
        # Create table group
        self.data_table_group = QGroupBox("Data Preview")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        layout.setSpacing(0)  # Reduced spacing
        
        # Create table
        self.data_table = QTableWidget()
        self.data_table.setMinimumHeight(300)
        
        # Set table style
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f6f6f6;
                selection-background-color: #0078d7;
                selection-color: white;
                gridline-color: #d4d4d4;
                border: 1px solid #d4d4d4;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d4d4d4;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
        
        # Set table properties
        self.data_table.setAlternatingRowColors(True)  # Alternate row colors
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)  # Select entire row
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Disable editing
        self.data_table.horizontalHeader().setStretchLastSection(True)  # Last column auto-fill
        self.data_table.verticalHeader().setVisible(False)  # Hide row numbers
        
        # Set table size policy
        self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addWidget(self.data_table)
        self.data_table_group.setLayout(layout)
        
    def create_candlestick_widget(self):
        self.candlestick_group = QGroupBox("Stock Trend Chart")
        layout = QVBoxLayout()
        
        # Create date selection widget
        date_widget = QWidget()
        date_layout = QHBoxLayout(date_widget)
        date_layout.setContentsMargins(0, 5, 0, 5)
        
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        
        date_layout.addWidget(QLabel("Start Date:"))
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("End Date:"))
        date_layout.addWidget(self.end_date)
        
        update_button = QPushButton("Update Chart")
        update_button.clicked.connect(self.update_candlestick)
        date_layout.addWidget(update_button)
        date_layout.addStretch()
        
        # Create chart layout
        plots_layout = QVBoxLayout()
        plots_layout.setSpacing(0)
        
        # Create candlestick chart
        self.price_plot = pg.PlotWidget()
        self.price_plot.setBackground('#FFFFFF')  # Pure white background
        self.price_plot.showGrid(x=True, y=True, alpha=0.2)  # Light grid lines
        self.price_plot.setLabel('left', 'Price ($)', color='#2C3E50', font='Arial')
        
        # Set axis style
        styles = {'color': '#2C3E50', 'font-size': '10pt'}
        self.price_plot.getAxis('left').setTextPen('#2C3E50')
        self.price_plot.getAxis('bottom').setTextPen('#2C3E50')
        
        # Create volume chart
        self.volume_plot = pg.PlotWidget()
        self.volume_plot.setBackground('#FFFFFF')
        self.volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self.volume_plot.setLabel('left', 'Volume', color='#2C3E50', font='Arial')
        
        # Set volume chart axis style
        self.volume_plot.getAxis('left').setTextPen('#2C3E50')
        self.volume_plot.getAxis('bottom').setTextPen('#2C3E50')
        
        # Link X axes of both charts
        self.price_plot.setXLink(self.volume_plot)
        
        # Set candlestick chart ratio to 3, volume chart ratio to 1
        plots_layout.addWidget(self.price_plot, stretch=3)
        plots_layout.addWidget(self.volume_plot, stretch=1)
        
        layout.addWidget(date_widget)
        layout.addLayout(plots_layout)
        self.candlestick_group.setLayout(layout)
        
    def create_prediction_widget(self):
        self.prediction_group = QGroupBox("Prediction Results")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        
        # Create prediction chart
        self.prediction_plot = pg.PlotWidget()
        self.prediction_plot.setBackground('#FFFFFF')
        self.prediction_plot.showGrid(x=True, y=True, alpha=0.2)
        self.prediction_plot.setLabel('left', 'Stock Price', color='#2C3E50', font='Arial')
        self.prediction_plot.setLabel('bottom', 'Time', color='#2C3E50', font='Arial')
        
        # Create prediction info label
        self.prediction_info = QLabel("Prediction information will be displayed here")
        self.prediction_info.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 15px;
                border: 1px solid #d4d4d4;
                border-radius: 4px;
                font-family: Arial;
                font-size: 12pt;
                line-height: 1.5;
            }
        """)
        self.prediction_info.setWordWrap(True)  # Allow text wrapping
        self.prediction_info.setMinimumHeight(100)  # Set minimum height
        
        layout.addWidget(self.prediction_plot, stretch=4)  # Chart takes more space
        layout.addWidget(self.prediction_info, stretch=1)  # Info label takes less space
        self.prediction_group.setLayout(layout)
        
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self.load_data(filename)
    
    def load_data(self, filename):
        try:
            # Read CSV file
            self.df = pd.read_csv(filename)
            
            # Ensure date column format is correct
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # Ensure numeric columns are float type and handle any invalid data
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Drop any rows containing NaN
            self.df = self.df.dropna()
            
            # Ensure data is sorted by date
            self.df = self.df.sort_values('Date')
            
            # Set date picker range
            min_date = self.df['Date'].min()
            max_date = self.df['Date'].max()
            
            self.start_date.setDateRange(
                QDate.fromString(min_date.strftime('%Y-%m-%d'), 'yyyy-MM-dd'),
                QDate.fromString(max_date.strftime('%Y-%m-%d'), 'yyyy-MM-dd')
            )
            self.end_date.setDateRange(
                QDate.fromString(min_date.strftime('%Y-%m-%d'), 'yyyy-MM-dd'),
                QDate.fromString(max_date.strftime('%Y-%m-%d'), 'yyyy-MM-dd')
            )
            
            # Set default date range (last 3 months)
            self.end_date.setDate(QDate.fromString(max_date.strftime('%Y-%m-%d'), 'yyyy-MM-dd'))
            self.start_date.setDate(QDate.fromString((max_date - pd.Timedelta(days=90)).strftime('%Y-%m-%d'), 'yyyy-MM-dd'))
            
            # Update table
            self.update_table()
            
            # Update chart
            self.update_candlestick()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def update_table(self):
        try:
            # Set table row and column count
            self.data_table.setRowCount(min(50, len(self.df)))
            self.data_table.setColumnCount(len(self.df.columns))
            
            # Set headers
            headers = self.df.columns
            self.data_table.setHorizontalHeaderLabels(headers)
            
            # Populate data (first 50 rows)
            for i in range(min(50, len(self.df))):
                for j, col in enumerate(headers):
                    value = self.df.iloc[i][col]
                    
                    # Format display based on data type
                    if col == 'Date':
                        formatted_value = value.strftime('%Y-%m-%d')
                    elif col == 'Volume':
                        formatted_value = f"{value:,.0f}"  # Add thousand separator
                    else:
                        formatted_value = f"{value:.2f}"  # Keep two decimal places
                    
                    item = QTableWidgetItem(formatted_value)
                    
                    # Right-align numeric values
                    if col != 'Date':
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    # Set color based on price change (for Close column)
                    if col == 'Close':
                        if i > 0 and value > self.df.iloc[i-1]['Close']:
                            item.setForeground(QBrush(QColor('red')))
                        elif i > 0 and value < self.df.iloc[i-1]['Close']:
                            item.setForeground(QBrush(QColor('green')))
                    
                    self.data_table.setItem(i, j, item)
            
            # Adjust column width
            self.data_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Date column auto-fit
            remaining_width = self.data_table.width() - self.data_table.columnWidth(0)
            column_width = remaining_width / (len(headers) - 1)
            for j in range(1, len(headers)):
                self.data_table.setColumnWidth(j, column_width)
            
        except Exception as e:
            print(f"Error updating table: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def update_candlestick(self):
        try:
            start_date = self.start_date.date().toPyDate()
            end_date = self.end_date.date().toPyDate()
            
            # Filter data
            mask = (self.df['Date'].dt.date >= start_date) & (self.df['Date'].dt.date <= end_date)
            df_filtered = self.df.loc[mask].copy()
            
            # Clear existing charts
            self.price_plot.clear()
            self.volume_plot.clear()
            
            # Set more professional colors
            up_color = '#FF3B30'    # Up red
            down_color = '#4CD964'  # Down green
            ma_colors = ['#007AFF', '#5856D6', '#FF2D55']  # Moving average colors
            
            # Draw candlestick chart
            for i in range(len(df_filtered)):
                x = i
                open_price = df_filtered['Open'].iloc[i]
                close_price = df_filtered['Close'].iloc[i]
                high_price = df_filtered['High'].iloc[i]
                low_price = df_filtered['Low'].iloc[i]
                
                color = up_color if close_price >= open_price else down_color
                
                self.price_plot.addItem(CandlestickItem(
                    x=x,
                    open=open_price,
                    close=close_price,
                    high=high_price,
                    low=low_price,
                    width=0.8,  # Slightly wider candlesticks
                    color=color
                ))
            
            # Draw volume bars
            volume_data = df_filtered['Volume'].values
            volume_colors = [up_color if df_filtered['Close'].iloc[i] >= df_filtered['Open'].iloc[i] 
                            else down_color for i in range(len(df_filtered))]
            
            for i, volume in enumerate(volume_data):
                self.volume_plot.addItem(pg.BarGraphItem(
                    x=[i], height=[volume],
                    width=0.8,
                    brush=volume_colors[i],
                    alpha=0.7  # Add transparency
                ))
            
            # Set X-axis ticks as dates
            dates = df_filtered['Date'].dt.strftime('%Y-%m-%d').values
            ticks = [(i, dates[i]) for i in range(0, len(dates), len(dates)//10)]
            self.price_plot.getAxis('bottom').setTicks([ticks])
            self.volume_plot.getAxis('bottom').setTicks([ticks])
            
            # Add moving averages
            for i, period in enumerate([5, 10, 20]):
                ma = df_filtered['Close'].rolling(window=period).mean()
                self.price_plot.plot(x=range(len(ma)), y=ma.values, 
                                   pen=pg.mkPen(color=ma_colors[i], width=1.5),
                                   name=f'MA{period}')
            
            # Show legend
            self.price_plot.addLegend(offset=(-10, 10))  # Adjust legend position
            
            # Set Y-axis range with some margin
            price_range = df_filtered['High'].max() - df_filtered['Low'].min()
            self.price_plot.setYRange(
                df_filtered['Low'].min() - price_range * 0.05,
                df_filtered['High'].max() + price_range * 0.05
            )
            
        except Exception as e:
            print(f"Error updating chart: {str(e)}")
            import traceback
            traceback.print_exc()

    def train_models(self):
        # Get selected models
        selected_models = []
        root = self.model_tree.invisibleRootItem()
        for i in range(root.childCount()):
            category = root.child(i)
            for j in range(category.childCount()):
                model_item = category.child(j)
                if model_item.checkState(0) == Qt.Checked:
                    model_name = model_item.text(0).split(' - ')[0]
                    selected_models.append(model_name)
        
        if not selected_models:
            QMessageBox.warning(self, "Warning", "Please select at least one model to train!")
            return
        
        if not hasattr(self, 'df') or self.df is None:
            QMessageBox.warning(self, "Warning", "Please import data first!")
            return
        
        # Get training data range
        start_date = self.train_start_date.date().toPyDate()
        end_date = self.train_end_date.date().toPyDate()
        
        # Filter training data
        training_data = self.df[(self.df['Date'].dt.date >= start_date) & 
                               (self.df['Date'].dt.date <= end_date)].copy()
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Training models...", "Cancel", 0, len(selected_models), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setWindowTitle("Training Progress")
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        
        # Create and start training thread
        self.training_thread = TrainingThread(selected_models, training_data, start_date, end_date)
        
        # Connect signals
        self.training_thread.progress_updated.connect(self.update_training_progress)
        self.training_thread.training_finished.connect(self.training_completed)
        self.training_thread.training_error.connect(self.training_error)
        self.progress_dialog.canceled.connect(self.training_thread.terminate)
        
        # Start training
        self.training_thread.start()
    
    def update_training_progress(self, value, message):
        self.progress_dialog.setValue(value)
        self.progress_dialog.setLabelText(message)
    
    def training_completed(self):
        self.progress_dialog.close()
        self.load_trained_models()  # Refresh trained models list
        QMessageBox.information(self, "Complete", "Model training completed!")
    
    def training_error(self, error_message):
        QMessageBox.warning(self, "Training Error", error_message)
    
    def predict_stock(self):
        # Get trained models
        selected_trained_models = []
        root = self.trained_models_tree.invisibleRootItem()
        for i in range(root.childCount()):
            category = root.child(i)
            for j in range(category.childCount()):
                model_item = category.child(j)
                if model_item.isSelected():
                    selected_trained_models.append(os.path.join('models', model_item.text(0)))
        
        if not selected_trained_models:
            QMessageBox.warning(self, "Warning", "Please select at least one trained model!")
            return
        
        # Use 30 days after training end date for prediction
        end_date = self.train_end_date.date().toPyDate()
        self.perform_prediction(selected_trained_models, end_date)
    
    def perform_prediction(self, selected_trained_models, target_date):
        try:
            # Clear existing prediction chart
            self.prediction_plot.clear()
            
            # Set chart style
            self.prediction_plot.setBackground('#FFFFFF')
            self.prediction_plot.showGrid(x=True, y=True, alpha=0.2)
            self.prediction_plot.setLabel('left', 'Stock Price', color='#2C3E50', font='Arial')
            self.prediction_plot.setLabel('bottom', 'Time', color='#2C3E50', font='Arial')
            
            # Use all historical data
            historical_data = self.df['Close'].values
            x_historical = np.arange(len(historical_data))
            
            # Plot historical data
            self.prediction_plot.plot(x_historical, historical_data, 
                                    pen=pg.mkPen(color='#2C3E50', width=1.5),
                                    name='Historical Data')
            
            # Fixed prediction of 30 days
            days_to_predict = 30
            predictions = []
            
            # Predict for each selected trained model
            for model_path in selected_trained_models:
                try:
                    model_name = os.path.basename(model_path).split('_')[0]
                    print(f"Loading model: {model_path}")  # Add debug info
                    
                    if model_path.endswith('.keras'):  # LSTM/GRU models
                        # Load model using tensorflow
                        model = tf.keras.models.load_model(model_path)
                        # Prepare prediction data
                        last_sequence = self.df['Close'].values[-10:].reshape(1, 10, 1)
                        pred_series = []
                        for _ in range(days_to_predict):
                            pred = model.predict(last_sequence, verbose=0)[0][0]
                            pred_series.append(pred)
                            last_sequence = np.roll(last_sequence, -1)
                            last_sequence[0, -1, 0] = pred
                        pred_series = np.array(pred_series)
                    else:  # Other models
                        model = joblib.load(model_path)
                        if model_name == 'ARIMA':
                            pred_series = model.forecast(steps=days_to_predict)
                        elif model_name == 'Prophet':
                            future_dates = pd.date_range(
                                start=self.df['Date'].max(),
                                periods=days_to_predict+1,
                                freq='D'
                            )[1:]
                            future = pd.DataFrame({'ds': future_dates})
                            forecast = model.predict(future)
                            pred_series = forecast['yhat'].values
                        # ... Other model prediction logic ...
                    
                    predictions.append((model_name, pred_series))
                    print(f"Model {model_name} prediction completed")  # Add debug info
                    
                except Exception as e:
                    print(f"Model {model_path} prediction failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Plot prediction results
            x_future = np.arange(len(historical_data), len(historical_data) + days_to_predict)
            colors = ['#FF3B30', '#4CD964', '#007AFF', '#5856D6', '#FF2D55']
            
            # Plot each model's prediction
            for i, (model_name, pred_series) in enumerate(predictions):
                color = colors[i % len(colors)]
                # Plot prediction line
                self.prediction_plot.plot(x_future, pred_series, 
                                        pen=pg.mkPen(color=color, width=2),
                                        name=f'{model_name} Predictions')
            
            # Add legend
            self.prediction_plot.addLegend(offset=(10, 10))
            
            # Set Y-axis range to include all data
            all_values = np.concatenate([historical_data] + [p[1] for p in predictions])
            y_min, y_max = np.min(all_values), np.max(all_values)
            y_range = y_max - y_min
            self.prediction_plot.setYRange(y_min - y_range * 0.05, y_max + y_range * 0.05)
            
            # Update prediction info
            target_date_str = target_date.strftime('%Y-%m-%d')
            prediction_info = (f"Prediction time range: {self.df['Date'].max().strftime('%Y-%m-%d')} to "
                             f"{(self.df['Date'].max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')}\n\n"
                             f"Each model's 30-day prediction ({target_date_str}):\n")
            
            # Calculate each model's final prediction and average
            final_predictions = []
            for model_name, pred_series in predictions:
                final_pred = pred_series[-1]
                final_predictions.append(final_pred)
                prediction_info += f"{model_name}: ${final_pred:.2f}\n"
            
            # Calculate and display prediction statistics
            avg_prediction = np.mean(final_predictions)
            std_prediction = np.std(final_predictions)
            prediction_info += f"\nPrediction Statistics:\n"
            prediction_info += f"Average Prediction: ${avg_prediction:.2f}\n"
            prediction_info += f"Prediction Std Dev: ${std_prediction:.2f}\n"
            prediction_info += f"Prediction Range: ${avg_prediction-std_prediction:.2f} ~ ${avg_prediction+std_prediction:.2f}"
            
            self.prediction_info.setText(prediction_info)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction error: {str(e)}")
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StockPredictionApp()
    window.show()
    sys.exit(app.exec_())