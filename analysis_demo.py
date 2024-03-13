import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QTableView, QComboBox, QLabel, QHBoxLayout, QInputDialog)
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import QAbstractItemView
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        elif role == Qt.ToolTipRole:
            # 返回用于工具提示的单元格内容
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False
        try:
            self._df.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, (role,))
            return True
        except Exception as e:
            QMessageBox.warning(None, "Input Error", str(e))
            return False

    def flags(self, index):
        return super().flags(index) | Qt.ItemIsEditable

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis App")
        self.setGeometry(100, 100, 1200, 800)
        self.df = pd.DataFrame()
        self.initUI()

    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout()

        self.tableView = QTableView()
        self.model = PandasModel(self.df)
        self.tableView.setModel(self.model)
        layout.addWidget(self.tableView)

        self.columnSelectorX = QComboBox()
        self.columnSelectorY = QComboBox()

        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)
        self.saveButton = QPushButton("Save Data")
        self.saveButton.clicked.connect(self.saveData)
        self.zScoreButton = QPushButton("Z-Score Normalization")
        self.zScoreButton.clicked.connect(self.normalizeZScore)
        self.minMaxButton = QPushButton("Min-Max Normalization")
        self.minMaxButton.clicked.connect(self.normalizeMinMax)
        self.regressionButton = QPushButton("Perform Regression")
        self.regressionButton.clicked.connect(self.performRegression)
        self.handleMissingButton = QPushButton("Handle Missing Values")
        self.handleMissingButton.clicked.connect(self.handleMissingValues)
        self.lofButton = QPushButton("Detect Anomalies (LOF)")
        self.lofButton.clicked.connect(self.detectAnomaliesLOF)
        self.plotButton = QPushButton("Plot Selected Data")
        self.plotButton.clicked.connect(self.plotSelectedData)
        self.removeDuplicatesButton = QPushButton("Remove Duplicates")
        self.removeDuplicatesButton.clicked.connect(self.removeDuplicates)
        self.plotDataButton = QPushButton("Plot Data")
        self.plotDataButton.clicked.connect(self.plotData)


        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.loadButton)
        btnLayout.addWidget(self.saveButton)
        btnLayout.addWidget(self.zScoreButton)
        btnLayout.addWidget(self.minMaxButton)
        btnLayout.addWidget(self.lofButton)
        btnLayout.addWidget(self.handleMissingButton)
        btnLayout.addWidget(self.removeDuplicatesButton)
        btnLayout.addWidget(self.plotDataButton)
        
        selectLayout = QHBoxLayout()
        selectLayout.addWidget(QLabel("X-axis:"))
        selectLayout.addWidget(self.columnSelectorX)
        selectLayout.addWidget(QLabel("Y-axis:"))
        selectLayout.addWidget(self.columnSelectorY)
        selectLayout.addWidget(self.regressionButton)
        selectLayout.addWidget(self.plotButton)

        layout.addLayout(btnLayout)
        layout.addLayout(selectLayout)
        self.centralWidget.setLayout(layout)
    
    def plotWithAnnotations(self, plotType, cols_list):
        fig, ax = plt.subplots()
        annot = ax.annotate("", xy=(0,0), xytext=(10,10),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(artist, x, y, text):
            annot.xy = (x, y)
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for artist in artists:
                    cont, ind = artist.contains(event)
                    if cont:
                        x, y = artist.get_offsets().data[ind["ind"][0]]
                        label = artist.get_label()
                        update_annot(artist, x, y, f"{label}: ({x:.2f}, {y:.2f})")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

        artists = []
        if plotType == "Scatter" and len(cols_list) >= 2:
            scatter = ax.scatter(self.df[cols_list[0]], self.df[cols_list[1]], label=f"{cols_list[0]} vs {cols_list[1]}")
            artists.append(scatter)
        elif plotType == "Line" and len(cols_list) >= 1:
            for col in cols_list:
                line, = ax.plot(self.df[col], label=col, picker=5)  # `picker=5` makes the line pickable
                artists.append(line)

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.legend()
        plt.show()



    def plotData(self):
        # 获取用户选择的图表类型和列
        plotType, ok = QInputDialog.getItem(self, "Plot Data", "Select plot type:", ["Scatter", "Line", "Bar"], 0, False)
        if ok and plotType:
            # 获取用户选择的列
            cols, ok = QInputDialog.getText(self, "Plot Data", "Enter columns to plot (comma-separated):")
            if ok and cols:
                cols_list = [col.strip() for col in cols.split(',')]
                if plotType == "Scatter" and len(cols_list) < 2:
                    QMessageBox.warning(self, "Input Error", "Scatter plot requires at least two columns.")
                    return
                self.plotWithAnnotations(plotType, cols_list)



    def drawPlot(self, plotType, cols_list):
        try:
            # 根据所选图表类型绘制图表
            if plotType == "Scatter":
                self.plotScatter(cols_list)
            elif plotType == "Line":
                self.plotLine(cols_list)
            elif plotType == "Bar":
                self.plotBar(cols_list)
        except Exception as e:
            QMessageBox.critical(self, "Error Plotting Data", str(e))

    def plotScatter(self, cols_list):
        if all(col in self.df.columns for col in cols_list):
            self.df.plot.scatter(x=cols_list[0], y=cols_list[1])
            plt.show()
        else:
            QMessageBox.warning(self, "Input Error", "One or more selected columns do not exist in the dataset.")


    def plotLine(self, cols_list):
        for col in cols_list:
            if col.strip() in self.df.columns:
                plt.plot(self.df[col.strip()], label=col.strip())
        plt.legend()
        plt.show()

    def plotBar(self, cols_list):
        self.df[cols_list].plot(kind='bar')
        plt.show()


    def handleMissingValues(self):
        col, ok = QInputDialog.getText(self, "Handle Missing Values", "Enter column name:")
        if ok and col:
            try:
                if col in self.df.columns:
                    method, ok = QInputDialog.getItem(self, "Fill Method", "Select fill method:", ["mean", "median", "mode"], 0, False)
                    if ok:
                        if method == "mean":
                            self.df[col + "_filled"] = self.df[col].fillna(self.df[col].mean())
                        elif method == "median":
                            self.df[col + "_filled"] = self.df[col].fillna(self.df[col].median())
                        elif method == "mode":
                            self.df[col + "_filled"] = self.df[col].fillna(self.df[col].mode()[0])
                        self.model.layoutChanged.emit()
                else:
                    QMessageBox.warning(self, "Error", "Column not found.")
            except Exception as e:
                QMessageBox.critical(self, "Error Handling Missing Values", str(e))

    def detectAnomaliesLOF(self):
        col, ok = QInputDialog.getText(self, "Detect Anomalies (LOF)", "Enter column name:")
        if ok and col:
            if col in self.df.columns:
                try:
                    # 确保列数据为数值型
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        X = self.df[[col]]  # LOF 需要二维数据
                        lof = LocalOutlierFactor()
                        # fit_predict 返回一个 numpy array，异常值标记为 -1
                        outliers = lof.fit_predict(X)
                        # 使用中位数替换异常值
                        median_value = self.df[col].median()
                        # 创建一个新列，其中异常值被替换为中位数，其他值保持不变
                        self.df[col + '_LOF_Treated'] = np.where(outliers == -1, median_value, self.df[col])
                        QMessageBox.information(self, "LOF Detection", "Anomalies in column '" + col + "' have been treated and stored in '" + col + "_LOF_Treated'.")
                        self.model.layoutChanged.emit()
                    else:
                        QMessageBox.warning(self, "Invalid Column Type", "The selected column must contain numeric data.")
                except Exception as e:
                    QMessageBox.critical(self, "Error Detecting Anomalies", str(e))
            else:
                QMessageBox.warning(self, "Column Not Found", f"The column '{col}' was not found in the dataset.")



    def plotSelectedData(self):
        try:
            cols, ok = QInputDialog.getText(self, "Plot Data", "Enter columns to plot (comma-separated):")
            if ok and cols:
                cols_list = cols.split(',')
                fig, ax = plt.subplots()
                for col in cols_list:
                    if col.strip() in self.df.columns:
                        if pd.api.types.is_datetime64_any_dtype(self.df[col.strip()]):
                            ax.plot(pd.to_datetime(self.df[col.strip()]), self.df.index, label=col.strip())
                        else:
                            ax.plot(self.df[col.strip()], label=col.strip())
                plt.legend()
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error Plotting Data", str(e))

    def loadData(self):
        try:
            filePath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
            if filePath:
                if filePath.endswith('.csv'):
                    self.df = pd.read_csv(filePath)
                else:
                    self.df = pd.read_excel(filePath)
                self.model = PandasModel(self.df)
                self.tableView.setModel(self.model)
                self.columnSelectorX.clear()
                self.columnSelectorY.clear()
                self.columnSelectorX.addItems(self.df.columns)
                self.columnSelectorY.addItems(self.df.columns)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Data", str(e))

    def saveData(self):
        try:
            filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
            if filePath:
                # Check for the proper extension and save accordingly
                if filePath.endswith('.csv'):
                    self.df.to_csv(filePath, index=False)
                elif filePath.endswith('.xlsx'):
                    self.df.to_excel(filePath, index=False)
                else:
                    # If the file path does not have a proper extension, default to CSV
                    filePath += '.csv'
                    self.df.to_csv(filePath, index=False)
                QMessageBox.information(self, "Success", "Data saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Data", str(e))


    def normalizeZScore(self):
        col, ok = QInputDialog.getText(self, "Z-Score Normalization", "Enter column name:")
        if ok and col:
            if col in self.df.columns:
                try:
                    self.df[col + '_zscore'] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                    QMessageBox.information(self, "Success", f"Z-Score normalization applied to {col}. Result in {col}_zscore.")
                    self.model.layoutChanged.emit()
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))
            else:
                QMessageBox.warning(self, "Error", "Column not found.")
    
    def normalizeMinMax(self):
        col, ok = QInputDialog.getText(self, "Min-Max Normalization", "Enter column name:")
        if ok and col:
            if col in self.df.columns:
                try:
                    self.df[col + '_minmax'] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())
                    QMessageBox.information(self, "Success", f"Min-Max normalization applied to {col}. Result in {col}_minmax.")
                    self.model.layoutChanged.emit()
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))
            else:
                QMessageBox.warning(self, "Error", "Column not found.")

    def performRegression(self):
        try:
            x_col = self.columnSelectorX.currentText()
            y_col = self.columnSelectorY.currentText()
            if not self.df.empty and x_col and y_col:
                # Handle datetime for X
                if pd.api.types.is_datetime64_any_dtype(self.df[x_col]):
                    X = self.df[x_col].map(datetime.toordinal).values.reshape(-1, 1)
                    formatter = mdates.DateFormatter("%Y-%m-%d")
                else:
                    X = self.df[[x_col]].values.reshape(-1, 1)
                    formatter = None

                y = self.df[y_col].values

                regression = LinearRegression().fit(X, y)
                y_pred = regression.predict(X)

                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue', label='Actual Data')
                ax.plot(X, y_pred, color='red', linewidth=2, label='Fitted Line')

                if formatter:
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(formatter)
                    fig.autofmt_xdate()

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Regression Analysis: {x_col} vs {y_col}')
                ax.legend()
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Regression Error", str(e))

    def performRegression(self):
        try:
            x_col = self.columnSelectorX.currentText()
            y_col = self.columnSelectorY.currentText()
            if not self.df.empty and x_col and y_col:
                # Handle datetime for X
                if pd.api.types.is_datetime64_any_dtype(self.df[x_col]):
                    X = self.df[x_col].map(datetime.toordinal).values.reshape(-1, 1)
                    formatter = mdates.DateFormatter("%Y-%m-%d")
                else:
                    X = self.df[[x_col]].values.reshape(-1, 1)
                    formatter = None

                y = self.df[y_col].values

                regression = LinearRegression().fit(X, y)
                y_pred = regression.predict(X)

                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue', label='Actual Data')
                ax.plot(X, y_pred, color='red', linewidth=2, label='Fitted Line')

                if formatter:
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(formatter)
                    fig.autofmt_xdate()

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Regression Analysis: {x_col} vs {y_col}')
                ax.legend()
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Regression Error", str(e))
        
    def removeDuplicates(self):
        try:
            original_rows = len(self.df)
            self.df.drop_duplicates(inplace=True)  
            removed_rows = original_rows - len(self.df)
            QMessageBox.information(self, "Remove Duplicates", f"{removed_rows} duplicate rows removed.")
            self.model.layoutChanged.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error Removing Duplicates", str(e))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
