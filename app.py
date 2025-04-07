import sys
import logging
import json
import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QStackedWidget,
                             QPushButton, QFileDialog, QListWidget, QMessageBox,
                             QTableWidget, QTableWidgetItem, QSizePolicy, QProgressDialog,
                             QDialog, QLineEdit, QFormLayout)
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# 设置日志记录
logging.basicConfig(filename='application.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')


# 定义接口 IDataProcessor
class IDataProcessor(ABC):
    @abstractmethod
    def load_father_data(self, file_path):
        pass

    @abstractmethod
    def load_kid_data(self, file_path):
        pass

    @abstractmethod
    def clean_and_prepare_data(self, father_df, kid_df):
        pass

    @abstractmethod
    def perform_analysis(self, father_df, kid_df):
        pass

    @abstractmethod
    def save_results(self, file_path):
        pass


# 数据清理类
class DataCleaner:
    def clean_and_prepare_data(self, father_df, kid_df):
        # 去掉列名中的前导和尾随空格
        father_df.columns = father_df.columns.str.strip()
        kid_df.columns = kid_df.columns.str.strip()

        # 清理并转换 Size 1 和 Size 2 列为浮点数，处理错误
        father_df['Size 1'] = pd.to_numeric(father_df['Size 1'], errors='coerce')
        father_df['Size 2'] = pd.to_numeric(father_df['Size 2'], errors='coerce')
        kid_df['Size 1'] = pd.to_numeric(kid_df['Size 1'], errors='coerce')
        kid_df['Size 2'] = pd.to_numeric(kid_df['Size 2'], errors='coerce')

        # 删除关键列中包含 NaN 值的行
        father_df.dropna(subset=['Size 1', 'Size 2'], inplace=True)
        kid_df.dropna(subset=['Size 1', 'Size 2'], inplace=True)

        # 基于 Sample Name 和 Marker 去除重复值
        father_df = father_df.drop_duplicates(subset=['Sample Name', 'Marker'])
        kid_df = kid_df.drop_duplicates(subset=['Sample Name', 'Marker'])

        # 打印清理后的数据
        print("Father Data after cleaning:")
        print(father_df.head())
        print("\nKid Data after cleaning:")
        print(kid_df.head())

        return father_df, kid_df


# 数据处理类实现接口 IDataProcessor
class DataProcessor(IDataProcessor):
    def __init__(self):
        self.father_file_path = None
        self.kid_file_path = None
        self.match_results = None
        self.cleaner = DataCleaner()
        self.excluded_markers = ['DXS2506']  # 需要排除的性别点位

    def load_father_data(self, file_path):
        self.father_file_path = file_path
        return pd.read_excel(self.father_file_path)

    def load_kid_data(self, file_path):
        self.kid_file_path = file_path
        return pd.read_excel(self.kid_file_path)

    def clean_and_prepare_data(self, father_df, kid_df):
        return self.cleaner.clean_and_prepare_data(father_df, kid_df)

    def calculate_match_score(self, kid_data, father_data, markers, threshold=1):
        score = 0
        for marker in markers:
            if marker in self.excluded_markers:  # 跳过排除的点位
                continue

            if marker in kid_data and marker in father_data:
                kid_size1 = kid_data[marker]['Size 1']
                kid_size2 = kid_data[marker]['Size 2']
                father_size1 = father_data[marker]['Size 1']
                father_size2 = father_data[marker]['Size 2']

                if (abs(kid_size1 - father_size1) <= threshold or abs(kid_size1 - father_size2) <= threshold or
                        abs(kid_size2 - father_size1) <= threshold or abs(kid_size2 - father_size2) <= threshold):
                    score += 1
        return score

    def calculate_marker_sums(self, data_dict):
        """计算每个样本的所有点位Size1和Size2的总和"""
        sums_dict = {}
        for sample_name, marker_data in data_dict.items():
            total_size1 = 0
            total_size2 = 0
            for marker, sizes in marker_data.items():
                if marker in self.excluded_markers:  # 跳过排除的点位
                    continue

                if 'Size 1' in sizes and not pd.isna(sizes['Size 1']):
                    total_size1 += sizes['Size 1']
                if 'Size 2' in sizes and not pd.isna(sizes['Size 2']):
                    total_size2 += sizes['Size 2']
            sums_dict[sample_name] = {
                'Total_Size1': total_size1,
                'Total_Size2': total_size2,
                'Total_Sum': total_size1 + total_size2
            }
        return sums_dict

    def perform_analysis(self, father_df, kid_df):
        # 获取所有点位并排除性别点位
        markers = [m for m in father_df['Marker'].unique() if m not in self.excluded_markers]

        # 修改校验逻辑：至少 20 个点位
        if len(markers) < 20:
            raise ValueError(f"需要至少 20 个点位(已排除性别点位)，当前为 {len(markers)}")
        markers = markers[:20]  # 取前 20 个非性别点位

        # 将父亲和孩子的数据转换为字典
        father_dict = {name: group.set_index('Marker').to_dict(orient='index')
                       for name, group in father_df.groupby('Sample Name')}
        kid_dict = {name: group.set_index('Marker').to_dict(orient='index')
                    for name, group in kid_df.groupby('Sample Name')}

        # 计算所有样本的点位总和(已排除性别点位)
        father_sums = self.calculate_marker_sums(father_dict)
        kid_sums = self.calculate_marker_sums(kid_dict)

        match_results = []

        # 分析每个孩子
        for kid_name, kid_data in kid_dict.items():
            try:
                result = self.calculate_best_match(kid_name, kid_data, father_dict, markers)

                # 添加点位总和信息(已排除性别点位)
                if kid_name in kid_sums:
                    kid_sum = kid_sums[kid_name]
                    result.update({
                        'Kid_Total_Size1': kid_sum['Total_Size1'],
                        'Kid_Total_Size2': kid_sum['Total_Size2'],
                        'Kid_Total_Sum': kid_sum['Total_Sum']
                    })

                if result['Matched Father Sample Name'] in father_sums:
                    father_sum = father_sums[result['Matched Father Sample Name']]
                    result.update({
                        'Father_Total_Size1': father_sum['Total_Size1'],
                        'Father_Total_Size2': father_sum['Total_Size2'],
                        'Father_Total_Sum': father_sum['Total_Sum']
                    })

                match_results.append(result)
            except Exception as e:
                logging.error(f"Error processing match for {kid_name}: {e}")

        self.match_results = pd.DataFrame(match_results)
        return self.match_results

    def calculate_best_match(self, kid_name, kid_data, father_dict, markers):
        best_match = None
        best_score = -1
        for father_name, father_data in father_dict.items():
            score = self.calculate_match_score(kid_data, father_data, markers)
            if score > best_score:
                best_score = score
                best_match = father_name

        return {
            'Kid Sample Name': kid_name,
            'Matched Father Sample Name': best_match,
            'Match Score': best_score,
            'Total Markers': 20,
            'Match Percentage': (best_score / 20) * 100,
            'Kid_Total_Size1': 0,
            'Kid_Total_Size2': 0,
            'Kid_Total_Sum': 0,
            'Father_Total_Size1': 0,
            'Father_Total_Size2': 0,
            'Father_Total_Sum': 0
        }

    def save_results(self, file_path):
        if self.match_results is not None:
            self.match_results.to_excel(file_path, index=False)
        else:
            raise ValueError("No results to save")


# 分析线程类
class AnalysisThread(QThread):
    update_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(pd.DataFrame)

    def __init__(self, processor, father_df, kid_df):
        super().__init__()
        self.processor = processor
        self.father_df = father_df
        self.kid_df = kid_df

    def run(self):
        try:
            # 清理数据
            self.update_progress.emit(5, "正在清理数据...")
            father_df, kid_df = self.processor.clean_and_prepare_data(self.father_df, self.kid_df)

            # 获取所有点位并排除性别点位
            markers = [m for m in father_df['Marker'].unique() if m not in self.processor.excluded_markers]
            markers = markers[:20]  # 取前20个非性别点位

            # 将数据转换为字典
            self.update_progress.emit(10, "正在准备分析数据...")
            father_dict = {name: group.set_index('Marker').to_dict(orient='index')
                           for name, group in father_df.groupby('Sample Name')}
            kid_dict = {name: group.set_index('Marker').to_dict(orient='index')
                        for name, group in kid_df.groupby('Sample Name')}

            # 计算所有样本的点位总和(已排除性别点位)
            self.update_progress.emit(15, "正在计算点位总和...")
            father_sums = self.processor.calculate_marker_sums(father_dict)
            kid_sums = self.processor.calculate_marker_sums(kid_dict)

            match_results = []

            # 分析每个孩子
            total_kids = len(kid_dict)
            for i, (kid_name, kid_data) in enumerate(kid_dict.items()):
                progress = 20 + int((i / total_kids) * 80)
                self.update_progress.emit(progress, f"正在分析 {kid_name} 的匹配情况...")

                result = self.processor.calculate_best_match(kid_name, kid_data, father_dict, markers)

                # 添加点位总和信息(已排除性别点位)
                if kid_name in kid_sums:
                    kid_sum = kid_sums[kid_name]
                    result.update({
                        'Kid_Total_Size1': kid_sum['Total_Size1'],
                        'Kid_Total_Size2': kid_sum['Total_Size2'],
                        'Kid_Total_Sum': kid_sum['Total_Sum']
                    })

                if result['Matched Father Sample Name'] in father_sums:
                    father_sum = father_sums[result['Matched Father Sample Name']]
                    result.update({
                        'Father_Total_Size1': father_sum['Total_Size1'],
                        'Father_Total_Size2': father_sum['Total_Size2'],
                        'Father_Total_Sum': father_sum['Total_Sum']
                    })

                match_results.append(result)

                # 实时发送部分结果
                partial_results = pd.DataFrame(match_results)
                self.analysis_complete.emit(partial_results)

            # 发送最终结果
            final_results = pd.DataFrame(match_results)
            self.analysis_complete.emit(final_results)
            self.update_progress.emit(100, "分析完成！")

        except Exception as e:
            logging.error(f"Error in analysis thread: {e}")
            self.update_progress.emit(100, f"分析出错: {str(e)}")


# 新建项目对话框
class NewProjectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("新建项目")
        self.project_name = None

        layout = QFormLayout(self)
        self.project_name_input = QLineEdit(self)
        layout.addRow("项目名称:", self.project_name_input)

        button_box = QHBoxLayout()
        self.create_button = QPushButton("创建")
        self.create_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addWidget(self.create_button)
        button_box.addWidget(self.cancel_button)

        layout.addRow(button_box)

    def get_project_name(self):
        return self.project_name_input.text()


# 项目页面
class ProjectPage(QWidget):
    def __init__(self, project_name, main_window, project_file=None):
        super().__init__()
        self.main_window = main_window
        self.project_name = project_name
        self.processor = DataProcessor()

        self.father_file_path = None
        self.father_df = None
        self.kid_file_path = None
        self.kid_df = None
        self.match_results = None
        self.progress_dialog = None
        self.analysis_thread = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        title_label = QLabel(f"项目名称: {self.project_name}")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #FFFFFF; background-color: #004080; padding: 10px; border-radius: 8px;")
        main_layout.addWidget(title_label)

        self.result_list = QListWidget()
        self.result_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_list.setStyleSheet("background-color: #f0f0f0; font-size: 14px; border-radius: 8px; padding: 5px;")
        main_layout.addWidget(self.result_list)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(11)
        self.result_table.setHorizontalHeaderLabels(
            ['Kid Sample Name', 'Matched Father Sample Name', 'Match Score',
             'Total Markers', 'Match Percentage',
             'Kid Total Size1', 'Kid Total Size2', 'Kid Total Sum',
             'Father Total Size1', 'Father Total Size2', 'Father Total Sum'])
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_table.setStyleSheet("""
            QTableWidget::item { padding: 10px; border-bottom: 1px solid #CCCCCC; }
            QHeaderView::section {
                background-color: #004080;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-bottom: 1px solid #CCCCCC;
            }
            QTableWidget { background-color: #ffffff; border-radius: 8px; }
        """)
        main_layout.addWidget(self.result_table)

        self.setLayout(main_layout)

        if project_file:
            self.load_project(project_file)

    def save_project(self):
        project_data = {
            "project_name": self.project_name,
            "father_file_path": self.father_file_path,
            "kid_file_path": self.kid_file_path,
            "match_results": self.processor.match_results.to_dict() if self.processor.match_results is not None else None
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "保存项目", "", "Project Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(project_data, f)
            QMessageBox.information(self, "保存成功", f"项目已成功保存到: {file_path}")

    def load_project(self, project_file):
        with open(project_file, 'r') as f:
            project_data = json.load(f)

        self.project_name = project_data["project_name"]
        self.father_file_path = project_data["father_file_path"]
        self.kid_file_path = project_data["kid_file_path"]

        if self.father_file_path and os.path.exists(self.father_file_path):
            self.father_df = pd.read_excel(self.father_file_path)
            self.result_list.addItem(f"父本基因库文件已导入: {self.father_file_path}")

        if self.kid_file_path and os.path.exists(self.kid_file_path):
            self.kid_df = pd.read_excel(self.kid_file_path)
            self.result_list.addItem(f"子本文件已导入: {self.kid_file_path}")
            self.display_kid_data()

        if project_data["match_results"]:
            self.processor.match_results = pd.DataFrame(project_data["match_results"])
            self.display_analysis_results(self.processor.match_results)

    def import_father_data(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择父本基因库文件", "", "Excel Files (*.xlsx)")
            if not file_path:
                return
            self.father_file_path = file_path

            if not file_path.endswith('.xlsx'):
                raise ValueError("请选择正确的 Excel 文件 (*.xlsx)")

            self.father_df = pd.read_excel(self.father_file_path)

            if self.father_df.empty:
                raise ValueError("父本基因库文件为空或无效数据，请检查文件内容。")

            self.result_list.addItem(f"父本基因库文件已导入: {file_path}")
            self.result_list.addItem(f"父本基因库共包含 {len(self.father_df)} 条记录")
            self.main_window.import_kid_button.setEnabled(True)

        except ValueError as ve:
            self.log_and_display_error("文件选择错误", ve)
        except pd.errors.EmptyDataError as ede:
            self.log_and_display_error("文件内容错误", ede)
        except Exception as e:
            self.log_and_display_error("导入父本基因库文件时发生错误", e)

    def import_kid_data(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择子本文件", "", "Excel Files (*.xlsx)")
            if file_path:
                self.kid_file_path = file_path
                self.kid_df = pd.read_excel(self.kid_file_path)
                self.result_list.addItem(f"子本文件已导入: {file_path}")
                self.result_list.addItem(f"子本基因库共包含 {len(self.kid_df)} 条记录")
                self.display_kid_data()
                self.main_window.analyze_button.setEnabled(True)
        except Exception as e:
            self.log_and_display_error("导入子本文件时发生错误", e)

    def display_kid_data(self):
        if self.kid_df is not None:
            self.result_table.setRowCount(len(self.kid_df))
            self.result_table.setColumnCount(len(self.kid_df.columns))
            self.result_table.setHorizontalHeaderLabels(self.kid_df.columns)

            for row in self.kid_df.itertuples():
                for col_index, value in enumerate(row[1:], start=0):
                    self.result_table.setItem(row.Index, col_index, QTableWidgetItem(str(value)))

    def analyze_data(self):
        if not self.father_file_path or not self.kid_file_path:
            QMessageBox.warning(self, "错误", "请先导入父本基因库和子本文件。")
            return

        try:
            self.main_window.analyze_button.setEnabled(False)
            self.main_window.import_father_button.setEnabled(False)
            self.main_window.import_kid_button.setEnabled(False)
            self.main_window.export_button.setEnabled(False)

            # 创建进度对话框
            self.progress_dialog = QProgressDialog("正在分析数据...", "取消", 0, 100, self)
            self.progress_dialog.setWindowTitle("分析进度")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setAutoClose(True)
            self.progress_dialog.setAutoReset(True)
            self.progress_dialog.canceled.connect(self.cancel_analysis)

            # 加载数据
            father_df = self.processor.load_father_data(self.father_file_path)
            kid_df = self.processor.load_kid_data(self.kid_file_path)

            # 创建并启动分析线程
            self.analysis_thread = AnalysisThread(self.processor, father_df, kid_df)
            self.analysis_thread.update_progress.connect(self.update_progress)
            self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
            self.analysis_thread.finished.connect(self.on_analysis_finished)
            self.analysis_thread.start()

            # 显示进度对话框
            self.progress_dialog.show()

        except Exception as e:
            self.log_and_display_error("数据分析时发生错误", e)
            self.reset_buttons()

    def update_progress(self, value, message):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
            self.result_list.addItem(message)
            self.result_list.scrollToBottom()

    def on_analysis_complete(self, results):
        # 保存结果到处理器和本地变量
        self.processor.match_results = results
        self.match_results = results
        self.display_analysis_results(results)

    def on_analysis_finished(self):
        if self.progress_dialog:
            self.progress_dialog.close()
        self.main_window.export_button.setEnabled(True)
        self.reset_buttons()

    def cancel_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.result_list.addItem("分析已取消")
            self.reset_buttons()

    def reset_buttons(self):
        self.main_window.analyze_button.setEnabled(True)
        self.main_window.import_father_button.setEnabled(True)
        self.main_window.import_kid_button.setEnabled(True)

    def display_analysis_results(self, match_results):
        self.result_table.setRowCount(len(match_results))
        self.result_table.setColumnCount(len(match_results.columns))
        self.result_table.setHorizontalHeaderLabels(match_results.columns)

        for row in match_results.itertuples():
            for col_index, value in enumerate(row[1:], start=0):
                item = QTableWidgetItem(str(value))
                if 'Total' in match_results.columns[col_index]:
                    # 对总和列设置不同颜色
                    item.setForeground(QColor('#006400'))  # 深绿色
                self.result_table.setItem(row.Index, col_index, item)

    def export_results(self):
        try:
            if self.processor.match_results is not None:
                file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Excel Files (*.xlsx)")
                if file_path:
                    self.processor.save_results(file_path)
                    QMessageBox.information(self, "导出成功", f"结果已成功导出到: {file_path}")
            else:
                QMessageBox.warning(self, "错误", "没有可导出的结果，请先进行分析。")
        except Exception as e:
            self.log_and_display_error("导出结果时发生错误", e)

    def log_and_display_error(self, message, exception):
        logging.error(f"{message}: {exception}")
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(message)
        error_dialog.setInformativeText("详细信息如下：")
        error_dialog.setWindowTitle("错误")
        error_dialog.setDetailedText(f"{type(exception).__name__}: {str(exception)}")
        error_dialog.exec_()


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Genetic Analysis App")
        self.setGeometry(100, 100, 1200, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #f9f9f9; }
            QLabel#title {
                font-size: 28px; font-weight: bold; margin-bottom: 20px;
            }
            QPushButton#actionButton {
                background-color: #004080;
                border: none; color: white; font-size: 16px;
                padding: 8px; margin: 10px 0; border-radius: 8px;
                transition: background-color 0.3s ease;
            }
            QPushButton#actionButton:hover { background-color: #0066cc; }
            QWidget#sidebar {
                background-color: #004080; padding: 10px;
                border-radius: 8px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
            }
            QPushButton#sidebarButton {
                background-color: transparent; color: white;
                font-size: 14px; padding: 8px; margin: 5px 0;
                text-align: left; border: none; border-radius: 5px;
                transition: background-color 0.3s ease;
            }
            QPushButton#sidebarButton:hover { background-color: #0066cc; }
        """)

        main_layout = QGridLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)

        self.new_project_button = QPushButton("新建项目")
        self.new_project_button.setObjectName("sidebarButton")
        self.new_project_button.clicked.connect(self.new_project)
        sidebar_layout.addWidget(self.new_project_button)

        self.load_project_button = QPushButton("加载项目")
        self.load_project_button.setObjectName("sidebarButton")
        self.load_project_button.clicked.connect(self.load_project)
        sidebar_layout.addWidget(self.load_project_button)

        self.import_father_button = QPushButton("导入父本基因库")
        self.import_father_button.setObjectName("sidebarButton")
        self.import_father_button.clicked.connect(self.import_father_data)
        self.import_father_button.setEnabled(False)
        sidebar_layout.addWidget(self.import_father_button)

        self.import_kid_button = QPushButton("导入子本文件")
        self.import_kid_button.setObjectName("sidebarButton")
        self.import_kid_button.clicked.connect(self.import_kid_data)
        self.import_kid_button.setEnabled(False)
        sidebar_layout.addWidget(self.import_kid_button)

        self.analyze_button = QPushButton("分析")
        self.analyze_button.setObjectName("sidebarButton")
        self.analyze_button.clicked.connect(self.analyze_data)
        self.analyze_button.setEnabled(False)
        sidebar_layout.addWidget(self.analyze_button)

        self.export_button = QPushButton("导出结果")
        self.export_button.setObjectName("sidebarButton")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        sidebar_layout.addWidget(self.export_button)

        sidebar_layout.addStretch()
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        main_layout.addWidget(sidebar, 0, 0, 2, 1)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.stack, 0, 1, 1, 2)

    def new_project(self):
        dialog = NewProjectDialog()
        if dialog.exec_() == QDialog.Accepted:
            project_name = dialog.get_project_name()
            if project_name:
                project_page = ProjectPage(project_name, self)
                self.stack.addWidget(project_page)
                self.stack.setCurrentWidget(project_page)
                self.import_father_button.setEnabled(True)

    def load_project(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载项目", "", "Project Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
                project_name = project_data.get("project_name", "Unknown Project")
                project_page = ProjectPage(project_name, self, project_file=file_path)
                self.stack.addWidget(project_page)
                self.stack.setCurrentWidget(project_page)
                self.import_father_button.setEnabled(True)
                self.import_kid_button.setEnabled(True)
                self.analyze_button.setEnabled(True)
                self.export_button.setEnabled(True)

    def import_father_data(self):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, ProjectPage):
            current_page.import_father_data()

    def import_kid_data(self):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, ProjectPage):
            current_page.import_kid_data()

    def analyze_data(self):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, ProjectPage):
            current_page.analyze_data()

    def export_results(self):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, ProjectPage):
            current_page.export_results()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

