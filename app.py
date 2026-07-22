#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASTQ Lane Merger Desktop Edition
完整改进版：翻译完善、强制合并预览、线程安全关闭、文件夹拖拽、异常容错。
"""

import sys
import os
import re
import gzip
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar, QLabel, QPlainTextEdit,
    QSplitter, QCheckBox, QGroupBox, QToolBar, QAction, QStatusBar,
    QScrollArea, QSpinBox, QComboBox, QDialog, QInputDialog, QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QColor, QPixmap

# ================== 多语言字典 ==================
TRANSLATIONS = {
    "zh": {
        "title": "FASTQ Lane 合并工具专业版",
        "add_files": "添加文件...",
        "clear": "清空列表",
        "set_out": "设置输出目录...",
        "manual_merge": "🛠️ 强制合并选中项",
        "quit": "退出",
        "menu_file": "文件",
        "menu_lang": "语言 (Language)",
        "menu_help": "帮助",
        "about": "关于",
        "expected_lanes": "预期 Lane 数:",
        "cat_mode": "使用 cat 模式直接拼接 gz (推荐)",
        "merge_btn": "▶ 合并勾选分组",
        "cancel_btn": "取消",
        "pending_groups": "待合并分组 (请人工确认)",
        "col_name": "文件名",
        "col_sample": "样本",
        "col_lane": "Lane",
        "col_read": "Read",
        "col_size": "大小(MB)",
        "col_type": "类型",
        "col_status": "状态",
        "status_ready": "就绪 | 拖拽文件/文件夹到此窗口",
        "pe": "双端(PE)",
        "se": "单端(SE)",
        "err_no_select": "没有勾选任何需要合并的分组",
        "log_add": "添加了 {} 个文件",
        "log_clear": "已清空所有文件",
        "out_prefix": "输出: ",
        "check_ok": "✅ 正常",
        "check_err_lane": "❌ Lane数量不符",
        "check_err_empty": "❌ 包含空文件",
        "check_err_size": "⚠️ R1/R2体积差异过大",
        "warn_anomalies_title": "深度检查异常确认",
        "warn_anomalies_msg": "以下勾选的分组存在异常:\n\n{}\n\n您确定要强行合并这些存在风险的数据吗？",
        "manual_select_err": "请在左侧表格中选中至少 1 个文件进行强制合并 (按住 Ctrl/Cmd 多选)。",
        "manual_out_name": "请输入强制合并后的输出文件名 (例: merged_manual.fastq.gz):",
        "manual_auth_msg": "这是一次强制合并操作，将忽略所有规则。\n请输入 YES 授权确认执行:",
        "manual_cancel_log": "❌ 强制合并已被取消或授权不通过。",
        "merge_cancel_log": "⚠️ 因存在风险，您已取消合并操作。",
        "col_status_wait": "等待中",
        "col_status_merged": "已合并",
        "col_status_fail": "失败",
        "merge_in_progress": "合并任务正在进行中",
        "merge_in_progress_msg": "有合并任务正在运行，是否取消并退出？",
        "file_skip_error": "跳过无效文件 {} (错误: {})",
    },
    "en": {
        "title": "FASTQ Lane Merger Pro",
        "add_files": "Add Files...",
        "clear": "Clear List",
        "set_out": "Set Output Dir...",
        "manual_merge": "🛠️ Force Merge Selected",
        "quit": "Quit",
        "menu_file": "File",
        "menu_lang": "Language",
        "menu_help": "Help",
        "about": "About",
        "expected_lanes": "Expected Lanes:",
        "cat_mode": "Use 'cat' mode for gz",
        "merge_btn": "▶ Merge Checked",
        "cancel_btn": "Cancel",
        "pending_groups": "Pending Groups (Please manually confirm)",
        "col_name": "Filename",
        "col_sample": "Sample",
        "col_lane": "Lane",
        "col_read": "Read",
        "col_size": "Size(MB)",
        "col_type": "Type",
        "col_status": "Status",
        "status_ready": "Ready | Drop files/folders here",
        "pe": "Paired-End(PE)",
        "se": "Single-End(SE)",
        "err_no_select": "No groups selected for merging.",
        "log_add": "Added {} files",
        "log_clear": "Cleared all files",
        "out_prefix": "Output: ",
        "check_ok": "✅ OK",
        "check_err_lane": "❌ Lane Mismatch",
        "check_err_empty": "❌ Empty File Detected",
        "check_err_size": "⚠️ R1/R2 Size Mismatch",
        "warn_anomalies_title": "Anomaly Confirmation",
        "warn_anomalies_msg": "The following selected groups have anomalies:\n\n{}\n\nAre you sure you want to FORCE merge them?",
        "manual_select_err": "Please select at least 1 file from the left table (Ctrl/Cmd + click).",
        "manual_out_name": "Enter output filename (e.g., merged.fastq.gz):",
        "manual_auth_msg": "This is a forced operation bypassing all rules.\nType YES to authorize:",
        "manual_cancel_log": "❌ Forced merge cancelled or authorization failed.",
        "merge_cancel_log": "⚠️ Merge cancelled due to risk.",
        "col_status_wait": "Waiting",
        "col_status_merged": "Merged",
        "col_status_fail": "Failed",
        "merge_in_progress": "Merge in progress",
        "merge_in_progress_msg": "A merge task is running. Cancel it?",
        "file_skip_error": "Skipping invalid file {} (Error: {})",
    },
    "ru": {
        "title": "Слияние FASTQ Lane Pro",
        "add_files": "Добавить файлы...",
        "clear": "Очистить список",
        "set_out": "Установить вывод...",
        "manual_merge": "🛠️ Принудительное слияние",
        "quit": "Выход",
        "menu_file": "Файл",
        "menu_lang": "Язык",
        "menu_help": "Помощь",
        "about": "О программе",
        "expected_lanes": "Ожидаемые Lane:",
        "cat_mode": "Использовать cat для gz",
        "merge_btn": "▶ Объединить выбранные",
        "cancel_btn": "Отмена",
        "pending_groups": "Группы для слияния (Подтвердите вручную)",
        "col_name": "Имя файла",
        "col_sample": "Образец",
        "col_lane": "Lane",
        "col_read": "Read",
        "col_size": "Размер(МБ)",
        "col_type": "Тип",
        "col_status": "Статус",
        "status_ready": "Готов | Перетащите файлы/папки сюда",
        "pe": "Парные(PE)",
        "se": "Одиночные(SE)",
        "err_no_select": "Не выбраны группы.",
        "log_add": "Добавлено {} файлов",
        "log_clear": "Все файлы удалены",
        "out_prefix": "Вывод: ",
        "check_ok": "✅ Норма",
        "check_err_lane": "❌ Несовпадение Lane",
        "check_err_empty": "❌ Пустой файл",
        "check_err_size": "⚠️ Несовпадение размеров",
        "warn_anomalies_title": "Подтверждение аномалии",
        "warn_anomalies_msg": "Следующие группы имеют аномалии:\n\n{}\n\nВы уверены, что хотите принудительно объединить их?",
        "manual_select_err": "Выберите не менее 1 файла.",
        "manual_out_name": "Имя выходного файла:",
        "manual_auth_msg": "Введите YES для подтверждения:",
        "manual_cancel_log": "❌ Принудительное слияние отменено.",
        "merge_cancel_log": "⚠️ Слияние отменено из-за риска.",
        "col_status_wait": "Ожидание",
        "col_status_merged": "Слито",
        "col_status_fail": "Ошибка",
        "merge_in_progress": "Идет слияние",
        "merge_in_progress_msg": "Задача слияния выполняется. Отменить?",
        "file_skip_error": "Пропущен неверный файл {} (Ошибка: {})",
    }
}


# ================== 文件名解析器 ==================
class FastqParser:
    READ_PATTERN = re.compile(r'[._]R([12])(?:[._]|$)', re.IGNORECASE)
    LANE_PATTERN = re.compile(r'[._]L0*([1-9]\d?)(?:[._]|$)', re.IGNORECASE)
    GZ_PATTERN = re.compile(r'\.(gz|g2|9z|92)$', re.IGNORECASE)

    @staticmethod
    def parse(filepath: str) -> dict:
        name = Path(filepath).name
        try:
            size = os.path.getsize(filepath)
        except OSError as e:
            raise ValueError(f"Cannot access file: {filepath}") from e

        gz = bool(FastqParser.GZ_PATTERN.search(name))
        clean_name = FastqParser.GZ_PATTERN.sub('', name)
        clean_name = re.sub(r'\.(fastq|fq|fasta)(?:\.\d+)?$', '', clean_name, flags=re.IGNORECASE)

        lane_match = FastqParser.LANE_PATTERN.search(clean_name)
        lane = int(lane_match.group(1)) if lane_match else None

        read_match = FastqParser.READ_PATTERN.search(clean_name)
        read = int(read_match.group(1)) if read_match else None

        sample = clean_name
        if lane_match: sample = sample.replace(lane_match.group(0), '')
        if read_match: sample = sample.replace(read_match.group(0), '')
        sample = re.sub(r'[_\s]+', '_', sample).strip('_')
        if not sample: sample = clean_name.strip('_')

        if gz:
            ext = '.fastq.gz'
        elif clean_name.endswith(('.fastq', '.fq')):
            ext = '.fastq'
        else:
            ext = '.fastq'

        return {
            'sample': sample,
            'lane': lane,
            'read': read,
            'is_gz': gz,
            'extension': ext,
            'size': size,
            'original_name': name,
            'full_path': str(Path(filepath).absolute())
        }


# ================== 合并工作线程 ==================
class MergeWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict, bool, str)   # task, success, message
    log = pyqtSignal(str)

    def __init__(self, tasks: List[dict], output_dir: str, use_cat_mode: bool):
        super().__init__()
        self.tasks = tasks
        self.output_dir = output_dir
        self.use_cat_mode = use_cat_mode
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def run(self):
        total_tasks = len(self.tasks)
        completed = 0
        # 计算总字节数用于进度（可选）
        total_bytes = sum(f['size'] for task in self.tasks for f in task['files'])
        processed_bytes = 0

        for task in self.tasks:
            if self._is_canceled:
                self.log.emit("⚠️ Canceled")
                break

            is_manual = task.get('is_manual', False)
            if is_manual:
                merged_name = task['out_name']
                files = task['files']
            else:
                sample = task['sample']
                read_num = task['read'] if task['read'] else 1
                files = task['files']
                ext = '.fastq.gz' if files[0]['is_gz'] else '.fastq'
                # 合并后 lane 编号使用 merged 标识而非 L001
                merged_name = f"merged_{sample}_S1_Lmerged_R{read_num}_001{ext}"

            merged_path = os.path.join(self.output_dir, merged_name)
            self.progress.emit(int(processed_bytes / total_bytes * 100) if total_bytes > 0 else 0,
                               f"Merging {merged_name}...")

            try:
                all_gz = all(f['is_gz'] for f in files)
                if all_gz and self.use_cat_mode:
                    with open(merged_path, 'wb') as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            with open(f_info['full_path'], 'rb') as in_f:
                                shutil.copyfileobj(in_f, out_f)
                elif all_gz and not self.use_cat_mode:
                    with gzip.open(merged_path, 'wt', compresslevel=6) as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            with gzip.open(f_info['full_path'], 'rt') as in_f:
                                for line in in_f:
                                    out_f.write(line)
                else:
                    with open(merged_path, 'w', encoding='utf-8') as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            if f_info['is_gz']:
                                with gzip.open(f_info['full_path'], 'rt') as in_f:
                                    shutil.copyfileobj(in_f, out_f)
                            else:
                                with open(f_info['full_path'], 'r', encoding='utf-8') as in_f:
                                    shutil.copyfileobj(in_f, out_f)

                md5 = self._calculate_md5(merged_path)
                size_mb = os.path.getsize(merged_path) / 1024 / 1024

                self.finished.emit(task, True, f"✅ {merged_name} (Size: {size_mb:.1f}MB, MD5: {md5})")
                processed_bytes += sum(f['size'] for f in files)
            except Exception as e:
                # 清理不完整输出
                if os.path.exists(merged_path):
                    os.remove(merged_path)
                self.finished.emit(task, False, f"❌ Fail: {merged_name} -> {str(e)}")
                # 即使失败，已处理的字节数也应计入，避免进度停滞
                processed_bytes += sum(f['size'] for f in files)

            completed += 1

        if not self._is_canceled:
            self.progress.emit(100, "All Done")

    @staticmethod
    def _calculate_md5(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


# ================== 主窗口 ==================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.lang = "zh"
        self.setMinimumSize(1200, 750)
        self.setAcceptDrops(True)

        self.file_records = []
        self.samples_dict = defaultdict(lambda: {1: [], 2: []})
        self.group_checkboxes = {}
        self.group_warnings = {}
        self.output_dir = os.path.expanduser("~/FASTQ_Merged")
        self.merge_worker = None

        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_widget()
        self._create_status_bar()

        self._load_settings()
        self.update_ui_text()

    def tr(self, key: str) -> str:
        return TRANSLATIONS.get(self.lang, TRANSLATIONS["zh"]).get(key, key)

    def change_language(self, lang_code: str):
        self.lang = lang_code
        self.update_ui_text()

    def update_ui_text(self):
        self.setWindowTitle(self.tr("title"))
        self.add_files_action.setText(self.tr("add_files"))
        self.clear_action.setText(self.tr("clear"))
        self.set_output_action.setText(self.tr("set_out"))
        self.manual_merge_action.setText(self.tr("manual_merge"))
        self.quit_action.setText(self.tr("quit"))

        self.menu_file.setTitle(self.tr("menu_file"))
        self.menu_help.setTitle(self.tr("menu_help"))

        self.output_label.setText(f"{self.tr('out_prefix')}{self.output_dir}")
        self.groups_groupbox.setTitle(self.tr("pending_groups"))

        self.lane_label.setText(self.tr("expected_lanes"))
        self.cat_mode_checkbox.setText(self.tr("cat_mode"))
        self.merge_all_btn.setText(self.tr("merge_btn"))
        self.cancel_btn.setText(self.tr("cancel_btn"))
        self.manual_btn.setText(self.tr("manual_merge"))

        headers = [self.tr("col_name"), self.tr("col_sample"), self.tr("col_lane"),
                   self.tr("col_read"), self.tr("col_size"), self.tr("col_type"), self.tr("col_status")]
        self.file_table.setHorizontalHeaderLabels(headers)

        if not self.file_records:
            self.status_bar.showMessage(self.tr("status_ready"))

        self._refresh_group_panel()
        self._refresh_file_table()

    def _create_actions(self):
        self.add_files_action = QAction(self)
        self.add_files_action.triggered.connect(self.select_files)
        self.clear_action = QAction(self)
        self.clear_action.triggered.connect(self.clear_files)
        self.set_output_action = QAction(self)
        self.set_output_action.triggered.connect(self.set_output_directory)
        self.manual_merge_action = QAction(self)
        self.manual_merge_action.triggered.connect(self.execute_manual_merge)
        self.quit_action = QAction(self)
        self.quit_action.triggered.connect(self.close)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        self.menu_file = menu_bar.addMenu("File")
        self.menu_file.addAction(self.add_files_action)
        self.menu_file.addAction(self.set_output_action)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.manual_merge_action)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.clear_action)
        self.menu_file.addAction(self.quit_action)

        self.menu_help = menu_bar.addMenu("Help")
        about_act = QAction(self.tr("about"), self)
        about_act.triggered.connect(lambda: QMessageBox.about(self, self.tr("about"),
                                                              "FASTQ Lane Merger Pro\nDeep Inspection & Manual Override Features included."))
        self.menu_help.addAction(about_act)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(self.add_files_action)
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.set_output_action)
        toolbar.addSeparator()

        self.manual_btn = QPushButton()
        self.manual_btn.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold; padding: 5px;")
        self.manual_btn.clicked.connect(self.execute_manual_merge)
        toolbar.addWidget(self.manual_btn)
        toolbar.addSeparator()

        lang_combo = QComboBox()
        lang_combo.addItems(["中文 (ZH)", "English (EN)", "Русский (RU)"])
        lang_combo.currentIndexChanged.connect(lambda idx: self.change_language(["zh", "en", "ru"][idx]))
        toolbar.addWidget(QLabel(" 🌐 "))
        toolbar.addWidget(lang_combo)
        toolbar.addSeparator()

        self.output_label = QLabel()
        toolbar.addWidget(self.output_label)

    def _create_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.file_table = QTableWidget(0, 7)
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.ExtendedSelection)
        left_layout.addWidget(self.file_table)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        filter_layout = QHBoxLayout()
        self.lane_label = QLabel()
        self.lane_spin = QSpinBox()
        self.lane_spin.setRange(1, 16)
        self.lane_spin.setValue(4)
        self.lane_spin.valueChanged.connect(self._refresh_group_panel)
        filter_layout.addWidget(self.lane_label)
        filter_layout.addWidget(self.lane_spin)
        filter_layout.addStretch()
        right_layout.addLayout(filter_layout)

        self.groups_groupbox = QGroupBox()
        groupbox_layout = QVBoxLayout(self.groups_groupbox)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_widget)
        groupbox_layout.addWidget(self.scroll_area)
        right_layout.addWidget(self.groups_groupbox)

        options_layout = QVBoxLayout()
        self.cat_mode_checkbox = QCheckBox()
        self.cat_mode_checkbox.setChecked(True)
        options_layout.addWidget(self.cat_mode_checkbox)

        btn_layout = QHBoxLayout()
        self.merge_all_btn = QPushButton()
        self.merge_all_btn.setMinimumHeight(40)
        self.merge_all_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.merge_all_btn.clicked.connect(self.merge_selected_groups)
        self.cancel_btn = QPushButton()
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.clicked.connect(self.cancel_merge)
        self.cancel_btn.setEnabled(False)

        btn_layout.addWidget(self.merge_all_btn)
        btn_layout.addWidget(self.cancel_btn)
        options_layout.addLayout(btn_layout)
        right_layout.addLayout(options_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 400])
        main_layout.addWidget(splitter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        self.log_text.setFixedHeight(120)
        main_layout.addWidget(self.log_text)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, self.tr("add_files"),
                                                filter="FASTQ (*.fastq *.fastq.gz *.fq *.fq.gz *.fasta *.fasta.gz);;All (*)")
        if files:
            self.add_files(files)

    def add_files(self, file_paths: List[str]):
        new_records = []
        for fp in file_paths:
            try:
                info = FastqParser.parse(fp)
            except ValueError as e:
                self.log(self.tr("file_skip_error").format(fp, e))
                continue
            if not any(r['full_path'] == info['full_path'] for r in self.file_records):
                info['status'] = 'wait'
                new_records.append(info)

        self.file_records.extend(new_records)
        self._rebuild_groups()
        self._refresh_file_table()
        self._refresh_group_panel()
        self.log(self.tr("log_add").format(len(new_records)))

    def clear_files(self):
        self.file_records.clear()
        self.samples_dict.clear()
        self._refresh_file_table()
        self._refresh_group_panel()
        self.log(self.tr("log_clear"))

    def set_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, self.tr("set_out"), self.output_dir)
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"{self.tr('out_prefix')}{self.output_dir}")

    def execute_manual_merge(self):
        selected_rows = sorted(set(item.row() for item in self.file_table.selectedItems()))
        if not selected_rows:
            QMessageBox.warning(self, "Warning", self.tr("manual_select_err"))
            return

        files_to_merge = [self.file_records[r] for r in selected_rows]

        # 预览选中的文件
        preview = "\n".join(
            f"{i+1}. {f['original_name']} ({f['size']/1024/1024:.1f} MB)"
            for i, f in enumerate(files_to_merge)
        )
        info_msg = f"Selected {len(files_to_merge)} file(s):\n{preview}\n\nProceed to forced merge?"
        reply = QMessageBox.question(self, "Confirm Forced Merge", info_msg,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        out_name, ok = QInputDialog.getText(self, "Manual Merge", self.tr("manual_out_name"),
                                            text="merged_manual.fastq.gz")
        if not ok or not out_name.strip():
            return

        auth_text, ok = QInputDialog.getText(self, "Authorization Required", self.tr("manual_auth_msg"))
        if not ok or auth_text.strip().upper() != "YES":
            self.log(self.tr("manual_cancel_log"))
            return

        tasks = [{
            'is_manual': True,
            'out_name': out_name.strip(),
            'files': files_to_merge
        }]
        self.start_merge_worker(tasks)

    def _rebuild_groups(self):
        self.samples_dict.clear()
        for rec in self.file_records:
            sample = rec['sample']
            read_num = rec['read'] if rec['read'] else 1
            self.samples_dict[sample][read_num].append(rec)

    def _refresh_file_table(self):
        self.file_table.setRowCount(len(self.file_records))
        for i, rec in enumerate(self.file_records):
            self.file_table.setItem(i, 0, QTableWidgetItem(rec['original_name']))
            self.file_table.setItem(i, 1, QTableWidgetItem(rec['sample']))
            self.file_table.setItem(i, 2, QTableWidgetItem(f"L{rec['lane']:03d}" if rec['lane'] else "?"))
            self.file_table.setItem(i, 3, QTableWidgetItem(f"R{rec['read']}" if rec['read'] else "SE"))
            size_mb = rec['size'] / 1024 / 1024
            self.file_table.setItem(i, 4, QTableWidgetItem(f"{size_mb:.1f}"))
            self.file_table.setItem(i, 5, QTableWidgetItem("gz" if rec['is_gz'] else "text"))

            status_map = {
                "wait": self.tr("col_status_wait"),
                "merged": self.tr("col_status_merged"),
                "fail": self.tr("col_status_fail")
            }
            self.file_table.setItem(i, 6, QTableWidgetItem(status_map.get(rec.get('status', 'wait'))))

    def _refresh_group_panel(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.group_checkboxes.clear()
        self.group_warnings.clear()
        expected = self.lane_spin.value()

        for sample, reads in self.samples_dict.items():
            r1_files = reads.get(1, [])
            r2_files = reads.get(2, [])

            is_pe = bool(r1_files and r2_files)
            mode_text = self.tr("pe") if is_pe else self.tr("se")

            valid = True
            status_text = self.tr("check_ok")
            detail_text = ""

            has_empty = any(f['size'] == 0 for f in r1_files + r2_files)
            if has_empty:
                valid = False
                status_text = self.tr("check_err_empty")
            else:
                if is_pe:
                    if len(r1_files) != expected or len(r2_files) != expected:
                        valid = False
                        status_text = self.tr("check_err_lane")
                    else:
                        size_r1 = sum(f['size'] for f in r1_files)
                        size_r2 = sum(f['size'] for f in r2_files)
                        if size_r1 > 0 and size_r2 > 0:
                            diff_ratio = abs(size_r1 - size_r2) / max(size_r1, size_r2)
                            if diff_ratio > 0.2:
                                valid = False
                                status_text = self.tr("check_err_size")
                    detail_text = f"R1: {len(r1_files)}L, R2: {len(r2_files)}L"
                else:
                    act_len = len(r1_files) if r1_files else len(r2_files)
                    if act_len != expected:
                        valid = False
                        status_text = self.tr("check_err_lane")
                    lbl = "R1" if r1_files else "R2"
                    detail_text = f"{lbl}: {act_len}L"

            cb = QCheckBox(f"{sample} | {mode_text} | {detail_text} | {status_text}")
            if valid:
                cb.setChecked(True)
                cb.setStyleSheet("color: #006600; font-weight: bold;")
            else:
                cb.setChecked(False)
                cb.setStyleSheet("color: #cc0000;")
                self.group_warnings[sample] = f"[{sample}] {status_text} ({detail_text})"

            self.scroll_layout.addWidget(cb)
            self.group_checkboxes[sample] = cb

    def merge_selected_groups(self):
        tasks = []
        warnings_found = []

        for sample, cb in self.group_checkboxes.items():
            if cb.isChecked():
                if sample in self.group_warnings:
                    warnings_found.append(self.group_warnings[sample])

                reads = self.samples_dict[sample]
                if reads[1]: tasks.append({"sample": sample, "read": 1, "files": reads[1]})
                if reads[2]: tasks.append({"sample": sample, "read": 2, "files": reads[2]})

        if not tasks:
            QMessageBox.warning(self, "Warning", self.tr("err_no_select"))
            return

        if warnings_found:
            msg = self.tr("warn_anomalies_msg").format("\n".join(warnings_found))
            reply = QMessageBox.warning(self, self.tr("warn_anomalies_title"), msg, QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.log(self.tr("merge_cancel_log"))
                return

        self.start_merge_worker(tasks)

    def start_merge_worker(self, tasks):
        os.makedirs(self.output_dir, exist_ok=True)
        self.merge_all_btn.setEnabled(False)
        self.manual_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self.merge_worker = MergeWorker(tasks, self.output_dir, self.cat_mode_checkbox.isChecked())
        self.merge_worker.progress.connect(self._on_progress)
        self.merge_worker.finished.connect(self._on_finished)
        self.merge_worker.log.connect(self.log)
        self.merge_worker.start()

    def cancel_merge(self):
        if self.merge_worker and self.merge_worker.isRunning():
            self.merge_worker.cancel()

    def _on_progress(self, value: int, msg: str):
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(msg)

    def _on_finished(self, task: dict, success: bool, msg: str):
        self.log(msg)
        # 更新文件状态
        if task.get('is_manual'):
            # 手动合并的文件
            file_paths = {f['full_path'] for f in task['files']}
            for rec in self.file_records:
                if rec['full_path'] in file_paths:
                    rec['status'] = 'merged' if success else 'fail'
        else:
            sample = task['sample']
            read_num = task['read'] if task['read'] else 1
            for rec in self.file_records:
                r_num = rec['read'] if rec['read'] else 1
                if rec['sample'] == sample and r_num == read_num:
                    rec['status'] = 'merged' if success else 'fail'

        self._refresh_file_table()

        if self.merge_worker and self.merge_worker.isFinished():
            self.merge_all_btn.setEnabled(True)
            self.manual_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.merge_worker = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        paths.append(os.path.join(root, f))
            elif os.path.isfile(path):
                paths.append(path)

        fastq_exts = ('.fastq', '.fastq.gz', '.fq', '.fq.gz', '.fasta', '.fasta.gz')
        fastq_files = [p for p in paths if any(p.lower().endswith(ext) for ext in fastq_exts)]
        if fastq_files:
            self.add_files(fastq_files)

    def _load_settings(self):
        settings = QSettings("FASTQMerger", "DesktopProMax")
        self.output_dir = settings.value("output_dir", self.output_dir)

    def closeEvent(self, event):
        if self.merge_worker and self.merge_worker.isRunning():
            reply = QMessageBox.question(self, self.tr("merge_in_progress"),
                                         self.tr("merge_in_progress_msg"),
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.merge_worker.cancel()
                self.merge_worker.wait(3000)
            else:
                event.ignore()
                return
        settings = QSettings("FASTQMerger", "DesktopProMax")
        settings.setValue("output_dir", self.output_dir)
        super().closeEvent(event)

    def log(self, message: str):
        self.log_text.appendPlainText(message)


# ================== 入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    splash_pix = QPixmap(400, 200)
    splash_pix.fill(QColor("#2196F3"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Loading...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)

    sys.exit(app.exec_())
