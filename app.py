#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASTQ Lane Merger Desktop Edition
Author: AI Assistant
Description: 合并同一样本不同 Lane 的 FASTQ 文件（支持 gzip）
"""

import sys
import os
import re
import gzip
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar, QLabel, QPlainTextEdit,
    QSplitter, QCheckBox, QLineEdit, QGroupBox, QStyle, QToolBar,
    QAction, QStatusBar, QMenuBar, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon, QFont, QDragEnterEvent, QDropEvent


# ================== 文件名解析器 ==================
class FastqParser:
    """
    从文件名中提取样本名、Lane、Read、是否压缩。
    支持格式如：
    Sample1_S1_L001_R1_001.fastq.gz
    1-V129-20260525_L002_R2.fastq
    """
    # 常见 Read 标志
    READ_PATTERN = re.compile(r'[._]R([12])(?:[._]|$)', re.IGNORECASE)
    # Lane 标志
    LANE_PATTERN = re.compile(r'[._]L0*([1-9]\d?)(?:[._]|$)', re.IGNORECASE)
    # 压缩后缀
    GZ_PATTERN = re.compile(r'\.(gz|g2|9z|92)$', re.IGNORECASE)

    @staticmethod
    def parse(filepath: str) -> dict:
        name = Path(filepath).name
        # 去压缩
        gz = bool(FastqParser.GZ_PATTERN.search(name))
        clean_name = FastqParser.GZ_PATTERN.sub('', name)

        # 再去 .fastq / .fq / .fasta
        clean_name = re.sub(r'\.(fastq|fq|fasta)(?:\.\d+)?$', '', clean_name, flags=re.IGNORECASE)

        # 提取 Lane
        lane_match = FastqParser.LANE_PATTERN.search(clean_name)
        lane = int(lane_match.group(1)) if lane_match else None

        # 提取 Read
        read_match = FastqParser.READ_PATTERN.search(clean_name)
        read = int(read_match.group(1)) if read_match else None

        # 提取样本名：去掉 Lane 和 Read 部分及末尾分隔符
        sample = clean_name
        if lane_match:
            sample = sample.replace(lane_match.group(0), '')
        if read_match:
            sample = sample.replace(read_match.group(0), '')
        # 清理多余符号
        sample = re.sub(r'[_\s]+', '_', sample).strip('_')
        if not sample:
            sample = clean_name.strip('_')

        # 猜测扩展名
        if gz:
            ext = '.fastq.gz'
        elif clean_name.endswith(('.fastq', '.fq')):
            ext = '.fastq'
        else:
            ext = '.fastq'  # 默认

        return {
            'sample': sample,
            'lane': lane,
            'read': read,
            'is_gz': gz,
            'extension': ext,
            'original_name': name,
            'full_path': str(Path(filepath).absolute())
        }


# ================== 合并工作线程 ==================
class MergeWorker(QThread):
    """在后台线程中执行合并，避免界面卡死"""
    progress = pyqtSignal(int, str)   # 进度百分比，状态文本
    finished = pyqtSignal(str, bool, str)  # 组标识，成功，消息
    log = pyqtSignal(str)             # 日志消息

    def __init__(self, groups: Dict[str, List[dict]], output_dir: str, use_cat_mode: bool = True):
        super().__init__()
        self.groups = groups
        self.output_dir = output_dir
        self.use_cat_mode = use_cat_mode
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def run(self):
        total_groups = len(self.groups)
        completed = 0
        for key, files in self.groups.items():
            if self._is_canceled:
                self.log.emit("⚠️ 合并已被用户取消")
                break

            sample, read_str = key.split('||')
            read_label = f"_R{read_str}" if read_str.isdigit() else ""
            is_gz = files[0]['is_gz']
            ext = '.fastq.gz' if is_gz else '.fastq'
            merged_name = f"{sample}{read_label}_merged{ext}"
            merged_path = os.path.join(self.output_dir, merged_name)

            self.progress.emit(int(completed / total_groups * 100),
                               f"正在合并 {sample} ({read_label or 'SE'})...")
            self.log.emit(f"开始合并组: {sample} ({len(files)} 个 Lane)")

            try:
                total_size = 0
                # 1. 合并文件
                if is_gz and self.use_cat_mode:
                    # 直接拼接 gzip 流（与 cat 命令等效）
                    with open(merged_path, 'wb') as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            with open(f_info['full_path'], 'rb') as in_f:
                                shutil.copyfileobj(in_f, out_f)
                            total_size += os.path.getsize(f_info['full_path'])
                elif is_gz and not self.use_cat_mode:
                    # 安全模式：解压后逐行拼接再压缩
                    with gzip.open(merged_path, 'wt', compresslevel=6) as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            with gzip.open(f_info['full_path'], 'rt') as in_f:
                                for line in in_f:
                                    out_f.write(line)
                else:
                    # 纯文本 FASTQ
                    with open(merged_path, 'w', encoding='utf-8') as out_f:
                        for f_info in sorted(files, key=lambda x: x.get('lane', 0) or 0):
                            with open(f_info['full_path'], 'r', encoding='utf-8') as in_f:
                                shutil.copyfileobj(in_f, out_f)

                # 2. 计算 MD5
                md5 = self._calculate_md5(merged_path)

                self.finished.emit(key, True,
                                   f"{sample}{read_label} 合并完成，大小 {os.path.getsize(merged_path)/1024/1024:.1f} MB, MD5: {md5}")
                self.log.emit(f"✅ {merged_name} 合并成功，MD5: {md5}")

            except Exception as e:
                self.finished.emit(key, False, f"合并失败: {str(e)}")
                self.log.emit(f"❌ 合并 {sample} 失败: {str(e)}")

            completed += 1
            if completed == total_groups:
                self.progress.emit(100, "全部合并完成")
                self.log.emit("🎉 所有分组合并完毕")

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
        self.setWindowTitle("FASTQ Lane Merger (Desktop)")
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)

        # 数据状态
        self.file_records = []          # 解析后的文件信息列表
        self.groups = {}                # 分组结果
        self.output_dir = os.path.expanduser("~/FASTQ_Merged")  # 默认输出文件夹
        self.merge_worker = None

        # 创建 UI
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_widget()
        self._create_status_bar()

        # 加载设置
        self._load_settings()

    # ---------- 菜单栏 ----------
    def _create_actions(self):
        self.add_files_action = QAction("添加文件...", self)
        self.add_files_action.triggered.connect(self.select_files)
        self.clear_action = QAction("清空列表", self)
        self.clear_action.triggered.connect(self.clear_files)
        self.set_output_action = QAction("设置输出目录...", self)
        self.set_output_action.triggered.connect(self.set_output_directory)
        self.quit_action = QAction("退出", self)
        self.quit_action.triggered.connect(self.close)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction(self.add_files_action)
        file_menu.addAction(self.set_output_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        help_menu = menu_bar.addMenu("帮助")
        help_action = QAction("关于", self)
        help_action.triggered.connect(lambda: QMessageBox.about(self, "关于", "FASTQ Lane Merger v1.0\n流式合并，支持 cat 模式"))
        help_menu.addAction(help_action)

    # ---------- 工具栏 ----------
    def _create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        self.addToolBar(toolbar)
        toolbar.addAction(self.add_files_action)
        toolbar.addAction(self.clear_action)
        toolbar.addSeparator()
        toolbar.addAction(self.set_output_action)
        toolbar.addSeparator()
        # 输出目录显示
        self.output_label = QLabel(f"输出: {self.output_dir}")
        toolbar.addWidget(self.output_label)

    # ---------- 中央区域 ----------
    def _create_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 上半部分：左右分栏（文件表格 + 合并面板）
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：文件表格
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.file_table = QTableWidget(0, 7)
        self.file_table.setHorizontalHeaderLabels(["文件名", "样本", "Lane", "Read", "大小(MB)", "类型", "状态"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.setEditTriggers(QTableWidget.NoEditTriggers)  # 后期可改为可编辑
        left_layout.addWidget(self.file_table)

        # 右侧：合并控制
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignTop)

        # 分组显示区域
        self.groups_groupbox = QGroupBox("待合并分组")
        groups_layout = QVBoxLayout()
        self.groups_layout_widget = QVBoxLayout()  # 动态添加到 groups_layout
        groups_layout.addLayout(self.groups_layout_widget)
        groups_layout.addStretch()
        self.groups_groupbox.setLayout(groups_layout)
        right_layout.addWidget(self.groups_groupbox)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.merge_all_btn = QPushButton("▶ 合并所有分组")
        self.merge_all_btn.clicked.connect(self.merge_all_groups)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_merge)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.merge_all_btn)
        btn_layout.addWidget(self.cancel_btn)
        right_layout.addLayout(btn_layout)

        # 选项
        options_layout = QHBoxLayout()
        self.cat_mode_checkbox = QCheckBox("使用 cat 模式直接拼接 gz (推荐)")
        self.cat_mode_checkbox.setChecked(True)
        options_layout.addWidget(self.cat_mode_checkbox)
        right_layout.addLayout(options_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])
        main_layout.addWidget(splitter)

        # 下半部分：进度条和日志
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)
        main_layout.addWidget(self.log_text)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 拖拽文件到此窗口或点击添加文件")

    # ---------- 文件操作 ----------
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 FASTQ 文件",
            filter="FASTQ 文件 (*.fastq *.fastq.gz *.fq *.fq.gz *.fasta *.fasta.gz);;所有文件 (*)"
        )
        if files:
            self.add_files(files)

    def add_files(self, file_paths: List[str]):
        new_records = []
        for fp in file_paths:
            info = FastqParser.parse(fp)
            # 简单去重
            if not any(r['full_path'] == info['full_path'] for r in self.file_records):
                new_records.append(info)
        self.file_records.extend(new_records)
        self._rebuild_groups()
        self._refresh_all()
        self.log(f"添加了 {len(new_records)} 个文件")

    def clear_files(self):
        self.file_records.clear()
        self.groups.clear()
        self._refresh_all()
        self.log("已清空所有文件")

    def set_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_dir)
        if directory:
            self.output_dir = directory
            self.output_label.setText(f"输出: {self.output_dir}")
            self.log(f"输出目录设置为: {directory}")

    # ---------- 分组逻辑 ----------
    def _rebuild_groups(self):
        self.groups.clear()
        for rec in self.file_records:
            read_key = str(rec['read']) if rec['read'] is not None else 'SE'
            group_key = f"{rec['sample']}||{read_key}"
            if group_key not in self.groups:
                self.groups[group_key] = []
            self.groups[group_key].append(rec)
        # 每个组内按 Lane 排序
        for key in self.groups:
            self.groups[key].sort(key=lambda x: x.get('lane') or 0)

    # ---------- 刷新界面 ----------
    def _refresh_all(self):
        self._refresh_file_table()
        self._refresh_group_panel()

    def _refresh_file_table(self):
        self.file_table.setRowCount(len(self.file_records))
        for i, rec in enumerate(self.file_records):
            self.file_table.setItem(i, 0, QTableWidgetItem(rec['original_name']))
            self.file_table.setItem(i, 1, QTableWidgetItem(rec['sample']))
            self.file_table.setItem(i, 2, QTableWidgetItem(f"L{rec['lane']:03d}" if rec['lane'] else "?"))
            self.file_table.setItem(i, 3, QTableWidgetItem(f"R{rec['read']}" if rec['read'] else "SE"))
            size_mb = os.path.getsize(rec['full_path']) / 1024 / 1024
            self.file_table.setItem(i, 4, QTableWidgetItem(f"{size_mb:.1f}"))
            self.file_table.setItem(i, 5, QTableWidgetItem("gz" if rec['is_gz'] else "text"))
            self.file_table.setItem(i, 6, QTableWidgetItem("待合并"))

    def _refresh_group_panel(self):
        # 清空动态布局
        while self.groups_layout_widget.count():
            item = self.groups_layout_widget.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for key, files in self.groups.items():
            sample, read_str = key.split('||')
            read_label = f"R{read_str}" if read_str.isdigit() else "SE"
            lanes = [f"L{f['lane']:03d}" if f['lane'] else "?" for f in files]
            group_widget = QWidget()
            hl = QHBoxLayout(group_widget)
            hl.setContentsMargins(5, 2, 5, 2)
            info_label = QLabel(f"<b>{sample}</b> ({read_label})<br>"
                                f"{len(files)} Lane(s): {', '.join(lanes)}")
            info_label.setTextFormat(Qt.RichText)
            hl.addWidget(info_label, 1)
            merge_btn = QPushButton("合并此组")
            merge_btn.clicked.connect(lambda checked, k=key: self.merge_single_group(k))
            hl.addWidget(merge_btn)
            self.groups_layout_widget.addWidget(group_widget)

    # ---------- 合并触发 ----------
    def merge_single_group(self, group_key: str):
        if group_key not in self.groups:
            return
        single_group = {group_key: self.groups[group_key]}
        self._start_merge(single_group)

    def merge_all_groups(self):
        if not self.groups:
            QMessageBox.information(self, "提示", "没有待合并的分组")
            return
        self._start_merge(self.groups.copy())

    def _start_merge(self, groups_to_merge: dict):
        # 检查输出目录是否存在，不存在则创建
        os.makedirs(self.output_dir, exist_ok=True)
        self.merge_all_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self.merge_worker = MergeWorker(
            groups_to_merge,
            self.output_dir,
            self.cat_mode_checkbox.isChecked()
        )
        self.merge_worker.progress.connect(self._on_progress)
        self.merge_worker.finished.connect(self._on_finished)
        self.merge_worker.log.connect(self.log)
        self.merge_worker.start()

    def cancel_merge(self):
        if self.merge_worker and self.merge_worker.isRunning():
            self.merge_worker.cancel()
            self.log("正在取消...")

    def _on_progress(self, value: int, msg: str):
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(msg)

    def _on_finished(self, group_key: str, success: bool, msg: str):
        self.log(msg)
        # 更新文件表中对应文件的状态
        for i, rec in enumerate(self.file_records):
            sample = rec['sample']
            read_key = str(rec['read']) if rec['read'] is not None else 'SE'
            if f"{sample}||{read_key}" == group_key:
                self.file_table.setItem(i, 6, QTableWidgetItem("已合并" if success else "失败"))
        # 恢复按钮
        self.merge_all_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.merge_worker = None

    # ---------- 拖拽支持 ----------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if os.path.isfile(fp):
                files.append(fp)
        if files:
            self.add_files(files)

    # ---------- 设置持久化 ----------
    def _load_settings(self):
        settings = QSettings("FASTQMerger", "Desktop")
        self.output_dir = settings.value("output_dir", self.output_dir)
        self.output_label.setText(f"输出: {self.output_dir}")

    def closeEvent(self, event):
        settings = QSettings("FASTQMerger", "Desktop")
        settings.setValue("output_dir", self.output_dir)
        super().closeEvent(event)

    # ---------- 日志 ----------
    def log(self, message: str):
        self.log_text.appendPlainText(message)


# ================== 入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
