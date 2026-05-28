#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import shutil
import threading
import time


class FastqMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FASTQ 专业合并工具")
        self.root.geometry("600x450")
        self.root.resizable(False, False)

        # 存储输入文件列表
        self.input_files = []
        self.output_file = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        # 1. 输入文件区域
        frame_input = tk.LabelFrame(self.root, text=" 1. 添加需要合并的文件 (FASTQ / FASTQ.GZ) ", padx=10, pady=10)
        frame_input.pack(fill="both", expand=True, padx=15, pady=10)

        self.listbox = tk.Listbox(frame_input, selectmode=tk.EXTENDED, height=8)
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame_input, orient="vertical")
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(frame_input)
        btn_frame.pack(side="right", fill="y", padx=5)

        tk.Button(btn_frame, text="添加文件", command=self.add_files, width=10).pack(pady=5)
        tk.Button(btn_frame, text="移除选中", command=self.remove_files, width=10).pack(pady=5)
        tk.Button(btn_frame, text="清空列表", command=self.clear_files, width=10).pack(pady=5)

        # 2. 输出文件区域
        frame_output = tk.LabelFrame(self.root, text=" 2. 设置保存路径 ", padx=10, pady=10)
        frame_output.pack(fill="x", padx=15, pady=5)

        tk.Entry(frame_output, textvariable=self.output_file, state="readonly", width=50).pack(side="left", fill="x",
                                                                                               expand=True)
        tk.Button(frame_output, text="浏览...", command=self.select_output).pack(side="right", padx=5)

        # 3. 操作与进度区域
        frame_action = tk.Frame(self.root, pady=10)
        frame_action.pack(fill="x", padx=15)

        self.btn_merge = tk.Button(frame_action, text="🚀 开始合并", command=self.start_merge_thread, bg="#4CAF50",
                                   fg="white", font=("Arial", 12, "bold"))
        self.btn_merge.pack(pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame_action, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

        self.status_label = tk.Label(frame_action, text="状态: 等待操作...", fg="gray")
        self.status_label.pack()

    # --- UI 交互逻辑 (已修复 macOS 文件选择器 Bug) ---
    def add_files(self):
        # 修复点：移除了复杂的复合扩展名，改为 Mac 系统兼容的格式
        files = filedialog.askopenfilenames(
            title="选择 FASTQ 文件",
            filetypes=[("All Files", "*.*"), ("FASTQ", "*.fastq"), ("GZIP", "*.gz")]
        )
        for f in files:
            if f not in self.input_files:
                self.input_files.append(f)
                self.listbox.insert(tk.END, os.path.basename(f))

    def remove_files(self):
        selected_indices = self.listbox.curselection()
        for i in reversed(selected_indices):
            self.listbox.delete(i)
            del self.input_files[i]

    def clear_files(self):
        self.listbox.delete(0, tk.END)
        self.input_files.clear()

    def select_output(self):
        # 修复点：移除了 defaultextension=".fastq.gz"，使用更安全的 filetypes
        file_path = filedialog.asksaveasfilename(
            title="保存合并后的文件",
            filetypes=[("All Files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)

    # --- 核心多线程与合并逻辑 ---
    def start_merge_thread(self):
        if len(self.input_files) < 2:
            messagebox.showwarning("警告", "请至少添加 2 个需要合并的文件！")
            return
        if not self.output_file.get():
            messagebox.showwarning("警告", "请选择合并后的保存路径！")
            return

        self.btn_merge.config(state="disabled")
        self.progress_var.set(0)
        self.status_label.config(text="状态: 正在初始化合并...", fg="blue")

        threading.Thread(target=self._merge_worker, daemon=True).start()

    def _merge_worker(self):
        out_path = self.output_file.get()
        total_files = len(self.input_files)
        start_time = time.time()

        try:
            with open(out_path, 'wb') as outfile:
                for idx, filepath in enumerate(self.input_files):
                    file_name = os.path.basename(filepath)
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

                    self.root.after(0, self._update_status,
                                    f"状态: 正在处理 ({idx + 1}/{total_files}) - {file_name} ({file_size_mb:.1f} MB)",
                                    "blue")

                    with open(filepath, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)

                    progress = ((idx + 1) / total_files) * 100
                    self.root.after(0, self.progress_var.set, progress)

            time_taken = time.time() - start_time
            self.root.after(0, self._merge_success, time_taken)

        except Exception as e:
            self.root.after(0, self._merge_error, str(e))

    def _update_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def _merge_success(self, time_taken):
        self.status_label.config(text=f"状态: 合并完成！耗时 {time_taken:.1f} 秒", fg="green")
        self.btn_merge.config(state="normal")
        messagebox.showinfo("成功", f"文件合并成功！\n耗时: {time_taken:.1f} 秒\n保存至:\n{self.output_file.get()}")

    def _merge_error(self, error_msg):
        self.status_label.config(text="状态: 合并失败！", fg="red")
        self.btn_merge.config(state="normal")
        messagebox.showerror("错误", f"合并过程中发生错误:\n{error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FastqMergerApp(root)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
