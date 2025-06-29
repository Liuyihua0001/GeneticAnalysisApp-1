import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class RenameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("批量重命名工具（支持Excel映射）")
        self.root.geometry("600x400")
        self.mapping = {}

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, font=('Arial', 10))
        style.configure("TLabel", font=('Arial', 10))

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="1. 选择Excel映射表 (A列barcode, B列新前缀)").pack(anchor='w', padx=10, pady=5)
        ttk.Button(self.root, text="选择Excel文件", command=self.select_excel).pack(padx=10, pady=5)

        ttk.Label(self.root, text="2. 选择包含fastq文件的文件夹").pack(anchor='w', padx=10, pady=5)
        ttk.Button(self.root, text="选择文件夹", command=self.select_folder).pack(padx=10, pady=5)

        ttk.Label(self.root, text="3. 执行重命名").pack(anchor='w', padx=10, pady=5)
        ttk.Button(self.root, text="开始重命名", command=self.rename_files).pack(padx=10, pady=5)

        self.log = tk.Text(self.root, height=12, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)

    def select_excel(self):
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file:
            try:
                df = pd.read_excel(file, header=None)
                self.mapping = dict(zip(df[0], df[1]))
                self.log.insert(tk.END, f"✅ 已加载 {len(self.mapping)} 条映射规则。\n")
            except Exception as e:
                messagebox.showerror("错误", f"Excel读取失败: {str(e)}")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            self.log.insert(tk.END, f"📁 已选择文件夹: {folder}\n")

    def rename_files(self):
        if not hasattr(self, 'folder_path') or not self.mapping:
            messagebox.showwarning("缺失信息", "请先选择Excel和文件夹")
            return

        renamed, skipped, unmapped = [], [], []
        for file in os.listdir(self.folder_path):
            if file.startswith("SQK-NBD114-96_") and file.endswith((".fastq", ".fastq.gz")):
                match = file.split("SQK-NBD114-96_")[-1]
                barcode = match.split(".fastq")[0].replace(".gz", "")
                suffix = file[len(f"SQK-NBD114-96_{barcode}"):]
                if barcode in self.mapping:
                    new_name = f"{self.mapping[barcode]}_{barcode}{suffix}"
                    src = os.path.join(self.folder_path, file)
                    dst = os.path.join(self.folder_path, new_name)
                    try:
                        os.rename(src, dst)
                        renamed.append((file, new_name))
                    except Exception as e:
                        skipped.append((file, str(e)))
                else:
                    unmapped.append(file)

        self.log.insert(tk.END, f"✅ 成功重命名 {len(renamed)} 个文件。\n")
        if skipped:
            self.log.insert(tk.END, f"⚠️ 跳过 {len(skipped)} 个文件（重命名失败）\n")
        if unmapped:
            self.log.insert(tk.END, f"❌ 未在映射表中找到 {len(unmapped)} 个文件：\n  - " + "\n  - ".join(unmapped) + "\n")
        self.log.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = RenameApp(root)
    root.mainloop()
