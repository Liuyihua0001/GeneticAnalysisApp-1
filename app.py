# -*- coding: utf-8 -*-
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
import multiprocessing
from functools import partial
import time # For progress update throttling

# --- Configuration ---
MINIMUM_MARKERS_REQUIRED = 20 # Minimum number of non-excluded markers needed to run analysis
MATCH_THRESHOLD = 1.0       # Threshold for allele size difference
EXCLUDED_MARKERS = ['DXS2506'] # List of markers to exclude (e.g., sex markers)
# Set to None to use all available valid markers, or set to an integer (e.g., 20)
# to use exactly that many markers (if available).
NUM_MARKERS_TO_USE = None

# --- Logging Setup ---
logging.basicConfig(filename='application.log', level=logging.INFO, # Changed level to INFO for more detail
                    format='%(asctime)s %(levelname)s:%(processName)s:%(message)s') # Added process name

# --- Interfaces ---
class IDataProcessor(ABC):
    @abstractmethod
    def load_data(self, file_path):
        pass

    @abstractmethod
    def clean_and_prepare_data(self, df):
        pass

    @abstractmethod
    def perform_analysis(self, father_df, kid_df, config):
        pass

    @abstractmethod
    def save_results(self, results_df, file_path):
        pass

# --- Data Cleaning ---
class DataCleaner:
    def clean_and_prepare_data(self, df):
        """Cleans and prepares the input DataFrame."""
        if df is None or df.empty:
            logging.warning("Input DataFrame is empty or None in clean_and_prepare_data.")
            return pd.DataFrame() # Return empty DataFrame

        logging.info(f"Starting cleaning for DataFrame with shape {df.shape}")
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Standardize column names (example: remove leading/trailing spaces, convert to lower case)
        df.columns = df.columns.str.strip() #.str.lower() # Optional: convert to lower case

        # Ensure required columns exist
        required_cols = ['Sample Name', 'Marker', 'Size 1', 'Size 2']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Clean and convert Size columns
        df['Size 1'] = pd.to_numeric(df['Size 1'], errors='coerce')
        df['Size 2'] = pd.to_numeric(df['Size 2'], errors='coerce')

        # Drop rows with NaN in critical columns *after* conversion
        original_rows = len(df)
        df.dropna(subset=['Sample Name', 'Marker', 'Size 1', 'Size 2'], inplace=True)
        rows_after_na = len(df)
        if original_rows > rows_after_na:
            logging.info(f"Dropped {original_rows - rows_after_na} rows due to NaN in critical columns.")

        # Drop duplicate marker entries for the same sample
        original_rows = len(df)
        df.drop_duplicates(subset=['Sample Name', 'Marker'], inplace=True)
        rows_after_duplicates = len(df)
        if original_rows > rows_after_duplicates:
            logging.info(f"Dropped {original_rows - rows_after_duplicates} duplicate marker entries.")

        logging.info(f"Finished cleaning. DataFrame shape: {df.shape}")
        if df.empty:
            logging.warning("DataFrame is empty after cleaning.")

        return df

# --- Worker function for multiprocessing ---
# Needs to be defined at the top level or be a static method to be pickleable

def analyze_single_kid(kid_info, father_pivot, all_markers, threshold, excluded_markers):
    """
    Analyzes a single kid against all fathers using vectorized operations.

    Args:
        kid_info (tuple): Contains (kid_name, kid_series). kid_series is a pandas Series
                          multi-indexed by ('Size 1'/'Size 2', Marker).
        father_pivot (pd.DataFrame): Pivoted father data (Samples x MultiIndex[Size, Marker]).
        all_markers (list): List of markers to consider in the analysis.
        threshold (float): The matching threshold.
        excluded_markers (list): List of markers to exclude.

    Returns:
        dict: Analysis results for the single kid.
    """
    kid_name, kid_series = kid_info
    logging.debug(f"Analyzing kid: {kid_name}")

    # Filter markers for this specific kid (they might not have all markers)
    # Also exclude globally excluded markers
    valid_markers_for_kid = [
        m for m in all_markers
        if m in kid_series.index.get_level_values('Marker') and m not in excluded_markers
    ]

    if not valid_markers_for_kid:
        logging.warning(f"Kid {kid_name} has no valid markers for analysis.")
        return {
            'Kid Sample Name': kid_name,
            'Matched Father Sample Name': 'No Valid Markers',
            'Match Score': 0,
            'Total Markers Used': 0,
            'Match Percentage': 0.0,
        }

    try:
        # Extract kid's data for the valid markers
        kid_s1 = kid_series.loc[('Size 1', valid_markers_for_kid)].values
        kid_s2 = kid_series.loc[('Size 2', valid_markers_for_kid)].values

        # Extract father data for the same valid markers
        # Use reindex to ensure columns match exactly, handling cases where a father might miss a marker
        father_s1 = father_pivot.loc[:, ('Size 1', valid_markers_for_kid)].reindex(columns=valid_markers_for_kid, level='Marker').values
        father_s2 = father_pivot.loc[:, ('Size 2', valid_markers_for_kid)].reindex(columns=valid_markers_for_kid, level='Marker').values

        # --- Vectorized comparison ---
        # Handle potential NaNs in father data (kid NaNs are implicitly handled by marker selection)
        # A match occurs if kid_s1 matches father_s1/s2 OR kid_s2 matches father_s1/s2

        # Match Kid Allele 1 vs Father Alleles
        match_k1_f1 = np.abs(kid_s1 - father_s1) <= threshold
        match_k1_f2 = np.abs(kid_s1 - father_s2) <= threshold
        match1 = np.logical_or(match_k1_f1, match_k1_f2)

        # Match Kid Allele 2 vs Father Alleles
        match_k2_f1 = np.abs(kid_s2 - father_s1) <= threshold
        match_k2_f2 = np.abs(kid_s2 - father_s2) <= threshold
        match2 = np.logical_or(match_k2_f1, match_k2_f2)

        # Combine: Match if either kid allele matches either father allele for a marker
        # Use np.logical_or which handles NaNs by treating NaN comparison as False
        marker_matches = np.logical_or(match1, match2)

        # Calculate scores per father (sum across markers axis)
        # Sum boolean matches (True=1, False=0)
        scores = np.nansum(marker_matches, axis=1) # nansum treats NaNs as 0 for summation

        # Find best match
        if len(scores) > 0:
            best_score_idx = np.argmax(scores)
            best_score = scores[best_score_idx]
            best_match_father_name = father_pivot.index[best_score_idx]
        else:
            # Should not happen if father_pivot is not empty, but good to handle
            best_score = 0
            best_match_father_name = "No Fathers Analyzed"

        num_markers_analyzed = len(valid_markers_for_kid)
        match_percentage = (best_score / num_markers_analyzed) * 100 if num_markers_analyzed > 0 else 0.0

        logging.debug(f"Finished analyzing kid: {kid_name}. Best match: {best_match_father_name} with score {best_score}/{num_markers_analyzed}")

        return {
            'Kid Sample Name': kid_name,
            'Matched Father Sample Name': best_match_father_name,
            'Match Score': int(best_score), # Return as integer
            'Total Markers Used': num_markers_analyzed,
            'Match Percentage': round(match_percentage, 2),
        }

    except Exception as e:
        logging.error(f"Error analyzing kid {kid_name}: {e}", exc_info=True)
        return {
            'Kid Sample Name': kid_name,
            'Matched Father Sample Name': 'Error during analysis',
            'Match Score': 0,
            'Total Markers Used': 0,
            'Match Percentage': 0.0,
            'Error': str(e)
        }


# --- Data Processing Logic ---
class DataProcessor(IDataProcessor):
    def __init__(self):
        self.cleaner = DataCleaner()
        self.match_results = None # Store final results DataFrame

    def load_data(self, file_path):
        """Loads data from an Excel file."""
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.endswith('.xlsx'):
             raise ValueError("Invalid file type. Please select an Excel file (*.xlsx)")
        try:
            logging.info(f"Loading data from: {file_path}")
            df = pd.read_excel(file_path)
            if df.empty:
                logging.warning(f"Loaded file is empty: {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading Excel file {file_path}: {e}")
            raise

    def clean_and_prepare_data(self, df):
        """Cleans data using the DataCleaner."""
        return self.cleaner.clean_and_prepare_data(df)

    def calculate_marker_sums(self, df, markers_to_use):
        """Calculates the sum of Size1 and Size2 for specified markers per sample."""
        if df is None or df.empty:
            return pd.DataFrame(columns=['Total_Size1', 'Total_Size2', 'Total_Sum'])

        logging.info("Calculating marker sums...")
        # Filter for relevant markers *before* grouping
        df_filtered = df[df['Marker'].isin(markers_to_use)].copy()

        # Calculate sums
        sums_df = df_filtered.groupby('Sample Name')[['Size 1', 'Size 2']].sum()
        sums_df['Total_Sum'] = sums_df['Size 1'] + sums_df['Size 2']
        sums_df = sums_df.rename(columns={'Size 1': 'Total_Size1', 'Size 2': 'Total_Size2'})
        logging.info("Finished calculating marker sums.")
        return sums_df

    def perform_analysis(self, father_df, kid_df, config):
        """
        Prepares data and orchestrates the analysis (parallel execution handled by AnalysisThread).

        Args:
            father_df (pd.DataFrame): Cleaned father data.
            kid_df (pd.DataFrame): Cleaned kid data.
            config (dict): Analysis configuration (threshold, excluded_markers, etc.).

        Returns:
            tuple: Contains prepared data structures needed for parallel analysis:
                   (kid_pivot, father_pivot, all_markers, markers_to_use, father_sums, kid_sums)
                   Returns (None, None, ...) on error or insufficient data.
        """
        logging.info("Starting analysis preparation...")
        threshold = config.get('threshold', MATCH_THRESHOLD)
        excluded_markers = config.get('excluded_markers', EXCLUDED_MARKERS)
        num_markers_to_use = config.get('num_markers_to_use', NUM_MARKERS_TO_USE)
        min_markers_required = config.get('min_markers_required', MINIMUM_MARKERS_REQUIRED)

        if father_df.empty or kid_df.empty:
             logging.error("Cannot perform analysis: Father or Kid DataFrame is empty after cleaning.")
             raise ValueError("Father or Kid data is empty after cleaning. Cannot proceed.")


        # --- Determine Markers to Use ---
        common_markers = set(father_df['Marker'].unique()) & set(kid_df['Marker'].unique())
        valid_markers = sorted([m for m in common_markers if m not in excluded_markers])

        if len(valid_markers) < min_markers_required:
            raise ValueError(f"Insufficient valid markers. Need at least {min_markers_required}, found {len(valid_markers)} after exclusions and checking commonality.")

        if num_markers_to_use is not None and num_markers_to_use > 0:
             if len(valid_markers) < num_markers_to_use:
                 logging.warning(f"Requested {num_markers_to_use} markers, but only {len(valid_markers)} valid markers available. Using {len(valid_markers)}.")
                 markers_to_use = valid_markers
             else:
                 # If a specific number is requested, take the first N valid ones (consider if sorting/selection logic needed)
                 markers_to_use = valid_markers[:num_markers_to_use]
                 logging.info(f"Using the first {num_markers_to_use} valid markers.")
        else:
             markers_to_use = valid_markers # Use all valid markers
             logging.info(f"Using all {len(markers_to_use)} available valid markers.")

        # --- Calculate Marker Sums ---
        father_sums = self.calculate_marker_sums(father_df, markers_to_use)
        kid_sums = self.calculate_marker_sums(kid_df, markers_to_use)

        # --- Pivot Data for Vectorized Operations ---
        logging.info("Pivoting dataframes...")
        try:
            # Keep only necessary columns and markers before pivoting
            father_df_filt = father_df[father_df['Marker'].isin(markers_to_use)][['Sample Name', 'Marker', 'Size 1', 'Size 2']]
            kid_df_filt = kid_df[kid_df['Marker'].isin(markers_to_use)][['Sample Name', 'Marker', 'Size 1', 'Size 2']]

            # Pivot - use multi-index for columns ('Size 1'/'Size 2', Marker)
            father_pivot = father_df_filt.pivot_table(index='Sample Name', columns='Marker', values=['Size 1', 'Size 2'])
            kid_pivot = kid_df_filt.pivot_table(index='Sample Name', columns='Marker', values=['Size 1', 'Size 2'])

            # Ensure consistent marker columns across both pivots, handling missing markers per sample
            all_pivot_markers = pd.Index(markers_to_use, name='Marker')
            multi_index_cols = pd.MultiIndex.from_product([['Size 1', 'Size 2'], all_pivot_markers])

            father_pivot = father_pivot.reindex(columns=multi_index_cols)
            kid_pivot = kid_pivot.reindex(columns=multi_index_cols)

            logging.info("Pivoting complete.")

        except Exception as e:
            logging.error(f"Error during pivoting: {e}", exc_info=True)
            raise ValueError(f"Failed to pivot dataframes: {e}")


        # Check if pivots are empty (might happen if filtering removed all data)
        if father_pivot.empty or kid_pivot.empty:
             logging.error("Pivoted dataframes are empty. Cannot proceed.")
             raise ValueError("Pivoted dataframes are empty, possibly due to filtering. Cannot proceed.")


        logging.info("Analysis preparation complete.")
        # Return data structures needed by the parallel worker function
        return kid_pivot, father_pivot, markers_to_use, father_sums, kid_sums


    def save_results(self, results_df, file_path):
        """Saves the results DataFrame to an Excel file."""
        if results_df is None or results_df.empty:
            raise ValueError("No results to save.")
        try:
            logging.info(f"Saving results to: {file_path}")
            results_df.to_excel(file_path, index=False)
            logging.info("Results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving results to {file_path}: {e}")
            raise


# --- Analysis Thread using Multiprocessing ---
class AnalysisThread(QThread):
    # Signals: progress (int value, str message), partial_result (single kid result dict), finished (final DataFrame)
    update_progress = pyqtSignal(int, str)
    analysis_partial_result = pyqtSignal(dict) # Emit result for each kid
    analysis_finished = pyqtSignal(pd.DataFrame) # Emit final combined DataFrame
    analysis_error = pyqtSignal(str) # Emit error messages

    def __init__(self, processor, father_df, kid_df, config):
        super().__init__()
        self.processor = processor
        self.father_df = father_df
        self.kid_df = kid_df
        self.config = config
        self._is_running = True
        self.pool = None # Multiprocessing pool

    def stop(self):
        logging.info("Attempting to stop analysis thread...")
        self._is_running = False
        if self.pool:
            logging.info("Terminating multiprocessing pool.")
            self.pool.terminate() # Forcefully stop worker processes
            self.pool.join()      # Wait for termination
        self.terminate() # Terminate the QThread itself if needed (use with caution)
        self.wait()      # Wait for QThread termination
        logging.info("Analysis thread stopped.")


    def run(self):
        results_list = []
        total_kids = 0
        processed_count = 0
        start_time = time.time()

        try:
            self.update_progress.emit(0, "正在清理数据...")
            father_df_clean = self.processor.clean_and_prepare_data(self.father_df)
            kid_df_clean = self.processor.clean_and_prepare_data(self.kid_df)
            self.update_progress.emit(5, "数据清理完成.")

            if not self._is_running: return

            self.update_progress.emit(10, "正在准备分析数据结构...")
            kid_pivot, father_pivot, markers_to_use, father_sums, kid_sums = \
                self.processor.perform_analysis(father_df_clean, kid_df_clean, self.config)
            self.update_progress.emit(20, "数据准备完成.")

            total_kids = len(kid_pivot)
            if total_kids == 0:
                raise ValueError("No kid data available for analysis after preparation.")

            # Create the partial function with fixed arguments for the worker
            # This avoids passing large father_pivot repeatedly
            worker_func = partial(analyze_single_kid,
                                  father_pivot=father_pivot,
                                  all_markers=markers_to_use,
                                  threshold=self.config.get('threshold', MATCH_THRESHOLD),
                                  excluded_markers=self.config.get('excluded_markers', EXCLUDED_MARKERS))

            # Use context manager for the pool
            num_processes = max(1, multiprocessing.cpu_count() - 1) # Leave one core free
            logging.info(f"Starting multiprocessing pool with {num_processes} workers.")
            with multiprocessing.Pool(processes=num_processes) as self.pool:

                # Use imap_unordered for potentially faster result retrieval and progress updates
                # The items in kid_pivot.iterrows() need to be pickleable
                kid_items = list(kid_pivot.iterrows()) # Convert iterator to list

                last_update_time = time.time()

                for result in self.pool.imap_unordered(worker_func, kid_items):
                    if not self._is_running:
                        logging.info("Analysis cancelled during processing.")
                        self.pool.terminate()
                        break # Exit the loop if cancelled

                    processed_count += 1

                    # --- Add Sums Data ---
                    kid_name = result.get('Kid Sample Name')
                    father_name = result.get('Matched Father Sample Name')

                    if kid_name in kid_sums.index:
                        k_sums = kid_sums.loc[kid_name]
                        result['Kid_Total_Size1'] = k_sums.get('Total_Size1', 0)
                        result['Kid_Total_Size2'] = k_sums.get('Total_Size2', 0)
                        result['Kid_Total_Sum'] = k_sums.get('Total_Sum', 0)
                    else:
                         result['Kid_Total_Size1'], result['Kid_Total_Size2'], result['Kid_Total_Sum'] = 0, 0, 0

                    if father_name and father_name in father_sums.index:
                        f_sums = father_sums.loc[father_name]
                        result['Father_Total_Size1'] = f_sums.get('Total_Size1', 0)
                        result['Father_Total_Size2'] = f_sums.get('Total_Size2', 0)
                        result['Father_Total_Sum'] = f_sums.get('Total_Sum', 0)
                    else:
                         result['Father_Total_Size1'], result['Father_Total_Size2'], result['Father_Total_Sum'] = 0, 0, 0

                    results_list.append(result)
                    self.analysis_partial_result.emit(result) # Emit result for this kid

                    # --- Update Progress ---
                    # Throttle progress updates to avoid overwhelming the GUI thread
                    current_time = time.time()
                    if current_time - last_update_time > 0.1 or processed_count == total_kids: # Update every 100ms or on last item
                        progress_percentage = 20 + int((processed_count / total_kids) * 80)
                        self.update_progress.emit(progress_percentage, f"正在分析 {processed_count}/{total_kids} ({kid_name})...")
                        last_update_time = current_time

                # Ensure pool is properly handled even if loop breaks
                if self._is_running:
                    self.pool.close() # No more tasks will be submitted
                    self.pool.join()  # Wait for all tasks to complete
                else:
                    # If cancelled, we already terminated/joined
                    pass

            self.pool = None # Clear pool reference

            if not self._is_running:
                 self.update_progress.emit(100, "分析已取消.")
                 # Emit empty DataFrame or partial results? Let's emit what we have.
                 final_results_df = pd.DataFrame(results_list)
                 self.analysis_finished.emit(final_results_df)
                 return

            # --- Finalize ---
            final_results_df = pd.DataFrame(results_list)
            # Reorder columns for better readability
            cols_order = [
                'Kid Sample Name', 'Matched Father Sample Name', 'Match Score',
                'Total Markers Used', 'Match Percentage',
                'Kid_Total_Size1', 'Kid_Total_Size2', 'Kid_Total_Sum',
                'Father_Total_Size1', 'Father_Total_Size2', 'Father_Total_Sum',
                'Error' # Keep error column if it exists
            ]
            final_results_df = final_results_df.reindex(columns=[col for col in cols_order if col in final_results_df.columns])

            self.processor.match_results = final_results_df # Store in processor

            elapsed_time = time.time() - start_time
            self.update_progress.emit(100, f"分析完成! 处理了 {processed_count}/{total_kids} 个样本，用时 {elapsed_time:.2f} 秒.")
            self.analysis_finished.emit(final_results_df) # Emit final results

        except ValueError as ve:
            logging.error(f"Value error during analysis: {ve}", exc_info=True)
            self.analysis_error.emit(f"分析错误 (数据问题): {str(ve)}")
            self.update_progress.emit(100, f"分析出错: {str(ve)}")
        except MemoryError as me:
            logging.error(f"Memory error during analysis: {me}", exc_info=True)
            self.analysis_error.emit(f"分析错误 (内存不足): {str(me)}. 请尝试使用较小的数据集.")
            self.update_progress.emit(100, f"内存不足错误: {str(me)}")
        except Exception as e:
            logging.error(f"Unexpected error in analysis thread: {e}", exc_info=True)
            self.analysis_error.emit(f"分析时发生意外错误: {str(e)}")
            self.update_progress.emit(100, f"分析出错: {str(e)}")
        finally:
             if self.pool: # Ensure pool is cleaned up in case of exception
                 self.pool.terminate()
                 self.pool.join()
             self._is_running = False # Mark thread as not running


# --- GUI Classes ---

# NewProjectDialog (No changes needed)
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

    def accept(self):
        if not self.project_name_input.text().strip():
            QMessageBox.warning(self, "输入错误", "项目名称不能为空。")
            return
        self.project_name = self.project_name_input.text().strip()
        super().accept()

    def get_project_name(self):
        return self.project_name

# ProjectPage (Modified for new analysis flow and results display)
class ProjectPage(QWidget):
    def __init__(self, project_name, main_window, project_file=None):
        super().__init__()
        self.main_window = main_window
        self.project_name = project_name
        self.processor = DataProcessor() # Each project gets its own processor instance

        self.father_file_path = None
        self.father_df = None # Store loaded, uncleaned data
        self.kid_file_path = None
        self.kid_df = None    # Store loaded, uncleaned data
        self.match_results = None # Store results DataFrame
        self.progress_dialog = None
        self.analysis_thread = None

        # Store cleaned data separately if needed for display, or clean on demand
        # self.father_df_clean = None
        # self.kid_df_clean = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15) # Reduced spacing slightly

        title_label = QLabel(f"项目: {self.project_name}")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #FFFFFF; background-color: #004080; padding: 10px; border-radius: 8px;")
        main_layout.addWidget(title_label)

        # Log/Status List Widget
        self.status_list = QListWidget()
        self.status_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # Fixed height
        self.status_list.setMaximumHeight(150) # Max height for status
        self.status_list.setStyleSheet("background-color: #f0f0f0; font-size: 12px; border: 1px solid #ccc; border-radius: 5px; padding: 5px;")
        main_layout.addWidget(self.status_list)

        # Results Table Widget
        self.result_table = QTableWidget()
        # Define expected columns - update if analysis output changes
        self.expected_columns = [
            'Kid Sample Name', 'Matched Father Sample Name', 'Match Score',
            'Total Markers Used', 'Match Percentage',
            'Kid_Total_Size1', 'Kid_Total_Size2', 'Kid_Total_Sum',
            'Father_Total_Size1', 'Father_Total_Size2', 'Father_Total_Sum',
            'Error'
        ]
        self.result_table.setColumnCount(len(self.expected_columns))
        self.result_table.setHorizontalHeaderLabels(self.expected_columns)
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border-radius: 8px;
                gridline-color: #e0e0e0; /* Lighter grid lines */
                font-size: 13px; /* Slightly smaller font */
            }
            QTableWidget::item {
                padding: 8px; /* Adjust padding */
                border-bottom: 1px solid #e0e0e0;
            }
            QHeaderView::section {
                background-color: #0059b3; /* Slightly lighter blue */
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 8px;
                border: none; /* Remove border */
                border-bottom: 1px solid #004080; /* Darker bottom border */
            }
            QTableWidget::horizontalHeader {
                 border-bottom: 2px solid #004080;
            }
            QTableWidget::verticalHeader {
                 border-right: 1px solid #e0e0e0;
                 background-color: #f8f8f8;
                 width: 50px; /* Width for row numbers */
            }
        """)
        self.result_table.setAlternatingRowColors(True) # Improve readability
        self.result_table.verticalHeader().setVisible(True) # Show row numbers
        self.result_table.horizontalHeader().setStretchLastSection(True) # Stretch last column
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
        main_layout.addWidget(self.result_table)

        self.setLayout(main_layout)

        if project_file:
            self.load_project(project_file)
        else:
            self.add_status_message(f"项目 '{self.project_name}' 已创建.")

    def add_status_message(self, message):
        """Adds a message to the status list widget."""
        logging.info(f"Status Update ({self.project_name}): {message}")
        self.status_list.addItem(message)
        self.status_list.scrollToBottom() # Keep latest message visible

    def save_project(self):
        """Saves project state (file paths, results) to a JSON file."""
        # Note: Saving the entire DataFrame in JSON can be large.
        # Consider saving results to a separate Excel/CSV and only storing the path.
        project_data = {
            "project_name": self.project_name,
            "father_file_path": self.father_file_path,
            "kid_file_path": self.kid_file_path,
            # Convert DataFrame to dict for JSON serialization (can be large)
            "match_results": self.match_results.to_dict(orient='records') if self.match_results is not None else None
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "保存项目", f"{self.project_name}.json", "Project Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, indent=4)
                self.add_status_message(f"项目已保存到: {file_path}")
                QMessageBox.information(self, "保存成功", f"项目已成功保存到: {file_path}")
            except Exception as e:
                self.log_and_display_error("保存项目失败", e)

    def load_project(self, project_file):
        """Loads project state from a JSON file."""
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            self.project_name = project_data.get("project_name", "Unnamed Project")
            self.findChild(QLabel, "title").setText(f"项目: {self.project_name}") # Update title label
            self.add_status_message(f"正在加载项目: {self.project_name} 从 {project_file}")

            self.father_file_path = project_data.get("father_file_path")
            self.kid_file_path = project_data.get("kid_file_path")

            # Load data if paths exist
            if self.father_file_path and os.path.exists(self.father_file_path):
                try:
                    self.father_df = self.processor.load_data(self.father_file_path)
                    self.add_status_message(f"父本数据已加载: {os.path.basename(self.father_file_path)} ({len(self.father_df)} 行)")
                except Exception as e:
                    self.log_and_display_error(f"加载父本文件失败 ({self.father_file_path})", e)
                    self.father_file_path = None # Reset path if load fails
            else:
                 self.add_status_message("未找到父本文件路径或文件不存在。")


            if self.kid_file_path and os.path.exists(self.kid_file_path):
                try:
                    self.kid_df = self.processor.load_data(self.kid_file_path)
                    self.add_status_message(f"子本数据已加载: {os.path.basename(self.kid_file_path)} ({len(self.kid_df)} 行)")
                    # Optionally display raw kid data here if needed
                    # self.display_dataframe_in_table(self.kid_df) # Example call
                except Exception as e:
                    self.log_and_display_error(f"加载子本文件失败 ({self.kid_file_path})", e)
                    self.kid_file_path = None # Reset path if load fails
            else:
                 self.add_status_message("未找到子本文件路径或文件不存在。")


            # Load results if they exist
            if project_data.get("match_results"):
                try:
                    # Recreate DataFrame from dict
                    self.match_results = pd.DataFrame(project_data["match_results"])
                    self.display_dataframe_in_table(self.match_results)
                    self.add_status_message(f"已加载之前的分析结果 ({len(self.match_results)} 行).")
                    self.main_window.export_button.setEnabled(True) # Enable export if results loaded
                except Exception as e:
                    self.log_and_display_error("加载分析结果失败", e)
                    self.match_results = None

            # Update button states based on loaded data
            self.main_window.update_button_states()

        except Exception as e:
            self.log_and_display_error(f"加载项目文件失败: {project_file}", e)

    def import_father_data(self):
        """Imports father data from an Excel file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择父本基因库文件", "", "Excel Files (*.xlsx)")
        if not file_path:
            return # User cancelled

        try:
            self.father_df = self.processor.load_data(file_path) # Use processor's load method
            self.father_file_path = file_path
            self.add_status_message(f"父本基因库文件已导入: {os.path.basename(file_path)} ({len(self.father_df)} 行)")
            self.match_results = None # Clear previous results
            self.display_dataframe_in_table(None) # Clear table
            self.main_window.update_button_states() # Update button enables
        except (FileNotFoundError, ValueError, Exception) as e:
            self.log_and_display_error("导入父本基因库文件时发生错误", e)
            self.father_df = None
            self.father_file_path = None
            self.main_window.update_button_states()

    def import_kid_data(self):
        """Imports kid data from an Excel file."""
        if not self.father_file_path:
             QMessageBox.warning(self, "导入错误", "请先导入父本基因库文件。")
             return

        file_path, _ = QFileDialog.getOpenFileName(self, "选择子本文件", "", "Excel Files (*.xlsx)")
        if not file_path:
            return # User cancelled

        try:
            self.kid_df = self.processor.load_data(file_path) # Use processor's load method
            self.kid_file_path = file_path
            self.add_status_message(f"子本文件已导入: {os.path.basename(file_path)} ({len(self.kid_df)} 行)")
            self.match_results = None # Clear previous results
            self.display_dataframe_in_table(None) # Clear table
            # Optionally display raw kid data here upon import
            # self.display_dataframe_in_table(self.kid_df)
            self.main_window.update_button_states() # Update button enables
        except (FileNotFoundError, ValueError, Exception) as e:
            self.log_and_display_error("导入子本文件时发生错误", e)
            self.kid_df = None
            self.kid_file_path = None
            self.main_window.update_button_states()

    def display_dataframe_in_table(self, df):
        """Displays a pandas DataFrame in the QTableWidget."""
        self.result_table.clearContents() # Clear existing data

        if df is None or df.empty:
            self.result_table.setRowCount(0)
            # Use expected columns if df is None (e.g., after clearing)
            cols = self.expected_columns if df is None else []
            self.result_table.setColumnCount(len(cols))
            self.result_table.setHorizontalHeaderLabels(cols)
            return

        # Use actual columns from the dataframe being displayed
        actual_columns = df.columns.tolist()
        self.result_table.setColumnCount(len(actual_columns))
        self.result_table.setHorizontalHeaderLabels(actual_columns)
        self.result_table.setRowCount(len(df))

        for row_idx, row in enumerate(df.itertuples(index=False)):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value) if pd.notna(value) else "") # Handle NaN
                # Optional: Add specific formatting based on column name or value
                if isinstance(value, (int, float)):
                     item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Example: Color 'Match Percentage' based on value
                if actual_columns[col_idx] == 'Match Percentage' and isinstance(value, (int, float)):
                    if value >= 95.0:
                        item.setForeground(QColor('darkgreen'))
                    elif value < 80.0:
                         item.setForeground(QColor('red'))
                self.result_table.setItem(row_idx, col_idx, item)

        self.result_table.resizeColumnsToContents() # Adjust column widths

    def analyze_data(self):
        """Starts the data analysis process in a separate thread."""
        if self.father_df is None or self.kid_df is None:
            QMessageBox.warning(self, "错误", "请先导入父本基因库和子本文件。")
            return

        if self.analysis_thread and self.analysis_thread.isRunning():
             QMessageBox.warning(self, "正在进行", "分析已经在运行中。")
             return

        # Clear previous results display
        self.match_results = None
        self.display_dataframe_in_table(None)
        self.add_status_message("开始分析...")
        self.main_window.set_buttons_for_analysis(False) # Disable buttons

        # --- Create and show Progress Dialog ---
        self.progress_dialog = QProgressDialog("正在准备分析...", "取消", 0, 100, self.main_window) # Parent to main window
        self.progress_dialog.setWindowTitle("分析进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal) # Block main window
        self.progress_dialog.setAutoClose(False) # We will close it manually
        self.progress_dialog.setAutoReset(False) # We will reset manually
        self.progress_dialog.canceled.connect(self.cancel_analysis) # Connect cancel signal
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        # --- Prepare Config ---
        # TODO: Get these from UI elements if needed in the future
        analysis_config = {
            'threshold': MATCH_THRESHOLD,
            'excluded_markers': EXCLUDED_MARKERS,
            'num_markers_to_use': NUM_MARKERS_TO_USE,
            'min_markers_required': MINIMUM_MARKERS_REQUIRED
        }

        # --- Start Analysis Thread ---
        # Pass copies of dataframes to the thread to avoid issues if main GUI modifies them
        self.analysis_thread = AnalysisThread(self.processor, self.father_df.copy(), self.kid_df.copy(), analysis_config)
        self.analysis_thread.update_progress.connect(self.update_progress)
        self.analysis_thread.analysis_partial_result.connect(self.handle_partial_result) # Connect partial result
        self.analysis_thread.analysis_finished.connect(self.on_analysis_finished)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)
        self.analysis_thread.finished.connect(self.on_thread_actually_finished) # Signal when thread object is done
        self.analysis_thread.start()

    def update_progress(self, value, message):
        """Updates the progress dialog."""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
            # Optionally add major steps to the status list as well
            if value in [0, 5, 10, 20, 100] or "完成" in message or "错误" in message or "取消" in message:
                 self.add_status_message(message)

    def handle_partial_result(self, result_dict):
         """Handles a partial result (one kid analyzed) - could update table live."""
         # For now, we just log it. Updating the table live frequently can be slow.
         # If needed, could append row to table here.
         logging.debug(f"Received partial result for: {result_dict.get('Kid Sample Name')}")
         pass # Currently handled by final display

    def on_analysis_finished(self, final_results_df):
        """Called when the analysis thread emits the final results DataFrame."""
        self.add_status_message("分析线程报告完成。正在处理最终结果...")
        self.match_results = final_results_df # Store the final results

        if self.match_results is not None and not self.match_results.empty:
             self.display_dataframe_in_table(self.match_results)
             self.add_status_message(f"分析完成，显示 {len(self.match_results)} 条结果。")
        elif self.match_results is not None: # Empty dataframe
             self.add_status_message("分析完成，但没有生成结果。")
        else: # Should not happen if signal emits DataFrame, but check anyway
             self.add_status_message("分析完成，但未收到有效结果。")

        # Close progress dialog safely
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Re-enable buttons
        self.main_window.set_buttons_for_analysis(True)
        self.main_window.update_button_states() # Ensure export is enabled if results exist


    def on_analysis_error(self, error_message):
         """Handles errors reported by the analysis thread."""
         self.add_status_message(f"分析错误: {error_message}")
         self.log_and_display_error("分析过程中发生错误", Exception(error_message)) # Show dialog

         if self.progress_dialog:
             self.progress_dialog.close()
             self.progress_dialog = None

         self.main_window.set_buttons_for_analysis(True) # Re-enable buttons on error
         self.main_window.update_button_states()


    def on_thread_actually_finished(self):
        """Called when the QThread object itself has finished execution."""
        logging.info("Analysis QThread finished.")
        self.analysis_thread = None # Clear the thread reference
        # Final button state check
        self.main_window.set_buttons_for_analysis(True)
        self.main_window.update_button_states()


    def cancel_analysis(self):
        """Stops the running analysis thread."""
        self.add_status_message("正在尝试取消分析...")
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop() # Call the custom stop method
            # Progress dialog update/closure is handled by on_analysis_finished or on_analysis_error
            self.add_status_message("取消信号已发送。")
        else:
             self.add_status_message("没有正在运行的分析可以取消。")
        # Close the progress dialog immediately upon clicking cancel
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        # Re-enable buttons immediately
        self.main_window.set_buttons_for_analysis(True)
        self.main_window.update_button_states()


    def export_results(self):
        """Exports the analysis results to an Excel file."""
        if self.match_results is None or self.match_results.empty:
            QMessageBox.warning(self, "错误", "没有可导出的结果，请先进行分析。")
            return

        try:
            save_path = f"{self.project_name}_results.xlsx"
            file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", save_path, "Excel Files (*.xlsx)")
            if file_path:
                self.processor.save_results(self.match_results, file_path)
                self.add_status_message(f"结果已导出到: {file_path}")
                QMessageBox.information(self, "导出成功", f"结果已成功导出到: {file_path}")
        except Exception as e:
            self.log_and_display_error("导出结果时发生错误", e)

    def log_and_display_error(self, message, exception):
        """Logs an error and displays a detailed error message box."""
        logging.error(f"{message}: {exception}", exc_info=True) # Log with traceback
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("发生错误")
        error_dialog.setText(f"{message}.")
        error_dialog.setInformativeText("详细信息:")
        # Provide more context in detailed text
        error_dialog.setDetailedText(f"项目: {self.project_name}\n错误类型: {type(exception).__name__}\n\n{str(exception)}\n\n请查看 application.log 文件获取更多信息。")
        error_dialog.exec_()


# MainWindow (Modified to manage button states and project pages)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("基因亲缘关系分析工具 v2.0") # Updated Title
        self.setGeometry(100, 100, 1300, 850) # Slightly larger window
        self.setWindowIcon(QIcon()) # Add an icon later if available

        # --- Global Stylesheet ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #eef2f9; /* Lighter background */
            }
            QWidget#centralWidget {
                 background-color: #eef2f9;
            }
            /* Sidebar Styling */
            QWidget#sidebar {
                background-color: #2c3e50; /* Darker sidebar */
                border-radius: 0px; /* Sharp corners for sidebar */
            }
            QPushButton#sidebarButton {
                background-color: transparent;
                color: #ecf0f1; /* Light text color */
                font-size: 14px;
                padding: 12px 15px; /* More padding */
                margin: 1px 0; /* Minimal margin */
                text-align: left;
                border: none;
                border-radius: 4px; /* Subtle rounding */
                qproperty-iconSize: 18px; /* Icon size */
            }
            QPushButton#sidebarButton:hover {
                background-color: #34495e; /* Slightly lighter on hover */
            }
            QPushButton#sidebarButton:disabled {
                color: #7f8c8d; /* Disabled color */
                background-color: transparent;
            }
            QPushButton#sidebarButton:checked { /* Style for active project? */
                 background-color: #1abc9c; /* Example active color */
                 font-weight: bold;
            }

            /* General Widget Styling */
            QLabel { font-size: 14px; color: #333; }
            QLineEdit, QListWidget, QTableWidget {
                border: 1px solid #bdc3c7; /* Standard border */
                border-radius: 4px;
                padding: 5px;
                background-color: #ffffff;
            }
            QListWidget { background-color: #fdfefe; }
            QPushButton {
                 background-color: #3498db; /* Standard button blue */
                 color: white;
                 border: none;
                 padding: 8px 15px;
                 border-radius: 4px;
                 font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; }
            QMessageBox { font-size: 14px; }

            /* Progress Bar Styling */
            QProgressDialog {
                 font-size: 14px;
            }
            QProgressBar {
                 border: 1px solid #bdc3c7;
                 border-radius: 4px;
                 text-align: center;
                 background-color: white;
            }
            QProgressBar::chunk {
                 background-color: #3498db;
                 border-radius: 4px;
            }
        """)

        # --- Central Widget and Layout ---
        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget")
        main_layout = QHBoxLayout(self.central_widget) # Use QHBoxLayout for side-by-side
        main_layout.setContentsMargins(0, 0, 0, 0) # No margins for main layout
        main_layout.setSpacing(0) # No spacing between sidebar and content
        self.setCentralWidget(self.central_widget)

        # --- Sidebar ---
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 10, 5, 10) # Padding inside sidebar
        sidebar_layout.setSpacing(5) # Spacing between buttons
        sidebar.setFixedWidth(220) # Fixed width for sidebar

        # Add icons (replace with actual paths or QStyle icons if preferred)
        # icon_new = QIcon.fromTheme("document-new", QIcon("path/to/new.png")) # Example
        # self.new_project_button = QPushButton(icon_new, " 新建项目")
        self.new_project_button = QPushButton(" 新建项目") # Add space for icon alignment
        self.new_project_button.setObjectName("sidebarButton")
        self.new_project_button.clicked.connect(self.new_project)
        sidebar_layout.addWidget(self.new_project_button)

        self.load_project_button = QPushButton(" 加载项目")
        self.load_project_button.setObjectName("sidebarButton")
        self.load_project_button.clicked.connect(self.load_project)
        sidebar_layout.addWidget(self.load_project_button)

        sidebar_layout.addSpacing(20) # Separator

        self.import_father_button = QPushButton(" 导入父本库")
        self.import_father_button.setObjectName("sidebarButton")
        self.import_father_button.clicked.connect(self.import_father_data)
        sidebar_layout.addWidget(self.import_father_button)

        self.import_kid_button = QPushButton(" 导入子本文件")
        self.import_kid_button.setObjectName("sidebarButton")
        self.import_kid_button.clicked.connect(self.import_kid_data)
        sidebar_layout.addWidget(self.import_kid_button)

        self.analyze_button = QPushButton(" 开始分析")
        self.analyze_button.setObjectName("sidebarButton")
        self.analyze_button.clicked.connect(self.analyze_data)
        sidebar_layout.addWidget(self.analyze_button)

        self.export_button = QPushButton(" 导出结果")
        self.export_button.setObjectName("sidebarButton")
        self.export_button.clicked.connect(self.export_results)
        sidebar_layout.addWidget(self.export_button)

        sidebar_layout.addStretch() # Pushes buttons to the top
        main_layout.addWidget(sidebar) # Add sidebar to the left

        # --- Content Area (Stacked Widget) ---
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Add a default welcome/placeholder widget
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_label = QLabel("欢迎使用基因亲缘关系分析工具\n\n请新建或加载项目开始。")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 18px; color: #555;")
        welcome_layout.addWidget(welcome_label)
        self.stack.addWidget(welcome_widget)

        main_layout.addWidget(self.stack) # Add content stack to the right

        self.update_button_states() # Initial button state

    def get_current_project_page(self):
        """Gets the currently active ProjectPage instance, or None."""
        widget = self.stack.currentWidget()
        if isinstance(widget, ProjectPage):
            return widget
        return None

    def update_button_states(self):
        """Updates the enabled state of sidebar buttons based on current project state."""
        page = self.get_current_project_page()

        can_import_father = page is not None
        can_import_kid = page is not None and page.father_df is not None
        can_analyze = page is not None and page.father_df is not None and page.kid_df is not None
        can_export = page is not None and page.match_results is not None and not page.match_results.empty

        self.import_father_button.setEnabled(can_import_father)
        self.import_kid_button.setEnabled(can_import_kid)
        self.analyze_button.setEnabled(can_analyze)
        self.export_button.setEnabled(can_export)

        # Disable buttons if analysis is running
        if page and page.analysis_thread and page.analysis_thread.isRunning():
             self.set_buttons_for_analysis(False)


    def set_buttons_for_analysis(self, enabled):
         """Enable/disable buttons typically blocked during analysis."""
         self.new_project_button.setEnabled(enabled)
         self.load_project_button.setEnabled(enabled)
         # Don't override logic from update_button_states for these,
         # just ensure they are disabled if enabled=False
         if not enabled:
             self.import_father_button.setEnabled(False)
             self.import_kid_button.setEnabled(False)
             self.analyze_button.setEnabled(False)
             self.export_button.setEnabled(False)
         else:
             # If enabling, let update_button_states determine the final state
             self.update_button_states()


    def new_project(self):
        """Handles the 'New Project' action."""
        dialog = NewProjectDialog()
        if dialog.exec_() == QDialog.Accepted:
            project_name = dialog.get_project_name()
            if project_name:
                # Check if project name already exists? (Optional)
                project_page = ProjectPage(project_name, self)
                self.stack.addWidget(project_page)
                self.stack.setCurrentWidget(project_page)
                self.update_button_states()

    def load_project(self):
        """Handles the 'Load Project' action."""
        file_path, _ = QFileDialog.getOpenFileName(self, "加载项目", "", "Project Files (*.json)")
        if file_path:
            try:
                # Peek into the file to get the name without fully loading data yet
                with open(file_path, 'r', encoding='utf-8') as f:
                    project_data_peek = json.load(f)
                    project_name = project_data_peek.get("project_name", "Unnamed Project")

                # Create the page, which will handle the actual loading
                project_page = ProjectPage(project_name, self, project_file=file_path)
                self.stack.addWidget(project_page)
                self.stack.setCurrentWidget(project_page)
                # Button states are updated within project_page.load_project -> self.update_button_states()
            except Exception as e:
                 QMessageBox.critical(self, "加载错误", f"无法加载项目文件: {file_path}\n错误: {e}")
                 logging.error(f"Failed to load project file {file_path}: {e}", exc_info=True)
                 self.update_button_states() # Reset buttons state


    # --- Action Forwarding ---
    def import_father_data(self):
        page = self.get_current_project_page()
        if page: page.import_father_data()

    def import_kid_data(self):
        page = self.get_current_project_page()
        if page: page.import_kid_data()

    def analyze_data(self):
        page = self.get_current_project_page()
        if page: page.analyze_data()

    def export_results(self):
        page = self.get_current_project_page()
        if page: page.export_results()

    def closeEvent(self, event):
        """Handle closing the application, ensuring threads are stopped."""
        page = self.get_current_project_page()
        if page and page.analysis_thread and page.analysis_thread.isRunning():
            reply = QMessageBox.question(self, '退出确认',
                                       "分析正在进行中。确定要退出吗？分析将会被取消。",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logging.info("User confirmed exit during analysis. Stopping thread.")
                page.cancel_analysis() # Attempt graceful shutdown
                # Give thread a moment to stop? Or just exit? Let's just exit.
                event.accept()
            else:
                logging.info("User cancelled exit during analysis.")
                event.ignore()
        else:
            event.accept()


# --- Main Execution ---
if __name__ == '__main__':
    # !!! IMPORTANT for multiprocessing on Windows/macOS !!!
    # Freeze support is necessary if packaging the application (e.g., with PyInstaller)
    # It ensures worker processes can re-import the necessary code.
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)

    # Apply a style if desired (e.g., 'Fusion')
    # app.setStyle('Fusion')

    # Optional: Set a custom palette for more theme control
    # palette = QPalette()
    # ... set palette colors ...
    # app.setPalette(palette)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
