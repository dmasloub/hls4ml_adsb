# src/utils/hls_utils.py

import os
import re
from typing import Dict, Optional

import pandas as pd

from src.utils.logger import Logger


class HLSUtils:
    """
    A utility class to handle extraction of FPGA resource utilization metrics from HLS synthesis reports.
    """

    def __init__(self):
        """
        Initializes the HLSUtils class with a logger.
        """
        self.logger = Logger.get_logger(__name__)

    @staticmethod
    def _read_report(report_path: str) -> str:
        """
        Reads the content of the HLS synthesis report.

        Args:
            report_path (str): Path to the HLS synthesis report.

        Returns:
            str: Content of the report.

        Raises:
            FileNotFoundError: If the report file does not exist.
            IOError: If the report file cannot be read.
        """
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found at {report_path}.")

        try:
            with open(report_path, 'r') as f:
                report_content = f.read()
            return report_content
        except Exception as e:
            raise IOError(f"Failed to read report file: {e}")

    @staticmethod
    def _extract_metric(pattern: str, content: str, metric_name: str, is_percentage: bool = False) -> Optional[int]:
        """
        Extracts a single metric from the report content using a regex pattern.

        Args:
            pattern (str): Regex pattern to search for the metric.
            content (str): Content of the report.
            metric_name (str): Name of the metric being extracted.
            is_percentage (bool, optional): Indicates if the metric is a percentage. Defaults to False.

        Returns:
            Optional[int]: Extracted metric value or None if not found.
        """
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            try:
                value_str = match.group(1).replace(',', '').replace('%', '')
                value = int(float(value_str))
                return value
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_table(report_path: str, table_header: str) -> Optional[pd.DataFrame]:
        """
        Extracts a table from the HLS report based on a header and converts it to a pandas DataFrame.

        Args:
            report_path (str): Path to the HLS synthesis report.
            table_header (str): The header line that precedes the table.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the table data or None if not found.
        """
        logger = Logger.get_logger(__name__)
        if not os.path.exists(report_path):
            logger.error(f"Report file not found at {report_path}.")
            raise FileNotFoundError(f"Report file not found at {report_path}.")

        with open(report_path, 'r') as f:
            lines = f.readlines()

        table_start = None
        for idx, line in enumerate(lines):
            if table_header in line:
                table_start = idx + 1  # Start after the header
                break

        if table_start is None:
            logger.warning(f"Table with header '{table_header}' not found in the report.")
            return None

        # Extract table lines until an empty line or a line that does not start with '|'
        table_lines = []
        for line in lines[table_start:]:
            if not line.strip().startswith('|'):
                break
            table_lines.append(line.strip())

        if not table_lines:
            logger.warning(f"No table data found after header '{table_header}'.")
            return None

        # Convert table lines to a list of lists
        table_data = []
        for line in table_lines:
            # Split the line by '|' and strip whitespace
            columns = [col.strip() for col in line.split('|')[1:-1]]
            table_data.append(columns)

        # The first line after header is assumed to be the separator, so skip it
        if len(table_data) >= 2:
            headers = table_data[0]
            data_rows = table_data[1:]
            df = pd.DataFrame(data_rows, columns=headers)
            logger.debug(f"Extracted table with headers: {headers}")
            return df
        else:
            logger.warning("Insufficient table data to extract headers and rows.")
            return None

    def extract_utilization(self, report_path: str) -> Dict[str, Dict[str, int]]:
        """
        Extracts FPGA resource utilization metrics from the HLS synthesis report.

        Args:
            report_path (str): Path to the HLS synthesis report.

        Returns:
            Dict[str, Dict[str, int]]: Dictionary containing resource utilization metrics.
        """
        try:
            content = self._read_report(report_path)
        except Exception as e:
            self.logger.error(f"Error reading report: {e}", exc_info=True)
            return {}

        lines = content.split('\n')
        utilization = {}

        # Locate '== Utilization Estimates' section
        util_section_start = None
        for idx, line in enumerate(lines):
            if '== Utilization Estimates' in line:
                util_section_start = idx
                break

        if util_section_start is None:
            self.logger.error("Utilization Estimates section not found in the report.")
            return {}

        # Locate '* Summary:'
        summary_start = None
        for idx in range(util_section_start, len(lines)):
            if '* Summary:' in lines[idx]:
                summary_start = idx
                break

        if summary_start is None:
            self.logger.error("Summary section not found under Utilization Estimates.")
            return {}

        # Locate the table starting from summary_start
        table_start = None
        for idx in range(summary_start, len(lines)):
            if lines[idx].startswith('+'):
                table_start = idx
                break

        if table_start is None:
            self.logger.error("Could not find the start of the summary table.")
            return {}

        # Collect the table lines
        table_lines = []
        idx = table_start
        while idx < len(lines):
            line = lines[idx]
            if line.startswith('+'):
                table_lines.append(line)
                idx += 1
            elif line.startswith('|'):
                table_lines.append(line)
                idx += 1
            else:
                # Reached the end of the table
                break

        if not table_lines:
            self.logger.error("No table data found in the summary section.")
            return {}

        # Remove separator lines (those that start with '+')
        data_lines = [line for line in table_lines if line.startswith('|')]

        # Now process the data lines
        parsed_rows = []
        for line in data_lines:
            # Split the line by '|' and strip spaces
            columns = [col.strip() for col in line.strip().split('|')]
            # Ignore empty columns resulting from split
            columns = [col for col in columns if col]
            if columns:
                parsed_rows.append(columns)

        # Now create DataFrame
        try:
            df = pd.DataFrame(parsed_rows[1:], columns=parsed_rows[0])  # First row is header
            # Now extract the required rows
            total_row = df[df['Name'] == 'Total'].iloc[0]
            available_row = df[df['Name'] == 'Available'].iloc[0]
            utilization_row = df[df['Name'] == 'Utilization (%)'].iloc[0]
            # Create a dictionary of resource utilization
            resources = ['BRAM_18K', 'DSP48E', 'FF', 'LUT', 'URAM']
            utilization = {}
            for res in resources:
                total = int(total_row[res].replace(',', '').replace('-', '0'))
                available = int(available_row[res].replace(',', '').replace('-', '0'))
                utilization_pct = float(utilization_row[res].replace(',', '').replace('-', '0'))
                utilization[res] = {
                    'Total': total,
                    'Available': available,
                    'Utilization (%)': utilization_pct
                }
            self.logger.info(f"Utilization: {utilization}")
            return utilization
        except Exception as e:
            self.logger.error(f"Failed to parse utilization table: {e}", exc_info=True)
            return {}

    def extract_utilization_table(self, report_path: str, table_header: str) -> Optional[pd.DataFrame]:
        """
        Extracts a table from the HLS report based on a header and converts it to a pandas DataFrame.

        Args:
            report_path (str): Path to the HLS synthesis report.
            table_header (str): The header line that precedes the table.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the table data or None if not found.
        """
        return self._extract_table(report_path, table_header)

    def extract_all_utilizations(self, report_path: str) -> Dict[str, Dict[str, int]]:
        """
        Extracts all FPGA resource utilization metrics from the HLS synthesis report.

        Args:
            report_path (str): Path to the HLS synthesis report.

        Returns:
            Dict[str, Dict[str, int]]: Nested dictionary containing all resource metrics categorized by resource type.
        """
        utilization = {}
        resource_tables = {
            'LUTs': 'Resource Utilization Estimates',
            'Flip-Flops': 'Flip-Flops',
            'DSP48E Blocks': 'DSP48E Blocks',
            'Block RAM': 'Block RAM'
        }

        for resource, header in resource_tables.items():
            df = self.extract_utilization_table(report_path, header)
            if df is not None:
                # Assuming the table has columns like 'Resource', 'Used', 'Available', 'Utilization (%)'
                for _, row in df.iterrows():
                    resource_name = row.get('Resource')
                    if pd.isna(resource_name):
                        continue
                    used = row.get('Used', '0').replace(',', '').replace('%', '')
                    available = row.get('Available', '0').replace(',', '').replace('%', '')
                    utilization_pct = row.get('Utilization (%)', '0').replace(',', '').replace('%', '')
                    try:
                        utilization[f"{resource}_Used_{resource_name}"] = int(float(used))
                        utilization[f"{resource}_Available_{resource_name}"] = int(float(available))
                        utilization[f"{resource}_Utilization_{resource_name}"] = int(float(utilization_pct))
                        self.logger.debug(f"Extracted {resource} - {resource_name}: Used={used}, Available={available}, Utilization={utilization_pct}%")
                    except ValueError:
                        self.logger.warning(f"Non-integer value encountered in {resource} table for {resource_name}.")
            else:
                self.logger.warning(f"No data extracted for resource type: {resource}")

        return utilization

    @staticmethod
    def save_utilization(utilization: Dict[str, int], save_path: str):
        """
        Saves the utilization metrics to a pickle file.

        Args:
            utilization (Dict[str, int]): Resource utilization metrics.
            save_path (str): Path to save the pickle file.
        """
        logger = Logger.get_logger(__name__, log_level=logging.INFO)
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(utilization, f)
            logger.info(f"Utilization metrics saved to {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save utilization metrics to {save_path}: {e}")
            raise

    @staticmethod
    def load_utilization(load_path: str) -> Dict[str, int]:
        """
        Loads the utilization metrics from a pickle file.

        Args:
            load_path (str): Path to the pickle file containing utilization metrics.

        Returns:
            Dict[str, int]: Loaded resource utilization metrics.
        """
        logger = Logger.get_logger(__name__, log_level=logging.INFO)
        if not os.path.exists(load_path):
            logger.error(f"Utilization file not found at {load_path}.")
            raise FileNotFoundError(f"Utilization file not found at {load_path}.")

        try:
            with open(load_path, 'rb') as f:
                utilization = pickle.load(f)
            logger.info(f"Utilization metrics loaded from {load_path}.")
            return utilization
        except Exception as e:
            logger.error(f"Failed to load utilization metrics from {load_path}: {e}")
            raise


if __name__ == "__main__":
    from src.utils.common_utils import CommonUtils

    # Initialize HLSUtils
    hls_utils = HLSUtils()

    # Path to the synthesis report
    report_path = "/home/david/Bachelor/hls4ml_adsb/hls_model/hls4ml_prj/myproject_prj/solution1/syn/report/myproject_csynth.rpt"

    # Extract utilization metrics
    try:
        utilization_metrics = hls_utils.extract_all_utilizations(report_path)
        print("Extracted Utilization Metrics:")
        for resource, metrics in utilization_metrics.items():
            print(f"\n{resource}:")
            for metric_key, metric_value in metrics.items():
                print(f"  {metric_key}: {metric_value}")
    except Exception as e:
        print(f"An error occurred during utilization extraction: {e}")

    # Save the utilization metrics
    utilization_save_path = "/path/to/save/utilization_metrics.pkl"
    HLSUtils.save_utilization(utilization_metrics, utilization_save_path)

    # Load the utilization metrics
    loaded_utilization = HLSUtils.load_utilization(utilization_save_path)
    print("\nLoaded Utilization Metrics:")
    for key, value in loaded_utilization.items():
        print(f"{key}: {value}")
