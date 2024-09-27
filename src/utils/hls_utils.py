# utils/hls_util.py

def extract_utilization(report_path):
    """
    Extract FPGA resource utilization from the HLS synthesis report.

    Parameters:
    report_path (str): Path to the HLS synthesis report.

    Returns:
    dict: Dictionary containing resource utilization metrics.
    """
    utilization = {}
    with open(report_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Utilization Estimates" in line:
                # Skip lines until we reach the table header
                while i < len(lines) and not lines[i].strip().startswith('|'):
                    i += 1
                # Extract headers
                headers = [h.strip() for h in lines[i].strip().split('|')[1:-1]]
                i += 2  # Skip the separator line after headers
                # Parse the table rows
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('*'):
                    if lines[i].strip().startswith('+'):
                        i += 1
                        continue
                    parts = [p.strip() for p in lines[i].strip().split('|')[1:-1]]
                    if len(parts) == len(headers):
                        resource_name = parts[0]
                        for j in range(1, len(parts)):
                            key = headers[j]
                            value = parts[j]
                            if value != '-' and value != '':
                                try:
                                    value_int = int(value.replace(',', '').replace('%', ''))
                                    utilization_key = f"{resource_name}_{key}"
                                    utilization[utilization_key] = value_int
                                except ValueError:
                                    pass  # Ignore non-integer values
                    i += 1
                break
            i += 1
    return utilization

if __name__ == "__main__":
    report_path = "/home/david/Bachelor/hls4ml_adsb/hls_model/hls4ml_prj/myproject_prj/solution1/syn/report/myproject_csynth.rpt"
    utilization_data = extract_utilization(report_path)
    print(utilization_data)