import json
import os
import glob
import argparse
import sys
from datetime import datetime, timezone, timedelta
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def load_ignored_nodes(config_path: str = "ignored_nodes.txt") -> set[str]:
    """
    Load list of node IDs (with leading '!') to ignore from a text file.
    One node ID per line (e.g. !0b86cb0c). Lines starting with '#' are comments.
    Missing file is treated as 'no ignored nodes'.
    """
    ignored: set[str] = set()
    if not os.path.exists(config_path):
        return ignored

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Берём первый «слово»-токен на строке (ID до пробела/комментария)
            token = line.split()[0]
            if not token.startswith("!"):
                print(f"Warning: expected node ID starting with '!' in {config_path!r}: {line!r}")
                continue
            ignored.add(token)
    return ignored


# Nodes to ignore entirely when processing packets (from config file)
IGNORED_NODES = load_ignored_nodes()


def detect_dominant_node(jsonl_file: str) -> str | None:
    """
    Если в ignored_nodes.txt нет явного списка нод,
    автоматически выбираем ноду с наибольшим количеством пакетов
    в данном файле и игнорируем её. Это, как правило, приёмник: 
    он записывает свои пакеты, но большинство из них не уходит в сеть.
    """
    node_counts: Counter[str] = Counter()
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                node_id = data.get("from_node_id")
                if isinstance(node_id, str):
                    node_counts[node_id] += 1
    except OSError as e:
        print(f"Warning: failed to scan {jsonl_file!r} for dominant node: {e}")
        return None

    if not node_counts:
        return None

    dominant_node_id, _ = node_counts.most_common(1)[0]
    return dominant_node_id

# Tee class to write to both file and stdout
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Function to process a JSONL file and create histograms
def process_jsonl_file(jsonl_file, recreate=False):
    # Extract base name without extension for naming output files
    base_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    combined_histogram = f"{base_name}_combined.png"
    
    # Check if histogram already exists (unless recreate is True)
    if not recreate and os.path.exists(combined_histogram):
        print(f"\nSkipping {jsonl_file} - histogram already exists")
        return
    
    # Determine which nodes to ignore for this file
    # IDs are of the form '!xxxxxxxx' (same as 'from_node_id' in JSONL)
    effective_ignored_nodes = set(IGNORED_NODES)
    if not effective_ignored_nodes:
        auto_node_id = detect_dominant_node(jsonl_file)
        if auto_node_id is not None:
            effective_ignored_nodes.add(auto_node_id)
            print(f"Auto-ignoring dominant node {auto_node_id} in {jsonl_file} "
                  f"(ignored_nodes.txt is empty)")

    # Read and filter JSONL file
    position_counts = Counter()
    device_counts = Counter()
    environment_counts = Counter()
    total_lines = 0
    first_timestamp = None
    last_timestamp = None
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                from_node = data.get('from_node')
                node_id = data.get('from_node_id')
                if node_id in effective_ignored_nodes:
                    continue

                total_lines += 1

                timestamp = data.get('timestamp')
                if timestamp is not None:
                    if first_timestamp is None or timestamp < first_timestamp:
                        first_timestamp = timestamp
                    if last_timestamp is None or timestamp > last_timestamp:
                        last_timestamp = timestamp
                
                payload_preview = data.get('payload_preview', '')
                from_node_name = data.get('from_node_longName', f"Node {from_node}")
                short_name = data.get('from_node_id')
                # Include short name (ID) for better identification
                if short_name:
                    key = f"{from_node_name} [{short_name}]"
                else:
                    key = f"{from_node_name}"
                
                if payload_preview.startswith('[Position:'):
                    position_counts[key] += 1
                elif payload_preview == '[Telemetry: Device]':
                    device_counts[key] += 1
                elif payload_preview == '[Telemetry: Environment]':
                    environment_counts[key] += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {line.strip()}")
                print(f"Error: {e}")
                continue
    
    print(f"\n{'='*70}")
    print(f"Processing: {jsonl_file}")
    print(f"{'='*70}")
    
    # Get all unique nodes across all three types
    all_nodes = set(position_counts.keys()) | set(device_counts.keys()) | set(environment_counts.keys())
    
    if not all_nodes:
        print(f"\nNo telemetry entries found in {jsonl_file}")
        return
    
    # Sort nodes by total count (sum of all three types)
    def get_total_count(node):
        return position_counts[node] + device_counts[node] + environment_counts[node]
    
    sorted_nodes = sorted(all_nodes, key=get_total_count, reverse=True)
    
    # Prepare data for printing
    print(f"\nCombined Histogram - Position, Device, and Environment Telemetry:\n")
    print(f"{'Node Name (ID)':<50} {'Position':>10} {'Device':>10} {'Environment':>12} {'Total':>10}")
    print("-" * 96)
    
    position_total = 0
    device_total = 0
    environment_total = 0
    
    for node in sorted_nodes:
        pos_count = position_counts[node]
        dev_count = device_counts[node]
        env_count = environment_counts[node]
        total = pos_count + dev_count + env_count
        
        position_total += pos_count
        device_total += dev_count
        environment_total += env_count
        
        print(f"{node:<50} {pos_count:>10} {dev_count:>10} {env_count:>12} {total:>10}")
    
    grand_total = position_total + device_total + environment_total
    print("-" * 96)
    
    # Format time period in UTC+7
    time_period_str = ""
    if first_timestamp is not None and last_timestamp is not None:
        utc7 = timezone(timedelta(hours=7))
        first_dt = datetime.fromtimestamp(first_timestamp, tz=utc7)
        last_dt = datetime.fromtimestamp(last_timestamp, tz=utc7)
        time_period_str = f" ({first_dt.strftime('%Y-%m-%d %H:%M:%S')} - {last_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC+7)"
    
    print(f"{'TOTAL':<50} {position_total:>10} {device_total:>10} {environment_total:>12} {grand_total:>10} / {total_lines}{time_period_str}")
    
    # Create combined visual histogram with stacked bars (sorted by total)
    nodes = sorted_nodes
    n_nodes = len(nodes)
    pos_values = [position_counts[node] for node in nodes]
    dev_values = [device_counts[node] for node in nodes]
    env_values = [environment_counts[node] for node in nodes]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, max(6, n_nodes * 0.3)))
    
    # Bar positions
    y_pos = np.arange(n_nodes)
    
    # Create stacked bars - Position at bottom, Device in middle, Environment on top
    ax.barh(y_pos, pos_values, label='Position', color='#2E86AB', alpha=0.8)
    ax.barh(y_pos, dev_values, left=pos_values, label='Device', color='#A23B72', alpha=0.8)
    ax.barh(y_pos, env_values, left=np.array(pos_values) + np.array(dev_values), label='Environment', color='#F18F01', alpha=0.8)
    
    # Add value labels on stacked bars
    for i, (pos, dev, env) in enumerate(zip(pos_values, dev_values, env_values)):
        total = pos + dev + env
        if total > 0:
            # Position label (at middle of position segment)
            if pos > 0:
                ax.text(pos / 2, i, f'{pos}', va='center', ha='center', fontsize=8, color='white', weight='bold')
            # Device label (at middle of device segment)
            if dev > 0:
                ax.text(pos + dev / 2, i, f'{dev}', va='center', ha='center', fontsize=8, color='white', weight='bold')
            # Environment label (at middle of environment segment)
            if env > 0:
                ax.text(pos + dev + env / 2, i, f'{env}', va='center', ha='center', fontsize=8, color='white', weight='bold')
            # Total label at the end
            ax.text(total, i, f' {total}', va='center', ha='left', fontsize=9, weight='bold')
    
    # Calculate packets per hour and prepare additional info
    packets_per_hour_str = ""
    hours_in_interval = 0
    if first_timestamp is not None and last_timestamp is not None and n_nodes > 0:
        hours_in_interval = (last_timestamp - first_timestamp) / 3600.0
        if hours_in_interval > 0:
            packets_per_hour = grand_total / hours_in_interval
            packets_per_hour_str = f", {packets_per_hour:.2f} packets/hour"
    
    # Build title with nodes count and hours count
    title_info = f"Total: {grand_total} / {total_lines} packets"
    if n_nodes > 0:
        title_info += f", {n_nodes} nodes"
    if hours_in_interval > 0:
        title_info += f", {hours_in_interval:.2f} hours"
    title_info += f"{time_period_str}{packets_per_hour_str}"
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nodes)
    ax.set_xlabel('Count')
    ax.set_ylabel('Node')
    ax.set_title(f"Position, Device, and Environment Telemetry - {base_name}\n({title_info})")
    # Place legend outside plot area to the right to avoid overlapping with bars
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)
    
    # Set y-axis limits to eliminate padding - bars are at positions 0 to n_nodes-1
    # For barh, matplotlib inverts y-axis so y=0 appears at top
    # Set limits with top value first: (n_nodes-0.5, -0.5) to eliminate padding
    ax.set_ylim(n_nodes - 0.5, -0.5)
    
    # Adjust x-axis limits to accommodate text labels
    max_total = max([p + d + e for p, d, e in zip(pos_values, dev_values, env_values)], default=1)
    ax.set_xlim(0, max_total * 1.15 if max_total > 0 else 1)
    
    # Reduce top margin to minimize space between title and first bar
    # tight_layout with reduced padding, especially at the top
    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
    # Further reduce top margin and add right margin for legend
    fig.subplots_adjust(top=0.96, right=0.85)
    plt.savefig(combined_histogram, dpi=150, bbox_inches='tight')
    plt.close()

    # Also create a version with nodes sorted alphabetically (by label)
    alpha_histogram = f"{base_name}_alphabetical.png"
    nodes_alpha = sorted(all_nodes)
    n_nodes_alpha = len(nodes_alpha)
    pos_values_alpha = [position_counts[node] for node in nodes_alpha]
    dev_values_alpha = [device_counts[node] for node in nodes_alpha]
    env_values_alpha = [environment_counts[node] for node in nodes_alpha]

    fig_alpha, ax_alpha = plt.subplots(figsize=(14, max(6, n_nodes_alpha * 0.3)))
    y_pos_alpha = np.arange(n_nodes_alpha)

    ax_alpha.barh(y_pos_alpha, pos_values_alpha, label='Position', color='#2E86AB', alpha=0.8)
    ax_alpha.barh(y_pos_alpha, dev_values_alpha, left=pos_values_alpha, label='Device', color='#A23B72', alpha=0.8)
    ax_alpha.barh(y_pos_alpha, env_values_alpha,
                  left=np.array(pos_values_alpha) + np.array(dev_values_alpha),
                  label='Environment', color='#F18F01', alpha=0.8)

    for i, (pos, dev, env) in enumerate(zip(pos_values_alpha, dev_values_alpha, env_values_alpha)):
        total = pos + dev + env
        if total > 0:
            if pos > 0:
                ax_alpha.text(pos / 2, i, f'{pos}', va='center', ha='center',
                              fontsize=8, color='white', weight='bold')
            if dev > 0:
                ax_alpha.text(pos + dev / 2, i, f'{dev}', va='center', ha='center',
                              fontsize=8, color='white', weight='bold')
            if env > 0:
                ax_alpha.text(pos + dev + env / 2, i, f'{env}', va='center', ha='center',
                              fontsize=8, color='white', weight='bold')
            ax_alpha.text(total, i, f' {total}', va='center', ha='left',
                          fontsize=9, weight='bold')

    ax_alpha.set_yticks(y_pos_alpha)
    ax_alpha.set_yticklabels(nodes_alpha)
    ax_alpha.set_xlabel('Count')
    ax_alpha.set_ylabel('Node')
    ax_alpha.set_title(f"Position, Device, and Environment Telemetry (alphabetical) - {base_name}\n({title_info})")
    ax_alpha.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)
    ax_alpha.set_ylim(n_nodes_alpha - 0.5, -0.5)

    max_total_alpha = max([p + d + e for p, d, e in zip(pos_values_alpha,
                                                        dev_values_alpha,
                                                        env_values_alpha)], default=1)
    ax_alpha.set_xlim(0, max_total_alpha * 1.15 if max_total_alpha > 0 else 1)

    plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
    fig_alpha.subplots_adjust(top=0.96, right=0.85)
    plt.savefig(alpha_histogram, dpi=150, bbox_inches='tight')
    plt.close(fig_alpha)
    
    print(f"\nCombined histogram saved to: {combined_histogram}")
    print(f"Alphabetical histogram saved to: {alpha_histogram}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate histograms from packet monitor JSONL files')
parser.add_argument(
    '-r', '--recreate',
    nargs='?',
    const='1',
    default=None,
    metavar='N_OR_ALL',
    help='Recreate histograms even if they already exist. '
         'Optional value: N (number of newest histograms to regenerate, default 1) or "all" for all.'
)
parser.add_argument('-l', '--log', type=str, default=None,
                    help='Log file path (default: packets_YYYY-MM-DD_HH-MM-SS.log)')
args = parser.parse_args()

# Set up log file
if args.log is None:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'packets_{timestamp}.log'
else:
    log_file = args.log

# Redirect stdout to both console and log file
log_f = open(log_file, 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_f)

print(f"Log file: {log_file}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}\n")

try:
    # Find all JSONL files in the current directory
    jsonl_files = glob.glob('*.jsonl')
    # Sort by modification time: oldest first, newest last
    jsonl_files.sort(key=os.path.getmtime)

    if not jsonl_files:
        print("No JSONL files found in the current directory")
    else:
        print(f"Found {len(jsonl_files)} JSONL file(s)")

        # Determine recreate mode:
        #   None  -> do not force regeneration (only missing histograms are created)
        #   'all' -> regenerate histograms for all JSONL files
        #   'N'   -> regenerate histograms for N newest JSONL files
        recreate_spec = args.recreate
        recreate_count = None

        if recreate_spec is None:
            print("Recreate mode: only missing histograms will be generated")
        elif recreate_spec == 'all':
            print("Recreate mode: will regenerate histograms for ALL JSONL files")
        else:
            try:
                recreate_count = int(recreate_spec)
                if recreate_count <= 0:
                    print(f"Invalid recreate count '{recreate_spec}', must be positive. "
                          "Falling back to default of 1.")
                    recreate_count = 1
                print(f"Recreate mode: will regenerate histograms for {recreate_count} newest JSONL file(s)")
            except ValueError:
                print(f"Invalid value for --recreate: '{recreate_spec}'. "
                      "Use a positive integer or 'all'. Treating as no recreate.")
                recreate_spec = None

        total_files = len(jsonl_files)
        for idx, jsonl_file in enumerate(jsonl_files):
            # Decide whether to force recreation for this file
            if recreate_spec is None:
                recreate_flag = False
            elif recreate_spec == 'all':
                recreate_flag = True
            else:
                # Regenerate only the newest N files (at the end of sorted list)
                recreate_flag = idx >= max(0, total_files - recreate_count)

            process_jsonl_file(jsonl_file, recreate=recreate_flag)
        
        print(f"\n{'='*70}")
        print("Processing complete!")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
finally:
    # Restore stdout and close log file
    sys.stdout = original_stdout
    log_f.close()
    print(f"\nLog saved to: {log_file}")
