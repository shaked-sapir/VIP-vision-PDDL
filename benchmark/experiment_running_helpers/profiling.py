"""
Profiling utilities for timing experiment operations.
"""

import json
import time
from contextlib import contextmanager
from typing import Dict, List
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TimingProfiler:
    """Collects timing information for various operations."""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.detailed_timings: Dict[str, List[Dict]] = {}
    
    @contextmanager
    def time_operation(self, operation_name: str, add_to_detailed: bool = False, detail_key: str = None):
        """Context manager to time an operation."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings[operation_name] = elapsed
            if add_to_detailed and detail_key:
                if detail_key not in self.detailed_timings:
                    self.detailed_timings[detail_key] = []
                self.detailed_timings[detail_key].append({
                    'operation': operation_name,
                    'time_seconds': elapsed
                })
    
    def add_timing(self, operation_name: str, elapsed_seconds: float):
        """Manually add a timing."""
        self.timings[operation_name] = elapsed_seconds
    
    def add_detailed_timing(self, category: str, operation_name: str, elapsed_seconds: float, metadata: Dict = None):
        """Add a detailed timing entry with metadata."""
        if category not in self.detailed_timings:
            self.detailed_timings[category] = []
        entry = {
            'operation': operation_name,
            'time_seconds': elapsed_seconds
        }
        if metadata:
            entry.update(metadata)
        self.detailed_timings[category].append(entry)
    
    def get_report(self) -> Dict:
        """Get a complete timing report."""
        return {
            'summary': self.timings,
            'detailed': self.detailed_timings,
            'total_time': sum(self.timings.values())
        }
    
    def save_report(self, output_path: Path):
        """Save timing report to JSON file."""
        report = self.get_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_timing_report(self, output_path: Path):
        """
        Create a visualization of the timing report showing which parts take the most time.
        
        Args:
            output_path: Path to save the plot (PNG file)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("  Warning: matplotlib not available, skipping timing plot")
            return
        
        report = self.get_report()
        summary = report.get('summary', {})
        
        if not summary:
            print("  Warning: No timing data available for plotting")
            return
        
        # Sort by time (descending) to show largest time consumers first
        sorted_timings = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        operations = [op for op, _ in sorted_timings]
        times = [time_val for _, time_val in sorted_timings]
        
        # Create figure with two subplots: bar chart and pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Horizontal bar chart (main visualization)
        colors_bar = plt.cm.viridis_r([t / max(times) for t in times])  # Color by relative time
        bars = ax1.barh(operations, times, color=colors_bar)
        ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Operation', fontsize=12, fontweight='bold')
        ax1.set_title('Timing Breakdown by Operation', fontsize=14, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, times)):
            width = bar.get_width()
            # Format time: use seconds if < 60, otherwise minutes
            if width < 60:
                label = f'{width:.1f}s'
            else:
                label = f'{width/60:.1f}m'
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    label, ha='left' if width > max(times) * 0.05 else 'right',
                    va='center', fontsize=9, fontweight='bold')
        
        # Invert y-axis to show largest at top
        ax1.invert_yaxis()
        
        # Pie chart (percentage view)
        # Show top 10 operations if there are many
        max_pie_items = 10
        if len(times) > max_pie_items:
            pie_operations = operations[:max_pie_items]
            pie_times = times[:max_pie_items]
            other_time = sum(times[max_pie_items:])
            if other_time > 0:
                pie_operations.append(f'Others ({len(times) - max_pie_items} operations)')
                pie_times.append(other_time)
        else:
            pie_operations = operations
            pie_times = times
        
        colors_pie = plt.cm.Set3(range(len(pie_operations)))
        wedges, texts, autotexts = ax2.pie(pie_times, labels=pie_operations, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
        ax2.set_title('Time Distribution (%)', fontsize=14, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        for text in texts:
            text.set_fontsize(9)
        
        # Add total time as subtitle
        total_time = sum(times)
        if total_time < 60:
            total_label = f'Total Time: {total_time:.1f} seconds'
        elif total_time < 3600:
            total_label = f'Total Time: {total_time/60:.1f} minutes'
        else:
            total_label = f'Total Time: {total_time/3600:.1f} hours'
        
        fig.suptitle(total_label, fontsize=12, y=0.98, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [PROFILING] Timing plot saved to {output_path.name}")

