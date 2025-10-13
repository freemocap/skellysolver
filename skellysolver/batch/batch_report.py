"""Batch processing report generation.

Generates comprehensive reports from batch processing results:
- Summary statistics
- Comparison tables
- Performance metrics
- HTML reports
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .batch_processor  import BatchResult, BatchJobResult


class BatchReportGenerator:
    """Generate reports from batch processing results.
    
    Creates:
    - Summary statistics CSV
    - Comparison table
    - Performance metrics
    - HTML report
    """
    
    def __init__(self, *, batch_result: BatchResult) -> None:
        """Initialize report generator.
        
        Args:
            batch_result: Batch result to report on
        """
        self.batch_result = batch_result
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all jobs.
        
        Returns:
            DataFrame with job summaries
        """
        rows = []
        
        for job_result in self.batch_result.job_results:
            row = {
                "job_id": job_result.job_id,
                "job_name": job_result.job_name,
                "success": job_result.success,
                "duration_s": job_result.duration_seconds,
                "error": job_result.error,
            }
            
            # Add optimization metrics if available
            if job_result.optimization_result is not None:
                opt = job_result.optimization_result
                row.update({
                    "iterations": opt.num_iterations,
                    "initial_cost": opt.initial_cost,
                    "final_cost": opt.final_cost,
                    "cost_reduction_pct": opt.cost_reduction_percent,
                })
            else:
                row.update({
                    "iterations": None,
                    "initial_cost": None,
                    "final_cost": None,
                    "cost_reduction_pct": None,
                })
            
            # Add metadata
            for key, value in job_result.metadata.items():
                row[f"param_{key}"] = value
            
            rows.append(row)
        
        return pd.DataFrame(data=rows)
    
    def generate_statistics(self) -> dict[str, Any]:
        """Generate statistical summary.
        
        Returns:
            Dictionary with statistics
        """
        successful_results = [
            r for r in self.batch_result.job_results
            if r.success and r.optimization_result is not None
        ]
        
        if not successful_results:
            return {
                "n_jobs": self.batch_result.n_jobs_total,
                "n_successful": 0,
                "n_failed": self.batch_result.n_jobs_failed,
                "success_rate": 0.0,
            }
        
        # Extract metrics
        final_costs = [r.optimization_result.final_cost for r in successful_results]
        iterations = [r.optimization_result.num_iterations for r in successful_results]
        durations = [r.duration_seconds for r in successful_results]
        cost_reductions = [r.optimization_result.cost_reduction_percent for r in successful_results]
        
        stats = {
            "n_jobs": self.batch_result.n_jobs_total,
            "n_successful": self.batch_result.n_jobs_successful,
            "n_failed": self.batch_result.n_jobs_failed,
            "success_rate": self.batch_result.success_rate,
            "final_cost": {
                "mean": float(np.mean(final_costs)),
                "std": float(np.std(final_costs)),
                "min": float(np.min(final_costs)),
                "max": float(np.max(final_costs)),
                "median": float(np.median(final_costs)),
            },
            "iterations": {
                "mean": float(np.mean(iterations)),
                "std": float(np.std(iterations)),
                "min": int(np.min(iterations)),
                "max": int(np.max(iterations)),
            },
            "duration": {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "total": self.batch_result.total_duration_seconds,
            },
            "cost_reduction": {
                "mean": float(np.mean(cost_reductions)),
                "std": float(np.std(cost_reductions)),
                "min": float(np.min(cost_reductions)),
                "max": float(np.max(cost_reductions)),
            }
        }
        
        return stats
    
    def save_summary_csv(self, *, filepath: Path) -> None:
        """Save summary table to CSV.
        
        Args:
            filepath: Output CSV path
        """
        df = self.generate_summary_table()
        df.to_csv(path_or_buf=filepath, index=False)
        
        print(f"✓ Saved summary CSV: {filepath}")
    
    def save_statistics_json(self, *, filepath: Path) -> None:
        """Save statistics to JSON.
        
        Args:
            filepath: Output JSON path
        """
        import json
        
        stats = self.generate_statistics()
        
        with open(filepath, mode='w') as f:
            json.dump(obj=stats, fp=f, indent=2)
        
        print(f"✓ Saved statistics JSON: {filepath}")
    
    def generate_html_report(self, *, filepath: Path) -> None:
        """Generate HTML report.
        
        Args:
            filepath: Output HTML path
        """
        df = self.generate_summary_table()
        stats = self.generate_statistics()
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Batch Report: {self.batch_result.batch_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
        .stat-card {{ background: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .stat-card h3 {{ margin: 0 0 10px 0; color: #2980b9; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #ecf0f1; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .failure {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Batch Processing Report: {self.batch_result.batch_name}</h1>
        
        <h2>Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Jobs</h3>
                <div class="stat-value">{stats['n_jobs']}</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="stat-value">{stats['success_rate']*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>Total Time</h3>
                <div class="stat-value">{stats['duration']['total']/60:.1f}m</div>
            </div>
            <div class="stat-card">
                <h3>Avg Duration</h3>
                <div class="stat-value">{stats['duration']['mean']:.1f}s</div>
            </div>
        </div>
        
        <h2>Cost Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Mean Final Cost</h3>
                <div class="stat-value">{stats['final_cost']['mean']:.4f}</div>
                <div>± {stats['final_cost']['std']:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Best Final Cost</h3>
                <div class="stat-value">{stats['final_cost']['min']:.4f}</div>
            </div>
            <div class="stat-card">
                <h3>Mean Reduction</h3>
                <div class="stat-value">{stats['cost_reduction']['mean']:.1f}%</div>
            </div>
        </div>
        
        <h2>Job Details</h2>
        {df.to_html(classes='job-table', index=False, na_rep='-')}
        
        <h2>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
    </div>
</body>
</html>"""
        
        filepath.write_text(data=html, encoding='utf-8')
        
        print(f"✓ Saved HTML report: {filepath}")
    
    def save_all_reports(self, *, output_dir: Path) -> None:
        """Save all report formats.
        
        Args:
            output_dir: Directory for reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating batch reports...")
        
        self.save_summary_csv(filepath=output_dir / "batch_summary.csv")
        self.save_statistics_json(filepath=output_dir / "batch_statistics.json")
        self.generate_html_report(filepath=output_dir / "batch_report.html")
        
        print(f"\n✓ All reports saved to {output_dir}")
        print(f"  → Open {output_dir / 'batch_report.html'} in a browser!")


def compare_parameter_sweep_results(
    *,
    batch_result: BatchResult,
    parameter_name: str
) -> pd.DataFrame:
    """Compare results across parameter values.
    
    Useful for analyzing parameter sweep results.
    
    Args:
        batch_result: Batch result from parameter sweep
        parameter_name: Name of parameter to compare
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for job_result in batch_result.job_results:
        if not job_result.success or job_result.optimization_result is None:
            continue
        
        # Get parameter value
        param_value = job_result.metadata.get(parameter_name)
        if param_value is None:
            continue
        
        opt = job_result.optimization_result
        
        rows.append({
            "parameter_value": param_value,
            "final_cost": opt.final_cost,
            "cost_reduction_pct": opt.cost_reduction_percent,
            "iterations": opt.num_iterations,
            "duration_s": job_result.duration_seconds,
        })
    
    df = pd.DataFrame(data=rows)
    
    # Sort by parameter value
    df = df.sort_values(by="parameter_value")
    
    return df


def find_best_parameters(
    *,
    batch_result: BatchResult
) -> dict[str, Any]:
    """Find parameter combination with best results.
    
    Args:
        batch_result: Batch result from parameter sweep
        
    Returns:
        Dictionary with best parameter values
    """
    best_job = batch_result.get_best_result()
    
    if best_job is None:
        return {}
    
    return best_job.metadata
