# Activate conda environment `midi_dit` and run plotting tools
# Usage examples:
#   .\scripts\run_plot.ps1           -> generate latest run + comparison to plots/
#   .\scripts\run_plot.ps1 --per-run -> generate per-run plots

conda activate midi_dit
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate conda environment 'midi_dit'. Ensure conda is initialized in this shell (run 'conda init powershell' once), then restart PowerShell."
    exit 1
}

# Default: generate latest + comparison
python .\utils\plot_training_curves.py --logs-dir logs --output-dir plots @args

conda deactivate
