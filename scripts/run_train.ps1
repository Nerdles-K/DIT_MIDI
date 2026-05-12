# Activate conda environment `midi_dit` and run training script
# Usage: Open PowerShell, then run: .\scripts\run_train.ps1

# Try to activate the environment (requires conda initialized for PowerShell)
conda activate midi_dit
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate conda environment 'midi_dit'. Ensure conda is initialized in this shell (run 'conda init powershell' once), then restart PowerShell."
    exit 1
}

# Run training (pass through any args if provided)
python .\train_midi.py $args

# Deactivate optionally
conda deactivate
