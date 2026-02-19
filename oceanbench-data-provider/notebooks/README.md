# OceanBench Notebooks

If notebooks open as raw JSON instead of the interactive view:

1. **Install Jupyter extension**  
   - Open Extensions (Ctrl+Shift+X)
   - Search for "Jupyter" (by Microsoft)
   - Install it
   - Reload Cursor if prompted

2. **Reopen the notebook**  
   - Close the tab if it's open
   - Open `complete_workflow.ipynb` (or any `.ipynb` file) from the file explorer

3. **Choose "Open With" if needed**  
   - Right‑click the notebook file
   - Select "Open With..."
   - Choose "Jupyter Notebook" (not "Text Editor")

4. **Alternative: run in browser**  
   ```bash
   # From the oceanbench/ directory:
   source oceanbench-venv/bin/activate
   cd oceanbench-data-provider
   pip install jupyter
   jupyter notebook notebooks/
   ```

## Using the project environment as the kernel

To run notebook cells with the same environment where you installed OceanBench Data Provider:

### Step 1: Make oceanbench-venv discoverable (one-time setup)

First, install ipykernel in your `oceanbench-venv` to make it appear in the kernel list:

```bash
# From the oceanbench/ directory
source oceanbench-venv/bin/activate

# Option 1: Install via optional dependency (includes ipykernel + jupyter)
pip install "oceanbench-data-provider[notebooks]"

# Option 2: Or install ipykernel manually
pip install ipykernel

# Then register the kernel
python -m ipykernel install --user --name oceanbench-venv --display-name "Python (oceanbench-venv)"
```

### Step 2: Select the kernel in your notebook

**In Cursor / VS Code:**

1. Open your notebook (`complete_workflow.ipynb`).
2. Click the **kernel name** (e.g. "Python 3") in the top-right corner.
3. Choose **"Select Another Kernel"** → **"Python Environments"**.
4. **Important:** Look for **"Python (oceanbench-venv)"** in the list of existing environments.
   - **Do NOT** click **"+ Create Python Environment"** — that will create a new `.venv` folder!
   - If you see `.venv` as "Recommended", ignore it.
5. If `oceanbench-venv` doesn't appear:
   - Close the dialog
   - Press **Ctrl+Shift+P** (or **Cmd+Shift+P** on Mac) to open Command Palette
   - Type: `Python: Select Interpreter`
   - Look for **"Enter interpreter path..."** or **"Find..."**
   - Browse to: `../oceanbench-venv/bin/python` (relative path from `oceanbench-data-provider/notebooks/`)
   - Or use the absolute path: `[path-to-oceanbench]/oceanbench-venv/bin/python`

**In Jupyter in the browser:**
- Activate the env and start Jupyter from it (as in step 4 above). The default kernel will be that environment.
- To make the env available as a named kernel in any Jupyter session: activate the env, then run  
  `python -m ipykernel install --user --name oceanbench-venv`  
  and choose "Python (oceanbench-venv)" from the kernel list.

