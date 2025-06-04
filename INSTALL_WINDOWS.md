# Windows GPU Setup Guide

*For when you want the pain of dependency management with the convenience of Windows, now with bonus CUDA complexity.*

> **Note**: This guide covers Windows-specific setup details. For basic usage and MCP configuration, see the main `README.md`.

## Prerequisites

Before diving into this technological adventure, ensure you have:
- Windows 10 (21H2+) or Windows 11 
- An NVIDIA GPU (obviously)
- Patience (not included)
- Coffee (highly recommended)
- A sense of humor (required for Windows CUDA setup)

## Step 1: Update Windows

Yes, really. Update Windows first.

No, that random restart won't break everything. Probably.

## Step 2: Install Python

Download Python from [python.org](https://www.python.org/downloads/). 

**Important**: During installation, check "Add Python to PATH". This saves you from PATH hell later.

Verify installation:
```cmd
python --version
pip --version
```

## Step 3: NVIDIA Driver and CUDA Toolkit

### Install NVIDIA Driver
1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Download the latest Game Ready or Studio driver for your GPU
3. Install it
4. Restart (because Windows)

### Install CUDA Toolkit
1. Visit [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select:
   - OS: Windows
   - Architecture: x86_64
   - Version: 10/11 (your Windows version)
   - Installer: exe (network) - for the full experience

3. Download and run the installer
4. Choose "Custom" installation if you want to avoid the bloat
5. At minimum, select:
   - CUDA Toolkit
   - CUDA Runtime
   - CUDA Documentation (optional but helpful)

### Verify CUDA Installation
Open Command Prompt and run:
```cmd
nvidia-smi
nvcc --version
```

Both should work. If `nvcc` isn't found, CUDA toolkit isn't in your PATH.

## Step 4: Python Dependencies

Follow the installation instructions in `README.md`. The key Windows-specific notes:

- Virtual environment activation: `flow-env\Scripts\activate` (not `activate.bat`)
- Use `faiss-cpu` - don't fight the `faiss-gpu` battle on Windows
- PyTorch CUDA version must match your `nvcc --version` output

## Step 5: Test Everything

Test CUDA:
```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.device_count())        # Should show your GPU count  
print(torch.cuda.get_device_name(0))    # Should show your GPU name
```

Test the MCP server:
```cmd
cd C:\path\to\your\local_flow
flow-env\Scripts\activate
python rag_mcp_server.py
```

It should start without errors and wait for JSON-RPC input. Press Ctrl+C to exit.

## Windows-Specific MCP Configuration

The README shows the basic config. Here are Windows-specific gotchas:

### Path Requirements
- **Always use double backslashes**: `C:\\path\\to\\file`
- **Use absolute paths**: Relative paths cause mysterious failures
- **Python executable path**: Use `flow-env\\Scripts\\python.exe` directly

### Alternative Configurations

If the standard config doesn't work, try these:

**PowerShell version:**
```json
{
  "mcpServers": {
    "LocalFlow": {
      "command": "powershell.exe",
      "args": ["-Command", "& C:\\path\\to\\your\\local_flow\\flow-env\\Scripts\\Activate.ps1; python C:\\path\\to\\your\\local_flow\\rag_mcp_server.py"],
      "env": {
        "RAG_DATA_DIR": "C:\\path\\to\\your\\vector_db"
      },
      "scopes": ["rag_read", "rag_write"], 
      "tools": ["add_source", "query_context", "list_sources", "remove_source"]
    }
  }
}
```

## Troubleshooting

### Python/CUDA Issues
- **"torch.cuda.is_available() is False"**: Check CUDA toolkit installation and PyTorch CUDA version match
- **"nvcc not found"**: CUDA toolkit not in PATH. Reinstall with PATH option checked
- **"CUDA out of memory"**: Close other GPU applications, or use smaller models

### MCP Issues  
- **"Tool not found" in Cursor**: Check your `mcp.json` paths and restart Cursor
- **"Command failed"**: Verify the activation script path: `flow-env\Scripts\activate` (not `activate.bat`)
- **"Permission denied"**: Run Command Prompt as Administrator
- **JSON parsing errors**: Check for proper escaping of backslashes in paths

### Path Issues
- **"python not found"**: Python not in PATH. Reinstall Python with PATH option
- **"The system cannot find the path specified"**: Use absolute paths in `mcp.json`
- **Virtual environment activation fails**: Check if `flow-env\Scripts\activate` exists

### Dependencies
- **"No module named 'faiss'"**: Install `faiss-cpu`, not `faiss-gpu`
- **"Microsoft Visual C++ 14.0 is required"**: Install Visual Studio Build Tools or Visual Studio Community
- **Import errors**: Make sure virtual environment is activated when installing packages

## Performance Tips

- **GPU Memory**: Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` environment variable if you get memory errors
- **Embedding Speed**: The embedding model runs on GPU even with `faiss-cpu` 
- **Storage Location**: Put `RAG_DATA_DIR` on an SSD for better performance

## Final Notes

Your Windows setup should now have:
- Native Windows Python with CUDA toolkit
- PyTorch with GPU support  
- All RAG dependencies installed
- MCP server configured for Cursor

No WSL2 required. Welcome to native Windows AI development.

*It actually works pretty well once you get through the setup.* 