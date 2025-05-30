# WSL2 GPU Setup Guide

*For when you want the pain of Linux configuration with the convenience of Windows crashes.*

## Prerequisites

Before diving into this technological adventure, ensure you have:
- Windows 10 (21H2+) or Windows 11 
- An NVIDIA GPU (obviously)
- Patience (not included)
- Coffee (highly recommended)

## Step 1: Update Windows

Yes, really. Update Windows first.

No, that random restart won't break everything. Probably.

## Step 2: Install WSL2

Open PowerShell as Administrator. Feel the power.

```powershell
wsl --install
wsl --set-default-version 2
```

Install Ubuntu 24.04 LTS from Microsoft Store. Because we're fancy like that.

## Step 3: NVIDIA Driver (The Important Part)

**Critical**: Install the NVIDIA driver on *Windows*, not in WSL2. 

WSL2 uses your Windows driver. Installing Linux drivers inside WSL2 will create conflicts and sadness.

1. Visit the [NVIDIA CUDA on WSL page](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
2. Download the CUDA-enabled driver for WSL
3. Install it on Windows
4. Restart (because Windows)

## Step 4: Verify GPU Access in WSL2

Open your shiny Ubuntu terminal:

```bash
nvidia-smi
```

Should show your GPU info. If not:

```bash
# Try this path magic
export PATH="/usr/lib/wsl/lib/:$PATH"
nvidia-smi

# Or use Windows executable directly
/mnt/c/Windows/System32/nvidia-smi.exe
```

Look for the "CUDA Version" in the output. That's your maximum supported version.

## Step 5: Install CUDA Toolkit in WSL2

Visit [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads).

Select:
- OS: Linux
- Architecture: x86_64  
- Distribution: WSL-Ubuntu
- Version: 2.0
- Installer: deb (local) - because networking is hard

Follow their exact commands. Something like:

```bash
wget <their-magic-url>
sudo dpkg -i <cuda-repo-file>
sudo cp /var/cuda-repo-wsl-ubuntu-*/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4  # or whatever version
```

**Important**: Only install `cuda-toolkit-XX-X`. Don't install `cuda` or `cuda-drivers`.

Add to `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify with:
```bash
nvcc --version
```

## Step 6: Install PyTorch

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/).

Select your poison:
- PyTorch Build: Stable
- OS: Linux (yes, even for WSL2)
- Package: Pip
- Language: Python  
- CUDA: Match your `nvidia-smi` version

Example command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 7: The FAISS Situation

Here's where things get *interesting*.

`faiss-gpu` doesn't like recent CUDA versions. Your options:

### Option A: Use faiss-cpu (Recommended)
```bash
pip install faiss-cpu
```

Your embeddings still use GPU. FAISS search runs on CPU. Life goes on.

### Option B: Build faiss-gpu from source (for masochists)
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
mkdir build && cd build

# RTX 4080 Laptop = compute capability 8.9
cmake .. -DFAISS_ENABLE_GPU=ON -DCUDA_ARCHITECTURES="89" -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)  # Go get coffee. This takes forever.
cd ../python && pip install .
```

## Step 8: Does it work?

```python
import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.device_count())        # Should show your GPU count  
print(torch.cuda.get_device_name(0))    # Should show your GPU name
```

If all return expected values, congratulations! You've successfully navigated the maze of WSL2 GPU setup.

## Troubleshooting

- **nvidia-smi not found**: Check paths above
- **CUDA version mismatch**: Use the version from `nvidia-smi`, not what you think you have
- **faiss-gpu won't install**: Use faiss-cpu and move on with your life
- **Nothing works**: Turn it off and on again (classic Windows solution)

## Final Notes

Your setup should now have:
- Windows NVIDIA driver (handling GPU)
- WSL2 Ubuntu with CUDA toolkit  
- PyTorch with GPU support
- FAISS (probably the CPU version, and that's fine)

Welcome to the future. It's complicated, but it works.

*Most of the time.* 