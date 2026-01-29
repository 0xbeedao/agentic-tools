---
name: deploying-with-mpremote
description: Deploys files to MicroPython/CircuitPython devices using mpremote CLI. Use when copying code.py or other files to a connected device.
---

# Deploying Files with mpremote

Use `mpremote` to transfer files and interact with MicroPython/CircuitPython devices over serial.

## Prerequisites

- Python 3 installed
- mpremote installed: `pip install mpremote`
- Device connected via USB

## Connecting

```bash
# Auto-connect to first available device
mpremote

# List available devices
mpremote connect list

# Connect to specific port (Windows)
mpremote connect COM3

# Connect to specific port (Linux/Mac)
mpremote connect /dev/ttyACM0
```

### Windows shortcuts
- `c0` = COM0, `c1` = COM1, etc.

### Linux shortcuts
- `a0` = /dev/ttyACM0, `u0` = /dev/ttyUSB0, etc.

## File Operations

### Copy file to device
```bash
mpremote cp code.py :code.py
mpremote cp lib/mymodule.py :lib/mymodule.py
```

### Copy file from device
```bash
mpremote cp :code.py code.py
```

### Copy directory recursively
```bash
mpremote cp -r lib/ :lib/
```

### List files on device
```bash
mpremote ls
mpremote ls :lib/
```

### View file contents
```bash
mpremote cat :code.py
```

### Remove file
```bash
mpremote rm :old_file.py
```

### Remove directory recursively
```bash
mpremote rm -r :lib/
```

### Create directory
```bash
mpremote mkdir :lib
```

### Show disk usage
```bash
mpremote df
```

### Directory tree
```bash
mpremote tree :
```

## Running Code

### Enter REPL
```bash
mpremote repl
```
Exit with `Ctrl-]` or `Ctrl-x`

### Run a local script on device
```bash
mpremote run script.py
```

### Execute inline code
```bash
mpremote exec "import machine; print(machine.freq())"
```

## Common Workflows

### Deploy code.py
```bash
mpremote cp code.py :code.py
```

### Deploy and enter REPL
```bash
mpremote cp code.py :code.py + repl
```

### Deploy multiple files
```bash
mpremote cp code.py :code.py + cp boot.py :boot.py
```

### Soft reset device
```bash
mpremote soft-reset
```

### Hard reset device
```bash
mpremote reset
```

### Mount local directory (live development)
```bash
mpremote mount .
```
Device sees local files as `/remote`, useful for testing without copying.

## Targeting Specific Device

```bash
mpremote connect COM5 cp code.py :code.py
```

## Notes

- Prefix remote paths with `:` (e.g., `:code.py`)
- Local paths have no prefix
- Use `+` to chain commands: `mpremote cp file.py :file.py + repl`
- Commands auto-connect if no `connect` specified
