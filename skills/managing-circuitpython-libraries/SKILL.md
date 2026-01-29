---
name: managing-circuitpython-libraries
description: Manages CircuitPython libraries using circup CLI. Use when installing, updating, or managing libraries on a CircuitPython device.
---

# Managing CircuitPython Libraries with circup

Use the `circup` CLI tool to manage libraries on connected CircuitPython devices.

## Prerequisites

- Python 3.5+ installed
- circup installed: `pip3 install --user circup`
- CircuitPython device connected (mounted as CIRCUITPY)

## Common Commands

### View installed libraries
```bash
circup show
```

### Check for updates
```bash
circup list
```

### Update all outdated libraries
```bash
circup update
```

### Install specific library
```bash
circup install <library_name>
```

Example:
```bash
circup install adafruit_neopixel
circup install adafruit_bus_device adafruit_register
```

### Auto-install from code.py imports
```bash
circup install --auto
```

### Install from requirements.txt
```bash
circup install -r requirements.txt
```

### Export installed libraries to requirements.txt
```bash
circup freeze -r
```

### Uninstall a library
```bash
circup uninstall <library_name>
```

### Search for libraries
```bash
circup show <search_term>
```

Example:
```bash
circup show bme
```

## Flags

| Flag | Description |
|------|-------------|
| `--path <path>` | Specify custom CIRCUITPY mount path |
| `--py` | Install .py source instead of compiled .mpy |
| `--verbose` | Show detailed logs |
| `--version` | Show circup version |

## Workflow: Setting Up a New Device

1. Connect device (should mount as CIRCUITPY)
2. Auto-install dependencies: `circup install --auto`
3. Or install specific libraries: `circup install <lib1> <lib2>`
4. Verify: `circup show`

## Workflow: Keeping Libraries Updated

1. Check for updates: `circup list`
2. Update all: `circup update`

## Multiple Devices

Use `--path` to target a specific device:
```bash
circup --path /Volumes/CIRCUITPY2 install adafruit_neopixel
```

## Library Sources

circup pulls from:
- [Adafruit CircuitPython Bundle](https://github.com/adafruit/Adafruit_CircuitPython_Bundle/releases/latest)
- [CircuitPython Community Bundle](https://github.com/adafruit/CircuitPython_Community_Bundle/releases/latest)
