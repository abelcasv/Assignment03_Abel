param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MemoryStatus {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class MEMORYSTATUSEX {
        public uint dwLength;
        public uint dwMemoryLoad;
        public ulong ullTotalPhys;
        public ulong ullAvailPhys;
        public ulong ullTotalPageFile;
        public ulong ullAvailPageFile;
        public ulong ullTotalVirtual;
        public ulong ullAvailVirtual;
        public ulong ullAvailExtendedVirtual;
        public MEMORYSTATUSEX() { dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX)); }
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX status);
}
'@

$memory = [MemoryStatus+MEMORYSTATUSEX]::new()
if (-not [MemoryStatus]::GlobalMemoryStatusEx($memory)) {
    throw 'GlobalMemoryStatusEx failed.'
}

$cpu = (Get-Item 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0').GetValue('ProcessorNameString')
$os = [System.Environment]::OSVersion.Version
$gcc = (& 'C:\msys64\ucrt64\bin\gcc.exe' --version | Select-Object -First 1)
$report = @(
    "capture_timestamp: $([DateTimeOffset]::Now.ToString('o'))"
    "processor: $cpu"
    "logical_processors_visible_to_process: $([Environment]::ProcessorCount)"
    "physical_cores: 10 (Intel product specification; 2 P-cores + 8 E-cores)"
    "physical_memory_total_gib: $([math]::Round($memory.ullTotalPhys / 1GB, 2))"
    "physical_memory_available_at_capture_gib: $([math]::Round($memory.ullAvailPhys / 1GB, 2))"
    "windows_version: $($os.ToString())"
    "windows_build: $($os.Build)"
    "powershell_version: $($PSVersionTable.PSVersion)"
    "gcc_version: $gcc"
    'memory_module_type_and_speed: not captured'
    'affinity: no explicit OpenMP place binding (OMP_PROC_BIND=false; OMP_PLACES unset)'
)
$destination = Join-Path $ProjectRoot 'results/environment_host.txt'
$report | Set-Content -Path $destination -Encoding utf8
Write-Output "Wrote $destination"
