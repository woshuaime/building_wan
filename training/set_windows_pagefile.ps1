$ErrorActionPreference = 'Stop'

$resultPath = 'D:\code\Python_3D_Scanner\outputs\pagefile_change_result.txt'
$memoryManagement = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$pagingFiles = @(
    'C:\pagefile.sys 4096 8192',
    'D:\pagefile.sys 65536 98304'
)

$computerSystem = Get-CimInstance -ClassName Win32_ComputerSystem
Set-CimInstance -InputObject $computerSystem -Property @{ AutomaticManagedPagefile = $false } | Out-Null
New-ItemProperty -Path $memoryManagement -Name PagingFiles -PropertyType MultiString -Value $pagingFiles -Force | Out-Null

$configured = (Get-ItemProperty -Path $memoryManagement -Name PagingFiles).PagingFiles
@(
    "changed_at=$(Get-Date -Format o)"
    'automatic_managed_pagefile=false'
    "paging_files=$($configured -join ';')"
    'restart_required=true'
) | Set-Content -LiteralPath $resultPath -Encoding utf8
