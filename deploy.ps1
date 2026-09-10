[CmdletBinding()]
param(
    [ValidateSet("Build", "Upload", "BuildAndUpload")]
    [string]$Mode = "BuildAndUpload",
    [string]$Python = "python",
    [SecureString]$PyPIToken,
    [string]$DistributionDirectory = "dist",
    [switch]$Clean,
    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -LiteralPath $RepositoryRoot

$PackageName = "dsptoolbox"
$SupportedPythonTags = @("311", "312", "313", "314")

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    Write-Host "+ $Command $($Arguments -join ' ')"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command"
    }
}

function Get-VersionFromFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $match = Select-String -LiteralPath $Path -Pattern $Pattern | Select-Object -First 1
    if ($null -eq $match) {
        throw "Could not find $Description in $Path."
    }
    return $match.Matches[0].Groups[1].Value
}

function Get-ProjectVersion {
    $pythonVersion = Get-VersionFromFile `
        -Path (Join-Path $RepositoryRoot "dsptoolbox/__init__.py") `
        -Pattern '__version__\s*=\s*["'']([^"'']+)["'']' `
        -Description "the Python package version"
    $cargoVersion = Get-VersionFromFile `
        -Path (Join-Path $RepositoryRoot "Cargo.toml") `
        -Pattern '^version\s*=\s*["'']([^"'']+)["'']' `
        -Description "the Rust crate version"

    if ($pythonVersion -ne $cargoVersion) {
        throw "Python version $pythonVersion does not match Cargo version $cargoVersion."
    }
    return $pythonVersion
}

function Get-PyPIVersion {
    $uri = "https://pypi.org/pypi/$PackageName/json"
    try {
        $response = Invoke-RestMethod -Uri $uri -Method Get
        return [string]$response.info.version
    }
    catch {
        $webResponse = $_.Exception.Response
        if ($null -ne $webResponse -and [int]$webResponse.StatusCode -eq 404) {
            return $null
        }
        throw "Could not query PyPI at $uri`: $($_.Exception.Message)"
    }
}

function Assert-VersionIsNewer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CurrentVersion,
        [string]$PublishedVersion
    )

    if ([string]::IsNullOrWhiteSpace($PublishedVersion)) {
        Write-Host "PyPI has no published version for $PackageName; treating this as the first release."
        return
    }

    $isNewer = & $Python -c `
        'from packaging.version import Version; import sys; print(Version(sys.argv[1]) > Version(sys.argv[2]))' `
        $CurrentVersion $PublishedVersion
    if ($LASTEXITCODE -ne 0 -or ($isNewer | Select-Object -Last 1).ToString().Trim() -ne "True") {
        throw "Package version $CurrentVersion is not higher than the PyPI version $PublishedVersion."
    }
    Write-Host "Current package version: $CurrentVersion"
    Write-Host "Latest PyPI version:      $PublishedVersion"
}

function Assert-ModuleAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    & $Python -c "import $ModuleName" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Python module '$ModuleName' is required. Install requirements-dev.txt first."
    }
}

function Assert-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    if ($null -eq (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Command '$CommandName' is required. Install requirements-dev.txt first."
    }
}

function Get-HostPlatform {
    if ($env:OS -eq "Windows_NT") {
        return "windows"
    }

    $uname = (& uname -s).Trim()
    switch ($uname) {
        "Darwin" { return "macos" }
        "Linux" { return "linux" }
        default { throw "Unsupported host platform: $uname" }
    }
}

function Configure-Cibuildwheel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Platform
    )

    $env:CIBW_BUILD = ($SupportedPythonTags | ForEach-Object { "cp$_-*" }) -join " "
    $env:CIBW_BUILD_VERBOSITY = "1"
    $env:CIBW_TEST_COMMAND = 'python -c "from dsptoolbox._rust import lattice_filtering_fir, warp_time_series"'

    switch ($Platform) {
        "linux" {
            $env:CIBW_ARCHS_LINUX = "x86_64 aarch64"
        }
        "macos" {
            $env:CIBW_ARCHS_MACOS = "x86_64 arm64"
        }
        "windows" {
            $env:CIBW_ARCHS_WINDOWS = "AMD64 ARM64"
        }
        default {
            throw "Unsupported cibuildwheel platform: $Platform"
        }
    }
}

function Invoke-Validation {
    if ($SkipValidation) {
        Write-Warning "Skipping pytest and Cargo validation because -SkipValidation was supplied."
        return
    }

    Invoke-Checked -Command $Python -Arguments @("-m", "pytest")
    Invoke-Checked -Command "cargo" -Arguments @("fmt", "--check")
    Invoke-Checked -Command "cargo" -Arguments @("check")
}

function Build-HostArtifacts {
    $platform = Get-HostPlatform
    Configure-Cibuildwheel -Platform $platform

    if ($Clean -and (Test-Path -LiteralPath $DistributionDirectory)) {
        Remove-Item -LiteralPath $DistributionDirectory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $DistributionDirectory -Force | Out-Null

    Write-Host "Building $platform wheels for Python $($SupportedPythonTags -join ', ')..."
    Invoke-Checked -Command $Python -Arguments @(
        "-m", "cibuildwheel",
        "--platform", $platform,
        "--output-dir", $DistributionDirectory
    )

    $existingSdist = Get-ChildItem -LiteralPath $DistributionDirectory -Filter "$PackageName-$version.tar.gz" -File -ErrorAction SilentlyContinue
    if ($null -eq $existingSdist) {
        Invoke-Checked -Command "maturin" -Arguments @(
            "sdist", "--out", $DistributionDirectory
        )
    }
}

function Get-ReleaseArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    if (-not (Test-Path -LiteralPath $DistributionDirectory)) {
        throw "Distribution directory '$DistributionDirectory' does not exist."
    }

    $versionPattern = [regex]::Escape("$PackageName-$Version")
    $targets = foreach ($pythonTag in $SupportedPythonTags) {
        @(
            @{ Label = "cp$pythonTag manylinux x86_64"; Pattern = "cp$pythonTag-.*-manylinux.*_x86_64" }
            @{ Label = "cp$pythonTag manylinux aarch64"; Pattern = "cp$pythonTag-.*-manylinux.*_aarch64" }
            @{ Label = "cp$pythonTag macOS x86_64"; Pattern = "cp$pythonTag-.*-macosx.*_x86_64" }
            @{ Label = "cp$pythonTag macOS arm64"; Pattern = "cp$pythonTag-.*-macosx.*_arm64" }
            @{ Label = "cp$pythonTag Windows AMD64"; Pattern = "cp$pythonTag-.*-win_amd64" }
            @{ Label = "cp$pythonTag Windows ARM64"; Pattern = "cp$pythonTag-.*-win_arm64" }
        )
    }

    $artifacts = [System.Collections.Generic.List[string]]::new()
    foreach ($target in $targets) {
        $pattern = "^$versionPattern-$($target.Pattern)\.whl$"
        $matches = @(Get-ChildItem -LiteralPath $DistributionDirectory -Filter "*.whl" -File |
            Where-Object { $_.Name -match $pattern })
        if ($matches.Count -ne 1) {
            throw "Expected exactly one wheel for $($target.Label), found $($matches.Count). Build all native platform matrices before uploading."
        }
        [void]$artifacts.Add($matches[0].FullName)
    }

    $sdists = @(Get-ChildItem -LiteralPath $DistributionDirectory -Filter "$PackageName-$Version.tar.gz" -File)
    if ($sdists.Count -ne 1) {
        throw "Expected exactly one source distribution '$PackageName-$Version.tar.gz', found $($sdists.Count)."
    }
    [void]$artifacts.Add($sdists[0].FullName)

    return $artifacts.ToArray()
}

function ConvertTo-PlainText {
    param(
        [Parameter(Mandatory = $true)]
        [SecureString]$SecureValue
    )

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Upload-Release {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Artifacts
    )

    Invoke-Checked -Command $Python -Arguments (@("-m", "twine", "check") + $Artifacts)

    if ($null -eq $PyPIToken) {
        $script:PyPIToken = Read-Host -Prompt "Enter the PyPI API token" -AsSecureString
    }
    $plainToken = ConvertTo-PlainText -SecureValue $PyPIToken
    $previousUsername = $env:TWINE_USERNAME
    $previousPassword = $env:TWINE_PASSWORD
    $env:TWINE_USERNAME = "__token__"
    $env:TWINE_PASSWORD = $plainToken

    try {
        Invoke-Checked -Command $Python -Arguments (@(
            "-m", "twine", "upload",
            "--non-interactive",
            "--repository-url", "https://upload.pypi.org/legacy/"
        ) + $Artifacts)
    }
    finally {
        if ($null -eq $previousUsername) {
            Remove-Item Env:TWINE_USERNAME -ErrorAction SilentlyContinue
        }
        else {
            $env:TWINE_USERNAME = $previousUsername
        }
        if ($null -eq $previousPassword) {
            Remove-Item Env:TWINE_PASSWORD -ErrorAction SilentlyContinue
        }
        else {
            $env:TWINE_PASSWORD = $previousPassword
        }
        $plainToken = $null
    }
}

$version = Get-ProjectVersion
$publishedVersion = Get-PyPIVersion
Assert-VersionIsNewer -CurrentVersion $version -PublishedVersion $publishedVersion

if ($Mode -ne "Upload") {
    Assert-ModuleAvailable -ModuleName "cibuildwheel"
    Assert-CommandAvailable -CommandName "maturin"
    if (-not $SkipValidation) {
        Invoke-Validation
    }
    Build-HostArtifacts
}

if ($Mode -eq "Build") {
    Write-Host "Host artifacts built in '$DistributionDirectory'. Run this script on the remaining native hosts, then use -Mode Upload."
    exit 0
}

$releaseArtifacts = Get-ReleaseArtifacts -Version $version
Write-Host "Validated $($releaseArtifacts.Count) release artifacts for $PackageName $version."
Upload-Release -Artifacts $releaseArtifacts
Write-Host "Uploaded $PackageName $version to PyPI."
