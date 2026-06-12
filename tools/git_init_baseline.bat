@echo off
REM ============================================================
REM Initialize both v2 trees as git repos with a baseline commit.
REM Run ONCE from the v2 dir. After running, add a private remote
REM and push if desired.
REM ============================================================
setlocal enabledelayedexpansion

set "V2_DIR=%~dp0.."
set "RELEASE_DIR=F:\faceprodpbraw\faceswap_pro_staging\v2_github_release"

echo.
echo === Initializing v2 working tree ===
echo Path: %V2_DIR%
pushd "%V2_DIR%"

if exist .git (
    echo .git already exists -- skipping init
) else (
    git init
    if errorlevel 1 (
        echo ERROR: git init failed in v2
        popd
        exit /b 1
    )
)

git config user.email "kingkush@gmail.com" 2>nul
git config user.name "kk" 2>nul
git add -A
git commit -m "baseline 2026-06-11 -- pre-roadmap-tier0" 2>nul
git tag -f baseline-2026-06-11
echo v2 baseline committed.

popd

echo.
echo === Initializing v2_github_release tree ===
echo Path: %RELEASE_DIR%

if not exist "%RELEASE_DIR%" (
    echo WARN: release tree not found at %RELEASE_DIR% -- skipping
    goto :done
)

pushd "%RELEASE_DIR%"

if exist .git (
    echo .git already exists -- skipping init
) else (
    git init
    if errorlevel 1 (
        echo ERROR: git init failed in release tree
        popd
        exit /b 1
    )
)

git config user.email "kingkush@gmail.com" 2>nul
git config user.name "kk" 2>nul
git add -A
git commit -m "baseline 2026-06-11 -- pre-roadmap-tier0" 2>nul
git tag -f baseline-2026-06-11
echo release tree baseline committed.

popd

:done
echo.
echo ============================================================
echo Both trees initialized.  To push to a private remote, run:
echo.
echo   cd %V2_DIR%
echo   git remote add origin git@github.com:YOURUSER/faceswap_pro_v2_private.git
echo   git push -u origin master --tags
echo.
echo   cd %RELEASE_DIR%
echo   git remote add origin git@github.com:YOURUSER/faceswap_pro_v2_release.git
echo   git push -u origin master --tags
echo ============================================================
endlocal
