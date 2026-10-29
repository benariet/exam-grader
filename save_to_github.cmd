@echo off
rem Save the project to GitHub: every change (add + commit with the date) and push - with the sign-in Windows already keeps.
chcp 65001 >nul
cd /d "%~dp0"
git add -A
git commit -m "Update %date% %time:~0,5%"
git push
echo.
git log -1 --oneline
git status -sb | findstr /b "##"
echo.
echo Done - you can close this window.
pause
