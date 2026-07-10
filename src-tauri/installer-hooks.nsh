; The bundled Python sidecar loads native modules from the installation
; directory. Windows refuses to overwrite or remove those modules while the
; process is alive, so stop any orphaned sidecar before NSIS touches files.
;
; The normal in-app update path already stops and waits for its child process
; through `prepare_for_update`; these hooks are the fallback for manual installs,
; upgrades from older versions, crashes, and uninstall operations.

!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Stopping PPT translation background service..."
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM "ppt-translator-sidecar.exe"'
  Sleep 500
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  DetailPrint "Stopping PPT translation background service..."
  nsExec::ExecToLog '"$SYSDIR\taskkill.exe" /F /T /IM "ppt-translator-sidecar.exe"'
  Sleep 500
!macroend
