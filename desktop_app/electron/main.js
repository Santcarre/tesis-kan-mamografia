const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const http = require('http');
const { spawn, execSync } = require('child_process');
const fs = require('fs');

const rootDir = path.resolve(__dirname, '..', '..');
let pyProc = null;
let splashWindow = null;
let mainWindow = null;
let isQuitting = false;

// Resolve Python executable path
function resolvePythonPath() {
  const winVenv = path.join(rootDir, '.venv', 'Scripts', 'python.exe');
  const unixVenv = path.join(rootDir, '.venv', 'bin', 'python');

  if (process.platform === 'win32' && fs.existsSync(winVenv)) {
    return winVenv;
  }
  if (fs.existsSync(unixVenv)) {
    return unixVenv;
  }
  if (process.env.PYTHON_EXECUTABLE && fs.existsSync(process.env.PYTHON_EXECUTABLE)) {
    return process.env.PYTHON_EXECUTABLE;
  }
  return 'python';
}

// Kill the spawned Python process cleanly to guarantee Zero Orphan Processes
function killPythonProcess() {
  if (pyProc && !pyProc.killed) {
    const pid = pyProc.pid;
    console.log(`[Lifecycle] Finalizando subproceso Python (PID: ${pid})...`);
    if (process.platform === 'win32') {
      try {
        execSync(`taskkill /PID ${pid} /T /F`);
      } catch (e) {
        console.error("taskkill error:", e.message || e);
      }
    } else {
      try {
        pyProc.kill('SIGTERM');
      } catch (e) {
        console.error("SIGTERM error:", e.message || e);
      }
    }
    pyProc = null;
  }
}

// Spawn FastAPI backend server via uvicorn
function startPythonBackend() {
  const pythonPath = resolvePythonPath();
  const uvicornArgs = [
    "-m", "uvicorn",
    "desktop_app.backend.app:app",
    "--host", "127.0.0.1",
    "--port", "8000"
  ];

  console.log(`[Lifecycle] Iniciando backend con: ${pythonPath} ${uvicornArgs.join(' ')}`);
  pyProc = spawn(pythonPath, uvicornArgs, {
    cwd: rootDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: "1" }
  });

  pyProc.stdout.on('data', (data) => {
    console.log(`[FastAPI stdout]: ${data.toString().trim()}`);
  });

  pyProc.stderr.on('data', (data) => {
    console.error(`[FastAPI stderr]: ${data.toString().trim()}`);
  });

  pyProc.on('close', (code) => {
    console.log(`[FastAPI] Proceso terminado con código: ${code}`);
    if (!isQuitting && mainWindow) {
      dialog.showErrorBox(
        "Error en Backend",
        `El servidor FastAPI se detuvo inesperadamente con código ${code}.`
      );
    }
  });

  pyProc.on('error', (err) => {
    console.error("[FastAPI error al iniciar]:", err);
    dialog.showErrorBox(
      "Error al Iniciar Backend",
      `No se pudo iniciar el servidor Python: ${err.message}`
    );
  });
}

// Create the 500x320 frameless splash screen
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 500,
    height: 320,
    frame: false,
    resizable: false,
    center: true,
    show: false,
    alwaysOnTop: true,
    backgroundColor: '#0f172a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  splashWindow.loadFile(path.join(__dirname, 'splash.html'));

  splashWindow.once('ready-to-show', () => {
    splashWindow.show();
  });
}

// Poll http://127.0.0.1:8000/api/health until ready or timeout (25s)
function pollBackendHealth(callback) {
  const healthUrl = 'http://127.0.0.1:8000/api/health';
  const startTime = Date.now();
  const timeoutMs = 25000;
  const pollInterval = 500;

  function check() {
    if (isQuitting) return;

    http.get(healthUrl, (res) => {
      if (res.statusCode === 200) {
        console.log(`[HealthCheck] Backend listo en ${Date.now() - startTime}ms`);
        callback(true);
      } else {
        retry();
      }
    }).on('error', () => {
      retry();
    });
  }

  function retry() {
    if (Date.now() - startTime > timeoutMs) {
      console.error(`[HealthCheck] Timeout agotado tras ${timeoutMs}ms.`);
      callback(false);
    } else {
      setTimeout(check, pollInterval);
    }
  }

  check();
}

// Create the primary PACS radiological workstation window
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 920,
    minWidth: 1200,
    minHeight: 750,
    show: false,
    backgroundColor: '#F8FAFC',
    title: 'Prometeo PACS v2.0 — Estación Radiológica Neuro-Simbólica',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Limpia la caché antes de cargar: evita ver una versión anterior de la interfaz.
  mainWindow.webContents.session.clearCache().finally(() => {
    if (mainWindow) mainWindow.loadURL('http://127.0.0.1:8000/ui/index.html');
  });

  mainWindow.once('ready-to-show', () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    isQuitting = true;
    killPythonProcess();
    app.quit();
  });
}

// App lifecycle orchestration
app.whenReady().then(() => {
  createSplashWindow();
  startPythonBackend();

  pollBackendHealth((isHealthy) => {
    if (isHealthy) {
      createMainWindow();
    } else {
      if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.close();
        splashWindow = null;
      }
      dialog.showErrorBox(
        "Fallo de Inicialización",
        "El servidor FastAPI de Prometeo no respondió en el tiempo límite (25s).\nVerifique la instalación de Python y dependencias."
      );
      isQuitting = true;
      killPythonProcess();
      app.quit();
    }
  });
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on('before-quit', () => {
  isQuitting = true;
  killPythonProcess();
});

app.on('will-quit', () => {
  isQuitting = true;
  killPythonProcess();
});

process.on('SIGINT', () => {
  isQuitting = true;
  killPythonProcess();
  process.exit(0);
});

process.on('SIGTERM', () => {
  isQuitting = true;
  killPythonProcess();
  process.exit(0);
});

process.on('exit', () => {
  killPythonProcess();
});
