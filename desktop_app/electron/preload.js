const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('prometeoDesktop', {
  version: '2.0.0',
  platform: process.platform,
  isElectron: true,
  ping: () => ipcRenderer.invoke('ping')
});
