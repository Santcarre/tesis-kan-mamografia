"""
Trainer: Lógica de entrenamiento en dos fases para HybridResNetKAN.

Fase 1 (solo KAN):      8 épocas, Adam lr=1e-3, wd=1e-4, EarlyStopping(patience=4)
Fase 2 (layer4 + KAN): 10 épocas, Adam lr_kan=5e-4 / lr_backbone=1e-5
"""
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import get_weighted_loss
from .schedulers import get_scheduler, EarlyStopping


SEED = 42


class Trainer:
    """Entrena HybridResNetKAN siguiendo el protocolo de 2 fases.

    Args:
        model: Instancia de HybridResNetKAN.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        device: 'cuda' o 'cpu'.
        class_weights: Tensor con pesos de clase para CrossEntropyLoss.
        output_dir: Directorio donde guardar checkpoints e historial.
        model_name: Nombre base del archivo de salida (.pt).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        class_weights: torch.Tensor = None,
        output_dir: str = 'reports/models',
        model_name: str = 'resnet18_kan_hybrid_gap32',
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.output_dir   = output_dir
        self.model_name   = model_name
        self.criterion    = get_weighted_loss(class_weights, device)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Entrenamiento de una época
    # ------------------------------------------------------------------

    def _l1_sparsity_penalty(self, lamb: float = 0.0) -> torch.Tensor:
        """Penalización L1 sobre las activaciones spline cacheadas de la KAN.

        Tras un forward, pykan deja disponibles `spline_preacts` y
        `spline_postacts`. Usamos las activaciones post-spline medias
        (por capa) para penalizar aristas no nulas y favorecer esparsidad.

        Args:
            lamb (float): Coeficiente de la penalidad L1. Si 0, retorna 0.

        Returns:
            torch.Tensor (escalar): término L1 para sumar al loss.
        """
        if lamb is None or lamb <= 0:
            return torch.tensor(0.0, device=self.device)
        kan = getattr(self.model, 'kan_head', self.model).kan
        tot = torch.tensor(0.0, device=self.device)
        n = 0
        for l in range(len(kan.spline_postacts)):
            acts = kan.spline_postacts[l]          # [B, out, in]
            if acts is None:
                continue
            tot = tot + acts.abs().mean()
            n += 1
        if n == 0:
            return torch.tensor(0.0, device=self.device)
        return lamb * tot

    def _train_epoch(self, optimizer, grid_update: bool = False, grid_batches: int = 8, lamb: float = 0.0):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        skipped = 0

        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, leave=False)):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # Actualización de rejilla B-spline (primeros N batches)
            if grid_update and batch_idx < grid_batches:
                with torch.no_grad():
                    emb = self.model.backbone(images)
                    snap = self._snapshot_splines()
                self.model.kan_head.update_grid(emb)
                # Si la rejilla genera coeficientes NaN/Inf, se restaura la
                # rejilla anterior y se desactiva el grid update (causa del crash
                # en la reanudación del notebook 09, época 9).
                if not self._splines_finite():
                    print(f"  [warn] grid update -> NaN en batch {batch_idx}; rejilla restaurada y grid update desactivado")
                    self._restore_splines(snap)
                    grid_update = False

            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            if lamb and lamb > 0:
                loss = loss + self._l1_sparsity_penalty(lamb)

            # Guard anti-NaN: un batch corrompido no debe envenenar los pesos
            if not torch.isfinite(loss):
                skipped += 1
                if skipped == 1:
                    print(f"  [warn] loss no finita en batch {batch_idx}; se omite el batch")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        if total == 0:
            return float('nan'), 0.0
        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _val_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += images.size(0)

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import roc_auc_score
        import numpy as _np
        all_probs = _np.asarray(all_probs, dtype=float)
        all_labels = _np.asarray(all_labels, dtype=float)
        finite = _np.isfinite(all_probs) & _np.isfinite(all_labels)
        if not finite.all():
            n_bad = int((~finite).sum())
            print(f"  [warn] {n_bad} muestras con NaN/Inf en val; se descartan")
            all_probs = all_probs[finite]
            all_labels = all_labels[finite]
        if len(all_labels) == 0 or len(_np.unique(all_labels)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(all_labels, all_probs)
        return total_loss / total, correct / total, auc

    # ------------------------------------------------------------------
    # Utilidades de seguridad (NaN) y persistencia de estado completo
    # ------------------------------------------------------------------

    def _snapshot_splines(self) -> dict:
        """Copia rejilla/coef de las B-splines para poder hacer rollback."""
        return {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if ('grid' in n or 'coef' in n)
        }

    def _splines_finite(self) -> bool:
        """True si todas las rejillas/coef B-spline son finitos."""
        for n, p in self.model.named_parameters():
            if ('grid' in n or 'coef' in n) and not bool(torch.isfinite(p).all().item()):
                return False
        return True

    def _restore_splines(self, snap: dict):
        """Restaura rejilla/coef B-spline desde un snapshot previo."""
        with torch.no_grad():
            for name, t in snap.items():
                for n, p in self.model.named_parameters():
                    if n == name:
                        p.copy_(t)
                        break

    def _save_history_csv(self, history: list):
        """Escribe el historial CSV completo (se llama tras cada época)."""
        if not history:
            return
        csv_path = os.path.join(self.output_dir, f"{self.model_name}_history.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _save_full_state(self, path, phase, epoch, optimizer, scheduler, es, best_auc, history):
        """Guarda el estado completo (modelo + optimizer + scheduler + ES + historial)."""
        torch.save({
            'phase': phase,
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'es_state': es.__dict__,
            'best_auc': best_auc,
            'history': history,
        }, path)

    # ------------------------------------------------------------------
    # Entrenamiento completo 2 fases
    # ------------------------------------------------------------------

    def train(
        self,
        phase1_epochs: int = 8,
        phase2_epochs: int = 10,
        phase1_lr: float   = 1e-3,
        phase2_lr_kan: float    = 5e-4,
        phase2_lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        early_stop_patience: int = 4,
        grid_update_enabled: bool = True,
        lamb: float = 0.0,
        run_lbfgs: bool = False,
        lbfgs_lr: float = 0.3,
        resume_checkpoint: str = None,
        resume_phase: int = None,
        resume_epoch: int = None,
        initial_history: list = None,
        initial_best_auc: float = None,
    ):
        """Ejecuta el entrenamiento completo en 2 fases, con soporte de reanudación.

        Args:
            phase1_epochs: Épocas Fase 1.
            phase2_epochs: Épocas Fase 2.
            phase1_lr: LR Adam Fase 1.
            phase2_lr_kan: LR KAN Fase 2.
            phase2_lr_backbone: LR layer4 Fase 2.
            weight_decay: Regularización L2.
            early_stop_patience: Paciencia para EarlyStopping.
            grid_update_enabled: Actualizar rejilla B-spline.
            run_lbfgs: Ejecutar refinamiento L-BFGS al final.
            lbfgs_lr: Learning rate de L-BFGS.
            resume_checkpoint: Ruta de checkpoint. Si es de estado completo
                (contiene 'model_state') restaura también optimizer/scheduler/
                early-stopping. Si solo tiene pesos, usa resume_phase/resume_epoch.
            resume_phase: Fase a reanudar (1 o 2) para checkpoints de solo pesos.
            resume_epoch: Última época completada de resume_phase.
            initial_history: Épocas previas a conservar en el historial/CSV.
            initial_best_auc: AUC base para no sobrescribir el mejor checkpoint.
        """
        state_path = os.path.join(self.output_dir, f"{self.model_name}_train_state.pt")
        history    = list(initial_history) if initial_history else []
        best_auc   = initial_best_auc if initial_best_auc is not None else float('-inf')

        next_p1, next_p2 = 1, 1
        cur_phase  = resume_phase or 1
        opt_state = sched_state = es_state = None

        # ------------------------------------------------------------------
        # Restauración del estado (completo o solo pesos)
        # ------------------------------------------------------------------
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            try:
                raw = torch.load(resume_checkpoint, map_location=self.device)
            except Exception as e:
                raw = None
                print(f"  [warn] no se pudo leer el checkpoint {resume_checkpoint}: {e}")
            if isinstance(raw, dict) and 'model_state' in raw:
                # Checkpoint de estado completo (post-entrenamiento robusto)
                self.model.load_state_dict(raw['model_state'])
                history   = list(raw.get('history', []))
                best_auc  = raw.get('best_auc', float('-inf'))
                cur_phase = raw.get('phase', 1)
                last_ep   = raw.get('epoch', 0)
                if cur_phase == 1:
                    next_p1 = last_ep + 1
                else:
                    next_p2 = last_ep + 1
                opt_state, sched_state, es_state = (
                    raw.get('optimizer_state'), raw.get('scheduler_state'), raw.get('es_state'))
                print(f"  Reanudación de estado completo: Fase {cur_phase}, "
                      f"continúa en época {last_ep + 1} (best_auc={best_auc:.4f})")
            else:
                # Solo pesos (caso legacy: crash anterior)
                if raw is not None:
                    self.model.load_state_dict(raw)
                cur_phase = resume_phase or 1
                last_ep   = resume_epoch or 0
                if cur_phase == 1:
                    next_p1 = last_ep + 1
                else:
                    next_p2 = last_ep + 1
                print(f"  Reanudación solo-pesos: Fase {cur_phase}, continúa en época {last_ep + 1}")

        # ------------------------------------------------------------------
        # FASE 1: Solo KAN
        # ------------------------------------------------------------------
        if cur_phase == 1:
            print("\n=== FASE 1: Entrenamiento cabezal KAN (backbone congelado) ===")
            self.model.set_phase1()
            opt1  = torch.optim.Adam(self.model.kan_head.parameters(), lr=phase1_lr, weight_decay=weight_decay)
            sched1 = get_scheduler(opt1)
            es1   = EarlyStopping(patience=early_stop_patience)
            if opt_state is not None:
                opt1.load_state_dict(opt_state)
                sched1.load_state_dict(sched_state)
                if es_state:
                    es1.best, es1.counter, es1.stop = es_state['best'], es_state['counter'], es_state['stop']

            for epoch in range(next_p1, phase1_epochs + 1):
                tr_loss, tr_acc  = self._train_epoch(opt1, grid_update=grid_update_enabled, lamb=lamb)
                vl_loss, vl_acc, vl_auc = self._val_epoch()
                sched1.step(vl_auc)
                row = {'phase': 1, 'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': tr_acc,
                       'vl_loss': vl_loss, 'vl_acc': vl_acc, 'vl_auc': vl_auc}
                history.append(row)
                print(f"  Época {epoch:02d} | Loss {tr_loss:.4f}/{vl_loss:.4f} | Acc {tr_acc:.4f}/{vl_acc:.4f} | AUC {vl_auc:.4f}")
                best_path = os.path.join(self.output_dir, f"{self.model_name}.pt")
                if vl_auc > best_auc:
                    best_auc = vl_auc
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  ✓ Checkpoint guardado (AUC={best_auc:.4f})")
                self._save_full_state(state_path, 1, epoch, opt1, sched1, es1, best_auc, history)
                self._save_history_csv(history)
                if es1(vl_auc):
                    print(f"  EarlyStopping en época {epoch}")
                    break

            # Continúa a Fase 2 desde cero
            next_p2 = 1
            opt_state = sched_state = es_state = None

        # ------------------------------------------------------------------
        # FASE 2: layer4 + KAN
        # ------------------------------------------------------------------
        print("\n=== FASE 2: Fine-tuning layer4 + KAN ===")
        self.model.set_phase2()
        param_groups = self.model.get_phase2_param_groups(lr_kan=phase2_lr_kan, lr_backbone=phase2_lr_backbone)
        opt2   = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        sched2 = get_scheduler(opt2)
        es2    = EarlyStopping(patience=early_stop_patience)
        if opt_state is not None:
            opt2.load_state_dict(opt_state)
            sched2.load_state_dict(sched_state)
            if es_state:
                es2.best, es2.counter, es2.stop = es_state['best'], es_state['counter'], es_state['stop']

        for epoch in range(next_p2, phase2_epochs + 1):
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            tr_loss, tr_acc  = self._train_epoch(opt2, grid_update=grid_update_enabled, lamb=lamb)
            vl_loss, vl_acc, vl_auc = self._val_epoch()
            sched2.step(vl_auc)
            row = {'phase': 2, 'epoch': epoch, 'tr_loss': tr_loss, 'tr_acc': tr_acc,
                   'vl_loss': vl_loss, 'vl_acc': vl_acc, 'vl_auc': vl_auc}
            history.append(row)
            print(f"  Época {epoch:02d} | Loss {tr_loss:.4f}/{vl_loss:.4f} | Acc {tr_acc:.4f}/{vl_acc:.4f} | AUC {vl_auc:.4f}")
            best_path = os.path.join(self.output_dir, f"{self.model_name}.pt")
            if vl_auc > best_auc:
                best_auc = vl_auc
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ Checkpoint guardado (AUC={best_auc:.4f})")
            self._save_full_state(state_path, 2, epoch, opt2, sched2, es2, best_auc, history)
            self._save_history_csv(history)
            if es2(vl_auc):
                print(f"  EarlyStopping en época {epoch}")
                break

        # ------------------------------------------------------------------
        # Opcional: refinamiento L-BFGS
        # ------------------------------------------------------------------
        if run_lbfgs:
            print("\n=== Refinamiento L-BFGS (solo KAN) ===")
            self._run_lbfgs(lr=lbfgs_lr, max_iter=20, n_batches=6, n_steps=8)

        self._save_history_csv(history)
        print(f"\nHistorial guardado en {os.path.join(self.output_dir, self.model_name + '_history.csv')}")
        print(f"Mejor AUC validación: {best_auc:.4f}")
        return history

    def _run_lbfgs(self, lr: float = 0.3, max_iter: int = 20, n_batches: int = 6, n_steps: int = 8):
        """Refinamiento L-BFGS sobre los parámetros de la KAN."""
        kan_params = list(self.model.kan_head.parameters())
        opt_lbfgs  = torch.optim.LBFGS(kan_params, lr=lr, max_iter=max_iter)
        data_iter  = iter(self.train_loader)

        for step in range(n_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                images, labels = next(data_iter)
            images, labels = images.to(self.device), labels.to(self.device)

            def closure():
                opt_lbfgs.zero_grad()
                logits = self.model(images)
                loss   = self.criterion(logits, labels)
                loss.backward()
                return loss

            loss = opt_lbfgs.step(closure)
            print(f"  L-BFGS paso {step+1}/{n_steps} | Loss: {loss:.4f}")
