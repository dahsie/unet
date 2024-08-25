import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import UNet
from torch.nn import Module

class UNetTrainer:

    def __init__(self, model: Module, train_loader: DataLoader, test_loader: DataLoader, loss_func, optimizer, scheduler, device: str)-> None:
        """
        Initialiser l'entraîneur avec le modèle, les loaders de données, la fonction de perte,
        l'optimiseur, le scheduler, et le dispositif (CPU ou GPU).
        
        Parameters:
        -----------
        model : torch.nn.Module
            Le modèle UNet à entraîner.
        train_loader : torch.utils.data.DataLoader
            Le DataLoader pour les données d'entraînement.
        test_loader : torch.utils.data.DataLoader
            Le DataLoader pour les données de validation.
        loss_func : callable
            La fonction de perte utilisée pour l'entraînement.
        optimizer : torch.optim.Optimizer
            L'optimiseur utilisé pour la mise à jour des poids du modèle.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Le scheduler utilisé pour ajuster le taux d'apprentissage au fil du temps.
        device : torch.device
            L'appareil sur lequel le modèle et les données seront chargés.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_step(self, x, y):
        """
        Effectuer une seule étape d'entraînement : forward pass, calcul de la perte,
        rétropropagation et mise à jour des poids.

        Parameters:
        -----------
        x : torch.Tensor
            Les données d'entrée.
        y : torch.Tensor
            Les étiquettes correspondantes.

        Returns:
        --------
        loss : float
            La perte calculée pour cette étape.
        """
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass
        pred = self.model(x)
        loss = self.loss_func(pred, y)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, x, y):
        """
        Effectuer une seule étape d'évaluation : forward pass et calcul de la perte.

        Parameters:
        -----------
        x : torch.Tensor
            Les données d'entrée.
        y : torch.Tensor
            Les étiquettes correspondantes.

        Returns:
        --------
        loss : float
            La perte calculée pour cette étape.
        """
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        pred = self.model(x)
        loss = self.loss_func(pred, y)

        return loss.item()

    def train(self, num_epochs):
        """
        Entraîner le modèle UNet sur un certain nombre d'époques.

        Parameters:
        -----------
        num_epochs : int
            Le nombre d'époques d'entraînement.
        
        Returns:
        --------
        history : dict
            Un dictionnaire contenant les pertes moyennes pour chaque époque pour l'entraînement et la validation.
        """
        history = {"train_loss": [], "test_loss": []}
        start_time = time.time()

        for epoch in range(num_epochs):
            # Mode entraînement
            self.model.train()
            total_train_loss = 0

            for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss = self.train_step(x, y)
                total_train_loss += loss

            avg_train_loss = total_train_loss / len(self.train_loader)

            # Mode évaluation
            self.model.eval()
            total_test_loss = 0

            with torch.no_grad():
                for x, y in self.test_loader:
                    loss = self.eval_step(x, y)
                    total_test_loss += loss

            avg_test_loss = total_test_loss / len(self.test_loader)

            # Enregistrer l'historique des pertes
            history["train_loss"].append(avg_train_loss)
            history["test_loss"].append(avg_test_loss)

            

            # Afficher les informations d'entraînement
            print("Epoch %d: SGD lr=%.4f" % (epoch, self.optimizer.param_groups[0]["lr"]))
            print(f"Train loss: {avg_train_loss:.6f}, Test loss: {avg_test_loss:.4f}")
        

            # Scheduler step
            self.scheduler.step()

        # Temps total d'entraînement
        end_time = time.time()
        print(f"[INFO] Total time taken to train the model: {end_time - start_time:.2f}s")

        return history
