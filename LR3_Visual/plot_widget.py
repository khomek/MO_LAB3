import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets


class PlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self._contours = None  # Для хранения контуров

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_optimization(self, history, func):
        """Визуализация процесса оптимизации"""
        # Полная очистка графика
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if len(history) > 0:
            # Автоматическое определение границ
            x_coords = [p[0] for p in history]
            y_coords = [p[1] for p in history]
            x_margin = (max(x_coords) - min(x_coords)) * 0.2
            y_margin = (max(y_coords) - min(y_coords)) * 0.2

            x_min = min(x_coords) - x_margin
            x_max = max(x_coords) + x_margin
            y_min = min(y_coords) - y_margin
            y_max = max(y_coords) + y_margin

            # Сетка для отображения функции
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])

            # Контурный график (сохраняем ссылку)
            self._contours = self.ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

            # Colorbar (создаем новый каждый раз)
            self.figure.colorbar(self._contours, ax=self.ax, label='Значение функции')

            # Путь оптимизации
            path = np.array(history)
            self.ax.plot(path[:, 0], path[:, 1], 'ro-', markersize=5, linewidth=1.5, label='Путь оптимизации')
            self.ax.scatter(path[0, 0], path[0, 1], c='green', s=100, label='Начальная точка')
            self.ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=100, label='Конечная точка')

            self.ax.set_xlabel('x1')
            self.ax.set_ylabel('x2')
            self.ax.set_title('Визуализация метода Ньютона')
            self.ax.legend()
            self.ax.grid(True)

        self.canvas.draw()