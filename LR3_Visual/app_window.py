from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QTextEdit, QMessageBox)  #Import QMessageBox
from PyQt5.QtCore import Qt
from plot_widget import PlotWidget
from optimization import Optimization
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.optimizer = Optimization()
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Создание интерфейса"""
        self.setWindowTitle("Метод Ньютона с визуализацией")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель с контролами
        control_panel = QGroupBox("Параметры оптимизации")
        control_layout = QVBoxLayout()
        control_panel.setMinimumWidth(400)

        # Поля ввода
        self.inputs = {
            'x1': self.create_input_field(control_layout, "Первое значение начального x0:", "1.5"),
            'x2': self.create_input_field(control_layout, "Второе значение начального x0:", "0.5"),
            'eps1': self.create_input_field(control_layout, "Точность eps1:", "0.15"),
            'eps2': self.create_input_field(control_layout, "Точность eps2:", "0.2"),
            'max_iter': self.create_input_field(control_layout, "Макс. итераций:", "10")
        }

        # Кнопки
        self.run_btn = QPushButton("Запустить оптимизацию")
        self.clear_btn = QPushButton("Очистить результаты")
        control_layout.addWidget(self.run_btn)
        control_layout.addWidget(self.clear_btn)

        # Область вывода результатов
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(300)
        self.result_display.setLineWrapMode(QTextEdit.NoWrap)
        control_layout.addWidget(QLabel("Результаты:"))
        control_layout.addWidget(self.result_display)

        control_panel.setLayout(control_layout)

        # Правая панель с графиком
        self.plot_widget = PlotWidget()

        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.plot_widget, stretch=1)

    def create_input_field(self, layout, label_text, default_value=""):
        """Создание поля ввода с меткой"""
        layout.addWidget(QLabel(label_text))
        line_edit = QLineEdit(default_value)
        layout.addWidget(line_edit)
        return line_edit

    def connect_signals(self):
        """Подключение обработчиков событий"""
        self.run_btn.clicked.connect(self.run_optimization)
        self.clear_btn.clicked.connect(self.clear_results)

    def run_optimization(self):
        """Запуск оптимизации и отображение результатов"""
        try:
            # Получаем параметры из полей ввода
            try:
                x1 = float(self.inputs['x1'].text())
                x2 = float(self.inputs['x2'].text())
                eps1 = float(self.inputs['eps1'].text())
                eps2 = float(self.inputs['eps2'].text())
                max_iter = int(self.inputs['max_iter'].text())

                if eps1 <= 0 or eps2 <= 0 or max_iter <= 0:
                    QMessageBox.critical(self, "Ошибка ввода", "Точность (eps1, eps2) и Макс. итераций должны быть положительными.")
                    return

            except ValueError:
                QMessageBox.critical(self, "Ошибка ввода", "Пожалуйста, введите числа в поля ввода.")
                return

            params = {
                'x0': np.array([x1, x2]),
                'eps1': eps1,
                'eps2': eps2,
                'M': max_iter
            }

            # Запускаем оптимизацию
            result = self.optimizer.newton_method(**params)

            # Check if the result contains an error message
            if len(result) == 3:
                final_point, iterations, error_message = result
                result_text = f"Ошибка: {error_message}"  # Display error
            else:
                final_point, iterations = result  # Unpack result
                # Формируем текст результатов
                result_text = (
                    f"Функция: f(x1,x2) = 4*x1² + 0.5*x1*x2 + 2*x2²\n\n"
                    f"Конечная точка:\n"
                    f"x1 = {final_point[0]:.5f}, x2 = {final_point[1]:.5f}\n\n"
                    f"Значение функции в точке: {self.optimizer.func(final_point):.5f}\n\n"
                    f"Количество итераций: {iterations}\n"
                )

            # Выводим результаты
            self.result_display.setPlainText(result_text)

            # Обновляем график
            self.plot_widget.plot_optimization(
                self.optimizer.history,
                self.optimizer.func
            )

        except Exception as e: #Catch any other unexpected errors
            self.result_display.setPlainText(f"An unexpected error occurred: {str(e)}")

    def clear_results(self):
        """Очистка результатов и графика"""
        self.result_display.clear()
        self.plot_widget.ax.clear()
        self.plot_widget.canvas.draw()