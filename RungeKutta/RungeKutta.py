import numpy as np
import matplotlib.pyplot as plt

class DifferentialEquation:
    """Класс для описания дифференциального уравнения второго порядка"""
    def __init__(self, func):
        self.func = func  # func(x, y, z) где z = dy/dx
    
    def evaluate(self, x, y, z):
        return self.func(x, y, z)

class RungeKuttaSolver:
    """Решатель дифференциальных уравнений методом Рунге-Кутты 4-го порядка"""
    def __init__(self, equation):
        self.equation = equation
    
    def solve(self, x0, y0, z0, x_end, step_size):
        """
        Решение уравнения на интервале [x0, x_end]
        
        Parameters:
        x0, y0: начальные условия
        z0: начальное значение производной (dy/dx)
        x_end: конечная точка
        step_size: шаг интегрирования
        """
        # Создаем массивы для результатов
        x_values = np.arange(x0, x_end + step_size, step_size)
        y_values = np.zeros(len(x_values))
        z_values = np.zeros(len(x_values))
        
        # Устанавливаем начальные условия
        y_values[0] = y0
        z_values[0] = z0
        
        # Метод Рунге-Кутты 4-го порядка
        for i in range(len(x_values) - 1):
            x = x_values[i]
            y = y_values[i]
            z = z_values[i]
            h = step_size
            
            # Вычисляем коэффициенты для y (позиция)
            k1_y = h * z
            k1_z = h * self.equation.evaluate(x, y, z)
            
            k2_y = h * (z + 0.5 * k1_z)
            k2_z = h * self.equation.evaluate(x + 0.5*h, y + 0.5*k1_y, z + 0.5*k1_z)
            
            k3_y = h * (z + 0.5 * k2_z)
            k3_z = h * self.equation.evaluate(x + 0.5*h, y + 0.5*k2_y, z + 0.5*k2_z)
            
            k4_y = h * (z + k3_z)
            k4_z = h * self.equation.evaluate(x + h, y + k3_y, z + k3_z)
            
            # Обновляем значения
            y_values[i+1] = y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
            z_values[i+1] = z + (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        
        return x_values, y_values, z_values

class ResultVisualizer:
    """Класс для визуализации результатов решения"""
    @staticmethod
    def plot_solution(x_values, y_values, z_values=None, title="Решение ДУ"):
        fig, axes = plt.subplots(1, 2 if z_values is not None else 1, 
                               figsize=(12, 5))
        
        if z_values is not None:
            # График решения y(x)
            axes[0].plot(x_values, y_values, 'b-', linewidth=2, label='y(x)')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y(x)')
            axes[0].set_title('Решение дифференциального уравнения')
            axes[0].grid(True)
            axes[0].legend()
            
            # График производной z(x) = dy/dx
            axes[1].plot(x_values, z_values, 'r-', linewidth=2, label="dy/dx")
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('dy/dx')
            axes[1].set_title('Производная решения')
            axes[1].grid(True)
            axes[1].legend()
            
            # Фазовый портрет
            plt.figure(figsize=(8, 6))
            plt.plot(y_values, z_values, 'g-', linewidth=2)
            plt.xlabel('y')
            plt.ylabel('dy/dx')
            plt.title('Фазовый портрет')
            plt.grid(True)
            
        else:
            axes.plot(x_values, y_values, 'b-', linewidth=2, label='y(x)')
            axes.set_xlabel('x')
            axes.set_ylabel('y(x)')
            axes.set_title(title)
            axes.grid(True)
            axes.legend()
        
        plt.tight_layout()
        plt.show()

# Пример использования: уравнение гармонического осциллятора
def harmonic_oscillator(x, y, z):
    """
    Уравнение гармонического осциллятора: y'' + ω²y = 0
    Преобразуем к системе: 
    dy/dx = z
    dz/dx = -ω²y
    """
    omega = 2.0  # частота
    return -omega**2 * y

def main():
    # Создаем уравнение
    equation = DifferentialEquation(harmonic_oscillator)
    
    # Создаем решатель
    solver = RungeKuttaSolver(equation)
    
    # Начальные условия
    x0 = 0.0      # начальная координата x
    y0 = 1.0      # начальное значение y
    z0 = 0.0      # начальное значение производной dy/dx
    x_end = 10.0  # конечная точка
    h = 0.01      # шаг
    
    # Решаем уравнение
    x_vals, y_vals, z_vals = solver.solve(x0, y0, z0, x_end, h)
    
    # Визуализируем результаты
    visualizer = ResultVisualizer()
    visualizer.plot_solution(x_vals, y_vals, z_vals, "Гармонический осциллятор")
    
    # Выводим информацию о решении
    print(f"Решение найдено на интервале [{x0}, {x_end}]")
    print(f"Шаг интегрирования: {h}")
    print(f"Количество точек: {len(x_vals)}")
    print(f"y({x_end}) = {y_vals[-1]:.6f}")
    print(f"dy/dx({x_end}) = {z_vals[-1]:.6f}")

if __name__ == "__main__":
    main()