import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def visualize_integral(func, a, b, method, n=100):
    try:
        x = np.linspace(a, b, 1000)
        y = [func(xi) for xi in x]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, label='Function')
        ax.fill_between(x, y, alpha=0.3)

        if method == 'riemann':
            dx = (b - a) / n
            x_riemann = np.linspace(a, b, n+1)
            y_riemann = [func(xi) for xi in x_riemann]
            ax.bar(x_riemann[:-1], y_riemann[:-1], width=dx, alpha=0.5, align='edge', color='r', label='Riemann sum')
        elif method == 'lebesgue':
            # For Lebesgue, we'll just show the function itself
            ax.set_ylim(-0.1, 1.1)
            if "1 if is_rational(x) else 0" in str(func):
                ax.text(0.5, 0.5, "Lebesgue integral of Dirichlet function = 0\n(Measure of rationals is 0)", 
                        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.set_title(f'Integral of function using {method.capitalize()} method')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.grid(True, alpha=0.3)

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)

        return plot_url
    except Exception as e:
        print(f"Error in visualize_integral: {str(e)}")
        return None